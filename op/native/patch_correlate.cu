#include "util/device_util.cuh"

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

#include <c10/macros/Macros.h>

#include <cuda.h>
#include <cuda_runtime.h>

using torch::Tensor;

using namespace at::cuda::detail;

#if PATCH_CORRELATE_L1
constexpr bool use_l1 = true;
#else
constexpr bool use_l1 = false;
#endif

constexpr int C = PATCH_CORRELATE_FEATURE_DIM;
constexpr int CHANNEL_STRIDE = (C / C10_WARP_SIZE) * C10_WARP_SIZE == C ? C10_WARP_SIZE : 1;

constexpr int R = PATCH_CORRELATE_RADIUS;
constexpr int L = 2 * R + 1;
constexpr int AREA = L * L;

template<typename scalar_t>
static __forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
    // -100.0 does not have special meaning. This is just to make sure
    // it's not within_bounds_2d or within_bounds_3d, and does not cause
    // undefined behavior.
    if(x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
        return static_cast<scalar_t>(-100.0);
    return x;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

template<typename scalar_t>
static __forceinline__ __device__ void unsafe_add_2d(scalar_t *data, int h, int w, int sH, int sW, scalar_t delta) {
    gpuAtomicAdd(data + h * sH + w * sW, delta);
}

template<typename scalar_t>
static __forceinline__ __device__ void safe_add_2d(scalar_t *data, int h, int w, int sH, int sW, int H, int W, scalar_t delta) {
    if(within_bounds_2d(h, w, H, W)) {
        unsafe_add_2d(data, h, w, sH, sW, delta);
    }
}

template<typename scalar_t>
static __forceinline__ __device__ scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void patch_correlate_kernel(const int nthreads, TensorInfo<scalar_t, int> base, TensorInfo<scalar_t, int> input, TensorInfo<scalar_t, int> grid,
                                       TensorInfo<scalar_t, int> output) {
    const int inp_H = input.sizes[2];
    const int inp_W = input.sizes[3];
    const int out_H = grid.sizes[1];
    const int out_W = grid.sizes[2];
    const int base_sN = base.strides[0];
    const int base_sC = base.strides[1];
    const int base_sH = base.strides[2];
    const int base_sW = base.strides[3];
    const int inp_sN = input.strides[0];
    const int inp_sC = input.strides[1];
    const int inp_sH = input.strides[2];
    const int inp_sW = input.strides[3];
    const int grid_sN = grid.strides[0];
    const int grid_sH = grid.strides[1];
    const int grid_sW = grid.strides[2];
    const int grid_sCoor = grid.strides[3];
    const int out_sN = output.strides[0];
    const int out_sC = output.strides[1];
    const int out_sH = output.strides[2];
    const int out_sW = output.strides[3];

    const scalar_t fac = use_l1 ? static_cast<scalar_t>(C) : static_cast<scalar_t>(::rsqrt(static_cast<scalar_t>(C)));

    const scalar_t *const __restrict__ base_data = base.data;
    const scalar_t *const __restrict__ input_data = input.data;
    const scalar_t *const __restrict__ grid_data = grid.data;
    scalar_t *const __restrict__ output_data = output.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int ci = index % CHANNEL_STRIDE;

        const int w = (index / CHANNEL_STRIDE) % out_W;
        const int h = (index / (out_W * CHANNEL_STRIDE)) % out_H;

        const int n = index / (out_H * out_W * CHANNEL_STRIDE);

        int ww = w;
        int hh = h;

        const int grid_offset = n * grid_sN + hh * grid_sH + ww * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        scalar_t ix_ = grid_data[grid_offset];
        scalar_t iy_ = grid_data[grid_offset + grid_sCoor];

        scalar_t ix_base = safe_downgrade_to_int_range(ix_);
        scalar_t iy_base = safe_downgrade_to_int_range(iy_);

        for(int dy = -R; dy <= R; dy++) {
            for(int dx = -R; dx <= R; dx++) {
                scalar_t ix = ix_base + static_cast<scalar_t>(dx);
                scalar_t iy = iy_base + static_cast<scalar_t>(dy);

                // get NE, NW, SE, SW pixel values from (x, y)
                int ix_nw = static_cast<int>(::floor(ix));
                int iy_nw = static_cast<int>(::floor(iy));
                int ix_ne = ix_nw + 1;
                int iy_ne = iy_nw;
                int ix_sw = ix_nw;
                int iy_sw = iy_nw + 1;
                int ix_se = ix_nw + 1;
                int iy_se = iy_nw + 1;

                // get surfaces to each neighbor
                scalar_t nw = (ix_se - ix) * (iy_se - iy);
                scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                // calculate bilinear weighted pixel value and set output pixel
                auto inp_ptr_NC = input_data + n * inp_sN + ci * inp_sC;
                auto base_ptr_NCHW = base_data + n * base_sN + h * base_sH + w * base_sW + ci * base_sC;
                auto out_ptr_NCHW = output_data + n * out_sN + h * out_sH + w * out_sW + ((dx + R) * L + (dy + R)) * out_sC;
                scalar_t cv = static_cast<scalar_t>(0);
                for(int c = ci; c < C; c += CHANNEL_STRIDE, inp_ptr_NC += CHANNEL_STRIDE * inp_sC, base_ptr_NCHW += CHANNEL_STRIDE * base_sC) {
                    scalar_t cc = static_cast<scalar_t>(0);
                    if(within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                        cc += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                    }
                    if(within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                        cc += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                    }
                    if(within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                        cc += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                    }
                    if(within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                        cc += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                    }

                    scalar_t base_value = *base_ptr_NCHW;

                    if(use_l1) {
                        cv += ::abs(base_value - cc);
                    }
                    else {
                        cv += base_value * cc;
                    }
                }

                if(CHANNEL_STRIDE != 1) {
                    __syncwarp();
                    cv = warpReduceSum(cv);
                }

                if(ci == 0) {
                    *out_ptr_NCHW = cv * fac;
                }
            }
        }
    }
}

template<typename scalar_t, bool do_base, bool do_grid>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void patch_correlate_backward_base_grid_kernel(const int nthreads, TensorInfo<scalar_t, int> grad_output, TensorInfo<scalar_t, int> base,
                                                          TensorInfo<scalar_t, int> input, TensorInfo<scalar_t, int> grid,
                                                          TensorInfo<scalar_t, int> grad_base, // initialized to zeros
                                                          TensorInfo<scalar_t, int> grad_grid // initialized to zeros
) {
    const int inp_H = input.sizes[2];
    const int inp_W = input.sizes[3];
    const int out_H = grid.sizes[1];
    const int out_W = grid.sizes[2];
    const int base_sN = base.strides[0];
    const int base_sC = base.strides[1];
    const int base_sH = base.strides[2];
    const int base_sW = base.strides[3];
    const int inp_sN = input.strides[0];
    const int inp_sC = input.strides[1];
    const int inp_sH = input.strides[2];
    const int inp_sW = input.strides[3];
    const int grid_sN = grid.strides[0];
    const int grid_sH = grid.strides[1];
    const int grid_sW = grid.strides[2];
    const int grid_sCoor = grid.strides[3];
    const int gOut_sN = grad_output.strides[0];
    const int gOut_sC = grad_output.strides[1];
    const int gOut_sH = grad_output.strides[2];
    const int gOut_sW = grad_output.strides[3];
    const int gBase_sN = grad_base.strides[0];
    const int gBase_sC = grad_base.strides[1];
    const int gBase_sH = grad_base.strides[2];
    const int gBase_sW = grad_base.strides[3];
    const int gGrid_sN = grad_grid.strides[0];
    const int gGrid_sH = grad_grid.strides[1];
    const int gGrid_sW = grad_grid.strides[2];
    const int gGrid_sCoor = grad_grid.strides[3];

    const scalar_t fac = use_l1 ? static_cast<scalar_t>(C) : static_cast<scalar_t>(::rsqrt(static_cast<scalar_t>(C)));

    const scalar_t *const __restrict__ grad_output_data = grad_output.data;
    const scalar_t *const __restrict__ base_data = base.data;
    const scalar_t *const __restrict__ input_data = input.data;
    const scalar_t *const __restrict__ grid_data = grid.data;
    scalar_t *const __restrict__ grad_base_data = grad_base.data;
    scalar_t *const __restrict__ grad_grid_data = grad_grid.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int ci = index % CHANNEL_STRIDE;

        const int w = (index / CHANNEL_STRIDE) % out_W;
        const int h = (index / (out_W * CHANNEL_STRIDE)) % out_H;

        const int n = index / (out_H * out_W * CHANNEL_STRIDE);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);

        int ww = w;
        int hh = h;

        const int grid_offset = n * grid_sN + hh * grid_sH + ww * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        scalar_t ix_ = grid_data[grid_offset];
        scalar_t iy_ = grid_data[grid_offset + grid_sCoor];

        scalar_t ix_base = safe_downgrade_to_int_range(ix_);
        scalar_t iy_base = safe_downgrade_to_int_range(iy_);

        for(int dy = -R; dy <= R; dy++) {
            for(int dx = -R; dx <= R; dx++) {
                scalar_t ix = ix_base + static_cast<scalar_t>(dx);
                scalar_t iy = iy_base + static_cast<scalar_t>(dy);

                auto gOut_ptr_NCHW = grad_output_data + n * gOut_sN + h * gOut_sH + w * gOut_sW + ((dx + R) * L + (dy + R)) * gOut_sC;
                scalar_t gOut = *gOut_ptr_NCHW * fac;
                auto inp_ptr_NC = input_data + n * inp_sN + ci * inp_sC;
                auto base_ptr_NCHW = base_data + n * base_sN + h * base_sH + w * base_sW + ci * base_sC;
                auto gBase_ptr_NCHW = grad_base_data + n * gBase_sN + h * gBase_sH + w * gBase_sW + ci * gBase_sC;
                for(int c = ci; c < C; c += CHANNEL_STRIDE, inp_ptr_NC += CHANNEL_STRIDE * inp_sC, base_ptr_NCHW += CHANNEL_STRIDE * base_sC,
                        gBase_ptr_NCHW += CHANNEL_STRIDE * gBase_sC) {
                    // get NE, NW, SE, SW pixel values from (x, y)
                    int ix_nw = static_cast<int>(::floor(ix));
                    int iy_nw = static_cast<int>(::floor(iy));
                    int ix_ne = ix_nw + 1;
                    int iy_ne = iy_nw;
                    int ix_sw = ix_nw;
                    int iy_sw = iy_nw + 1;
                    int ix_se = ix_nw + 1;
                    int iy_se = iy_nw + 1;

                    // get surfaces to each neighbor
                    scalar_t nw = (ix_se - ix) * (iy_se - iy);
                    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                    scalar_t cc = static_cast<scalar_t>(0);
                    scalar_t gixc = static_cast<scalar_t>(0), giyc = static_cast<scalar_t>(0);

                    // calculate grad_grid
                    if(within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                        scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                        gixc -= nw_val * (iy_se - iy);
                        giyc -= nw_val * (ix_se - ix);
                        cc += nw_val * nw;
                    }
                    if(within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                        scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                        gixc += ne_val * (iy_sw - iy);
                        giyc -= ne_val * (ix - ix_sw);
                        cc += ne_val * ne;
                    }
                    if(within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                        scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                        gixc -= sw_val * (iy - iy_ne);
                        giyc += sw_val * (ix_ne - ix);
                        cc += sw_val * sw;
                    }
                    if(within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                        scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                        gixc += se_val * (iy - iy_nw);
                        giyc += se_val * (ix - ix_nw);
                        cc += se_val * se;
                    }

                    const scalar_t base_value = *base_ptr_NCHW;

                    scalar_t gFac;
                    scalar_t gBase;
                    if(use_l1) {
                        scalar_t gAbs = sign(base_value - cc);

                        gBase = gAbs * gOut;
                        gFac = -gBase;
                    }
                    else {
                        gBase = cc * gOut;
                        gFac = base_value * gOut;
                    }

                    if(do_base) {
                        *gBase_ptr_NCHW += gBase;
                    }

                    gix += gixc * gFac;
                    giy += giyc * gFac;
                }
            }
        }

        if(do_grid) {
            if(CHANNEL_STRIDE != 1) {
                __syncwarp();
                gix = warpReduceSum(gix);
                giy = warpReduceSum(giy);
            }

            if(ci == 0) {
                auto gGrid_ptr_NHW = grad_grid_data + n * gGrid_sN + h * gGrid_sH + w * gGrid_sW;
                gGrid_ptr_NHW[0] = gix;
                gGrid_ptr_NHW[gGrid_sCoor] = giy;
            }
        }
    }
}

template<bool do_base, bool do_grid>
void patch_correlate_backward_base_grid_wrapper(int count, const Tensor &grad_output, const Tensor &base, const Tensor &input, const Tensor &grid,
                                                Tensor &grad_base, Tensor &grad_grid) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "patch_correlate_backward_base_grid_cuda", [&] {
        patch_correlate_backward_base_grid_kernel<scalar_t, do_base, do_grid><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count, getTensorInfo<scalar_t, int>(grad_output), getTensorInfo<scalar_t, int>(base), getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid), getTensorInfo<scalar_t, int>(grad_base), getTensorInfo<scalar_t, int>(grad_grid));
    });
}

void patch_correlate_backward_base_grid(int count, const Tensor &grad_output, const Tensor &base, const Tensor &input, const Tensor &grid, Tensor &grad_base,
                                        Tensor &grad_grid, bool do_base, bool do_grid) {
    if(do_base) {
        if(do_grid) {
            patch_correlate_backward_base_grid_wrapper<true, true>(count, grad_output, base, input, grid, grad_base, grad_grid);
        }
        else {
            patch_correlate_backward_base_grid_wrapper<true, false>(count, grad_output, base, input, grid, grad_base, grad_grid);
        }
    }
    else {
        if(do_grid) {
            patch_correlate_backward_base_grid_wrapper<false, true>(count, grad_output, base, input, grid, grad_base, grad_grid);
        }
        else {
            // no-op
        }
    }
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void patch_correlate_backward_input_kernel(const int nthreads, TensorInfo<scalar_t, int> grad_output, TensorInfo<scalar_t, int> base,
                                                      TensorInfo<scalar_t, int> input, TensorInfo<scalar_t, int> grid,
                                                      TensorInfo<scalar_t, int> grad_input // initialized to zeros
) {
    const int inp_H = input.sizes[2];
    const int inp_W = input.sizes[3];
    const int out_H = grid.sizes[1];
    const int out_W = grid.sizes[2];
    const int base_sN = base.strides[0];
    const int base_sC = base.strides[1];
    const int base_sH = base.strides[2];
    const int base_sW = base.strides[3];
    const int inp_sN = input.strides[0];
    const int inp_sC = input.strides[1];
    const int inp_sH = input.strides[2];
    const int inp_sW = input.strides[3];
    const int grid_sN = grid.strides[0];
    const int grid_sH = grid.strides[1];
    const int grid_sW = grid.strides[2];
    const int grid_sCoor = grid.strides[3];
    const int gOut_sN = grad_output.strides[0];
    const int gOut_sC = grad_output.strides[1];
    const int gOut_sH = grad_output.strides[2];
    const int gOut_sW = grad_output.strides[3];
    const int gInp_sN = grad_input.strides[0];
    const int gInp_sC = grad_input.strides[1];
    const int gInp_sH = grad_input.strides[2];
    const int gInp_sW = grad_input.strides[3];

    const scalar_t fac = use_l1 ? static_cast<scalar_t>(C) : static_cast<scalar_t>(::rsqrt(static_cast<scalar_t>(C)));

    const scalar_t *const __restrict__ grad_output_data = grad_output.data;
    const scalar_t *const __restrict__ base_data = base.data;
    const scalar_t *const __restrict__ input_data = input.data;
    const scalar_t *const __restrict__ grid_data = grid.data;
    scalar_t *const __restrict__ grad_input_data = grad_input.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int ci = index % CHANNEL_STRIDE;

        const int w = (index / CHANNEL_STRIDE) % out_W;
        const int h = (index / (out_W * CHANNEL_STRIDE)) % out_H;

        const int n = index / (out_H * out_W * CHANNEL_STRIDE);

        int ww = w;
        int hh = h;

        const int grid_offset = n * grid_sN + hh * grid_sH + ww * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        scalar_t ix_ = grid_data[grid_offset];
        scalar_t iy_ = grid_data[grid_offset + grid_sCoor];

        scalar_t ix_base = safe_downgrade_to_int_range(ix_);
        scalar_t iy_base = safe_downgrade_to_int_range(iy_);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw_ = static_cast<int>(::floor(ix_base));
        int iy_nw_ = static_cast<int>(::floor(iy_base));
        int ix_ne = ix_nw_ + 1;
        int iy_ne = iy_nw_;
        int ix_sw = ix_nw_;
        int iy_sw = iy_nw_ + 1;
        int ix_se = ix_nw_ + 1;
        int iy_se = iy_nw_ + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix_base) * (iy_se - iy_base);
        scalar_t ne = (ix_base - ix_sw) * (iy_sw - iy_base);
        scalar_t sw = (ix_ne - ix_base) * (iy_base - iy_ne);
        scalar_t se = (ix_base - ix_nw_) * (iy_base - iy_nw_);

        for(int dy = -R; dy <= R + 1; dy++) {
            for(int dx = -R; dx <= R + 1; dx++) {
                scalar_t ix = ix_base + static_cast<scalar_t>(dx);
                scalar_t iy = iy_base + static_cast<scalar_t>(dy);

                auto gOut_se = (dx > -R && dy > -R) ? grad_output_data[n * gOut_sN + h * gOut_sH + w * gOut_sW + ((dx - 1 + R) * L + (dy - 1 + R)) * gOut_sC] :
                                                      static_cast<scalar_t>(0);
                auto gOut_sw = (dx <= R && dy > -R) ? grad_output_data[n * gOut_sN + h * gOut_sH + w * gOut_sW + ((dx + R) * L + (dy - 1 + R)) * gOut_sC] :
                                                      static_cast<scalar_t>(0);
                auto gOut_ne = (dx > -R && dy <= R) ? grad_output_data[n * gOut_sN + h * gOut_sH + w * gOut_sW + ((dx - 1 + R) * L + (dy + R)) * gOut_sC] :
                                                      static_cast<scalar_t>(0);
                auto gOut_nw = (dx <= R && dy <= R) ? grad_output_data[n * gOut_sN + h * gOut_sH + w * gOut_sW + ((dx + R) * L + (dy + R)) * gOut_sC] :
                                                      static_cast<scalar_t>(0);
                auto inp_ptr_NC = input_data + n * inp_sN + ci * inp_sC;
                auto gInp_ptr_NC = grad_input_data + n * gInp_sN + ci * gInp_sC;
                auto base_ptr_NCHW = base_data + n * base_sN + h * base_sH + w * base_sW + ci * base_sC;
                for(int c = ci; c < C; c += CHANNEL_STRIDE, inp_ptr_NC += CHANNEL_STRIDE * inp_sC, gInp_ptr_NC += CHANNEL_STRIDE * gInp_sC,
                        base_ptr_NCHW += CHANNEL_STRIDE * base_sC) {
                    // get NE, NW, SE, SW pixel values from (x, y)
                    int ix_nw = ix_nw_ + dx;
                    int iy_nw = iy_nw_ + dy;
                    int ix_ne = ix_nw + 1;
                    int iy_ne = iy_nw;
                    int ix_sw = ix_nw;
                    int iy_sw = iy_nw + 1;
                    int ix_se = ix_nw + 1;
                    int iy_se = iy_nw + 1;

                    scalar_t cc_se = static_cast<scalar_t>(0);
                    if(dx > -R && dy > -R) {
                        if(within_bounds_2d(iy_nw - 1, ix_nw - 1, inp_H, inp_W)) {
                            scalar_t nw_val = inp_ptr_NC[(iy_nw - 1) * inp_sH + (ix_nw - 1) * inp_sW];
                            cc_se += nw_val * nw;
                        }
                        if(within_bounds_2d(iy_ne - 1, ix_ne - 1, inp_H, inp_W)) {
                            scalar_t ne_val = inp_ptr_NC[(iy_ne - 1) * inp_sH + (ix_ne - 1) * inp_sW];
                            cc_se += ne_val * ne;
                        }
                        if(within_bounds_2d(iy_sw - 1, ix_sw - 1, inp_H, inp_W)) {
                            scalar_t sw_val = inp_ptr_NC[(iy_sw - 1) * inp_sH + (ix_sw - 1) * inp_sW];
                            cc_se += sw_val * sw;
                        }
                        if(within_bounds_2d(iy_se - 1, ix_se - 1, inp_H, inp_W)) {
                            scalar_t se_val = inp_ptr_NC[(iy_se - 1) * inp_sH + (ix_se - 1) * inp_sW];
                            cc_se += se_val * se;
                        }
                    }

                    scalar_t cc_sw = static_cast<scalar_t>(0);
                    if(dx <= R && dy > -R) {
                        if(within_bounds_2d(iy_nw - 1, ix_nw, inp_H, inp_W)) {
                            scalar_t nw_val = inp_ptr_NC[(iy_nw - 1) * inp_sH + ix_nw * inp_sW];
                            cc_sw += nw_val * nw;
                        }
                        if(within_bounds_2d(iy_ne - 1, ix_ne, inp_H, inp_W)) {
                            scalar_t ne_val = inp_ptr_NC[(iy_ne - 1) * inp_sH + ix_ne * inp_sW];
                            cc_sw += ne_val * ne;
                        }
                        if(within_bounds_2d(iy_sw - 1, ix_sw, inp_H, inp_W)) {
                            scalar_t sw_val = inp_ptr_NC[(iy_sw - 1) * inp_sH + ix_sw * inp_sW];
                            cc_sw += sw_val * sw;
                        }
                        if(within_bounds_2d(iy_se - 1, ix_se, inp_H, inp_W)) {
                            scalar_t se_val = inp_ptr_NC[(iy_se - 1) * inp_sH + ix_se * inp_sW];
                            cc_sw += se_val * se;
                        }
                    }

                    scalar_t cc_ne = static_cast<scalar_t>(0);
                    if(dx > -R && dy <= R) {
                        if(within_bounds_2d(iy_nw, ix_nw - 1, inp_H, inp_W)) {
                            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + (ix_nw - 1) * inp_sW];
                            cc_ne += nw_val * nw;
                        }
                        if(within_bounds_2d(iy_ne, ix_ne - 1, inp_H, inp_W)) {
                            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + (ix_ne - 1) * inp_sW];
                            cc_ne += ne_val * ne;
                        }
                        if(within_bounds_2d(iy_sw, ix_sw - 1, inp_H, inp_W)) {
                            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + (ix_sw - 1) * inp_sW];
                            cc_ne += sw_val * sw;
                        }
                        if(within_bounds_2d(iy_se, ix_se - 1, inp_H, inp_W)) {
                            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + (ix_se - 1) * inp_sW];
                            cc_ne += se_val * se;
                        }
                    }

                    scalar_t cc_nw = static_cast<scalar_t>(0);
                    if(dx <= R && dy <= R) {
                        if(within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                            cc_nw += nw_val * nw;
                        }
                        if(within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                            cc_nw += ne_val * ne;
                        }
                        if(within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                            cc_nw += sw_val * sw;
                        }
                        if(within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                            cc_nw += se_val * se;
                        }
                    }

                    const scalar_t base_value = *base_ptr_NCHW;

                    scalar_t gFac_se;
                    scalar_t gFac_sw;
                    scalar_t gFac_ne;
                    scalar_t gFac_nw;
                    if(use_l1) {
                        scalar_t gAbs_se = sign(base_value - cc_se);
                        scalar_t gAbs_sw = sign(base_value - cc_sw);
                        scalar_t gAbs_ne = sign(base_value - cc_ne);
                        scalar_t gAbs_nw = sign(base_value - cc_nw);

                        gFac_se = -gAbs_se * gOut_se;
                        gFac_sw = -gAbs_sw * gOut_sw;
                        gFac_ne = -gAbs_ne * gOut_ne;
                        gFac_nw = -gAbs_nw * gOut_nw;
                    }
                    else {
                        gFac_se = base_value * gOut_se;
                        gFac_sw = base_value * gOut_sw;
                        gFac_ne = base_value * gOut_ne;
                        gFac_nw = base_value * gOut_nw;
                    }

                    // calculate and set grad_input
                    scalar_t gInp = (se * gFac_se + sw * gFac_sw + ne * gFac_ne + nw * gFac_nw) * fac;
                    safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, gInp);
                }
            }
        }
    }
}

Tensor patch_correlate_cuda(const Tensor &base, const Tensor &input, const Tensor &grid) {
    auto N = input.size(0);
    auto H = grid.size(1);
    auto W = grid.size(2);
    auto output = at::empty_like(at::empty({N, AREA, H, W}, input.options()), c10::MemoryFormat::ChannelsLast);

    int count = static_cast<int>(N * H * W * CHANNEL_STRIDE);
    if(count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "patch_correlate_cuda", [&] {
            patch_correlate_kernel<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count, getTensorInfo<scalar_t, int>(base), getTensorInfo<scalar_t, int>(input), getTensorInfo<scalar_t, int>(grid),
              getTensorInfo<scalar_t, int>(output));
        });
    }

    return output;
}

std::tuple<Tensor, Tensor, Tensor> patch_correlate_backward_cuda(const Tensor &grad_output, const Tensor &base, const Tensor &input, const Tensor &grid) {
    auto N = input.size(0);
    auto H = grid.size(1);
    auto W = grid.size(2);

    bool do_base = base.requires_grad();
    bool do_input = input.requires_grad();
    bool do_grid = grid.requires_grad();

    auto grad_base = do_base ? at::zeros_like(base, c10::MemoryFormat::ChannelsLast) : at::empty({1}, base.options());
    auto grad_input = do_input ? at::zeros_like(input, c10::MemoryFormat::ChannelsLast) : at::empty({1}, input.options());
    auto grad_grid = do_grid ? at::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::empty({1}, grid.options());

    int count = static_cast<int>(N * H * W * CHANNEL_STRIDE);
    if(count > 0) {
        if(do_base || do_grid) {
            patch_correlate_backward_base_grid(count, grad_output, base, input, grid, grad_base, grad_grid, do_base, do_grid);
        }

        if(do_input) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "patch_correlate_backward_input_cuda", [&] {
                patch_correlate_backward_input_kernel<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  count, getTensorInfo<scalar_t, int>(grad_output), getTensorInfo<scalar_t, int>(base), getTensorInfo<scalar_t, int>(input),
                  getTensorInfo<scalar_t, int>(grid), getTensorInfo<scalar_t, int>(grad_input));
            });
        }
    }
    return std::make_tuple(grad_base, grad_input, grad_grid);
}
