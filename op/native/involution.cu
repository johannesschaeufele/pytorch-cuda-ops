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

constexpr int K = INVOLUTION_KERNEL_SIZE;
constexpr int KR = (INVOLUTION_KERNEL_SIZE - 1) / 2;
constexpr int KRH = INVOLUTION_KERNEL_SIZE / 2;

constexpr int SX = INVOLUTION_STRIDE_X;
constexpr int SY = INVOLUTION_STRIDE_Y;

constexpr int C = INVOLUTION_CHANNEL_COUNT;
constexpr int GROUPS = INVOLUTION_GROUP_COUNT;

constexpr int E = ((C + GROUPS - 1) / GROUPS);

// Separate function names for easier profiling and debugging
#define NAME_SUFFIX __##INVOLUTION_KERNEL_SIZE##_##INVOLUTION_STRIDE_X##_##INVOLUTION_STRIDE_Y##_##INVOLUTION_CHANNEL_COUNT##_##INVOLUTION_GROUP_COUNT

#define involution_kernel_with_suffix involution_kernel##NAME_SUFFIX
#define involution_backward_input_kernel_with_suffix involution_backward_input_kernel##NAME_SUFFIX
#define involution_backward_weight_kernel_with_suffix involution_backward_weight_kernel##NAME_SUFFIX

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__
  void involution_kernel_with_suffix(const int nthreads, TensorInfo<scalar_t, int> input, TensorInfo<scalar_t, int> weight, TensorInfo<scalar_t, int> output) {
    const int out_H = output.sizes[2];
    const int out_W = output.sizes[3];
    const int inp_sN = input.strides[0];
    const int inp_sC = input.strides[1];
    const int inp_sH = input.strides[2];
    const int inp_sW = input.strides[3];
    const int weight_sN = weight.strides[0];
    const int weight_sC = weight.strides[1];
    const int weight_sH = weight.strides[2];
    const int weight_sW = weight.strides[3];
    const int out_sN = output.strides[0];
    const int out_sC = output.strides[1];
    const int out_sH = output.strides[2];
    const int out_sW = output.strides[3];

    const scalar_t *const __restrict__ input_data = input.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % out_W;
        const int h = (index / out_W) % out_H;
        const int c = (index / (out_H * out_W)) % C;
        const int n = index / (C * out_H * out_W);

        scalar_t value = static_cast<scalar_t>(0);
        for(int dy = -KR; dy <= KRH; dy++) {
            for(int dx = -KR; dx <= KRH; dx++) {
                int x = SX * w + dx + KR;
                int y = SY * h + dy + KR;

                auto inp_ptr_NCHW = input_data + n * inp_sN + y * inp_sH + x * inp_sW + c * inp_sC;
                const scalar_t *const __restrict__ weight_ptr_NCHW =
                  weight.data + n * weight_sN + h * weight_sH + w * weight_sW + ((c / GROUPS) * (K * K) + (dy + KR) * K + (dx + KR)) * weight_sC;

                value += (*weight_ptr_NCHW) * (*inp_ptr_NCHW);
            }
        }

        scalar_t *const __restrict__ out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW + c * out_sC;
        *out_ptr_NCHW = value;
    }
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void involution_backward_input_kernel_with_suffix(const int nthreads, TensorInfo<scalar_t, int> grad_output, TensorInfo<scalar_t, int> weight,
                                                             TensorInfo<scalar_t, int> grad_input // initialized to empty
) {
    const int H = grad_input.sizes[2];
    const int W = grad_input.sizes[3];
    const int out_H = grad_output.sizes[2];
    const int out_W = grad_output.sizes[3];
    const int weight_sN = weight.strides[0];
    const int weight_sC = weight.strides[1];
    const int weight_sH = weight.strides[2];
    const int weight_sW = weight.strides[3];
    const int gOut_sN = grad_output.strides[0];
    const int gOut_sC = grad_output.strides[1];
    const int gOut_sH = grad_output.strides[2];
    const int gOut_sW = grad_output.strides[3];
    const int gInp_sN = grad_input.strides[0];
    const int gInp_sC = grad_input.strides[1];
    const int gInp_sH = grad_input.strides[2];
    const int gInp_sW = grad_input.strides[3];

    const scalar_t *const __restrict__ grad_output_data = grad_output.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % W;
        const int h = (index / W) % H;
        const int c = (index / (H * W)) % C;
        const int n = index / (C * H * W);

        scalar_t value = static_cast<scalar_t>(0);
        for(int dy = -KRH; dy <= KR; dy++) {
            for(int dx = -KRH; dx <= KR; dx++) {
                int x = w + dx - KR;
                int y = h + dy - KR;

                if(y >= 0 && (y < SY * out_H) && (y % SY == 0)) {
                    if(x >= 0 && (x < SX * out_W) && (x % SX == 0)) {
                        auto gOut_ptr_NCHW = grad_output_data + n * gOut_sN + (y / SY) * gOut_sH + (x / SX) * gOut_sW + c * gOut_sC;
                        const scalar_t *const __restrict__ weight_ptr_NCHW = weight.data + n * weight_sN + (y / SY) * weight_sH + (x / SX) * weight_sW +
                                                                             ((c / GROUPS) * (K * K) + (-dy + KR) * K + (-dx + KR)) * weight_sC;

                        value += (*weight_ptr_NCHW) * (*gOut_ptr_NCHW);
                    }
                }
            }
        }

        scalar_t *const __restrict__ gInp_ptr_NCHW = grad_input.data + n * gInp_sN + h * gInp_sH + w * gInp_sW + c * gInp_sC;
        *gInp_ptr_NCHW = value;
    }
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void involution_backward_weight_kernel_with_suffix(const int nthreads, TensorInfo<scalar_t, int> grad_output, TensorInfo<scalar_t, int> input,
                                                              TensorInfo<scalar_t, int> grad_weight // initialized to empty
) {
    const int out_H = grad_output.sizes[2];
    const int out_W = grad_output.sizes[3];
    const int inp_sN = input.strides[0];
    const int inp_sC = input.strides[1];
    const int inp_sH = input.strides[2];
    const int inp_sW = input.strides[3];
    const int gOut_sN = grad_output.strides[0];
    const int gOut_sC = grad_output.strides[1];
    const int gOut_sH = grad_output.strides[2];
    const int gOut_sW = grad_output.strides[3];
    const int gWeight_sN = grad_weight.strides[0];
    const int gWeight_sC = grad_weight.strides[1];
    const int gWeight_sH = grad_weight.strides[2];
    const int gWeight_sW = grad_weight.strides[3];

    const scalar_t *const __restrict__ input_data = input.data;

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % out_W;
        const int h = (index / out_W) % out_H;
        const int e = (index / (out_H * out_W)) % E;
        const int n = index / (E * out_H * out_W);

        for(int dy = -KR; dy <= KRH; dy++) {
            for(int dx = -KR; dx <= KRH; dx++) {
                int x = SX * w + dx + KR;
                int y = SY * h + dy + KR;

                scalar_t value = static_cast<scalar_t>(0);
                for(int ci = 0; ci < GROUPS; ci++) {
                    int c = GROUPS * e + ci;

                    if(c < C) {
                        const scalar_t *const __restrict__ gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW + c * gOut_sC;
                        scalar_t gOut = *gOut_ptr_NCHW;

                        auto inp_ptr_NCHW = input_data + n * inp_sN + y * inp_sH + x * inp_sW + c * inp_sC;
                        value += (*inp_ptr_NCHW) * gOut;
                    }

                    scalar_t *const __restrict__ gWeight_ptr_NCHW =
                      grad_weight.data + n * gWeight_sN + h * gWeight_sH + w * gWeight_sW + (e * (K * K) + (dy + KR) * K + (dx + KR)) * gWeight_sC;
                    *gWeight_ptr_NCHW = value;
                }
            }
        }
    }
}

Tensor involution_cuda(const Tensor &input, const Tensor &weight) {
    auto N = weight.size(0);
    auto H = weight.size(2);
    auto W = weight.size(3);
    auto output = at::empty({N, C, H, W}, input.options());

    int count = static_cast<int>(N * C * H * W);
    if(count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "involution_cuda", [&] {
            involution_kernel_with_suffix<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count, getTensorInfo<scalar_t, int>(input), getTensorInfo<scalar_t, int>(weight), getTensorInfo<scalar_t, int>(output));
        });
    }

    return output;
}

std::tuple<Tensor, Tensor> involution_backward_cuda(const Tensor &grad_output, const Tensor &input, const Tensor &weight) {
    auto N = weight.size(0);
    auto H = weight.size(2);
    auto W = weight.size(3);

    auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto grad_weight = at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int count = static_cast<int>(N * C * H * W);
    if(count > 0) {
        int count_input = static_cast<int>(N * C * input.size(2) * input.size(3));
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "involution_backward_input_cuda", [&] {
            involution_backward_input_kernel_with_suffix<scalar_t><<<GET_BLOCKS(count_input), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count_input, getTensorInfo<scalar_t, int>(grad_output), getTensorInfo<scalar_t, int>(weight), getTensorInfo<scalar_t, int>(grad_input));
        });

        int count_weight = static_cast<int>(N * E * H * W);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "involution_backward_weight_cuda", [&] {
            involution_backward_weight_kernel_with_suffix<scalar_t><<<GET_BLOCKS(count_weight), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count_weight, getTensorInfo<scalar_t, int>(grad_output), getTensorInfo<scalar_t, int>(input), getTensorInfo<scalar_t, int>(grad_weight));
        });
    }

    return std::make_tuple(grad_input, grad_weight);
}
