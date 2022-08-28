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

constexpr double alpha = GSMELU_ALPHA;
constexpr double beta = GSMELU_BETA;
constexpr double t = GSMELU_T;
constexpr double slope = GSMELU_SLOPE;
constexpr double neg_slope = GSMELU_NEG_SLOPE;

constexpr double a = (slope - neg_slope) / (2.0 * (alpha + beta));
constexpr double b = (alpha * slope + beta * neg_slope) / (alpha + beta);
constexpr double c = t + (alpha * alpha * (slope + neg_slope) + 2.0 * alpha * beta * neg_slope) / (2.0 * (alpha + beta));

template<typename scalar_t>
static __forceinline__ __device__ scalar_t gsmelu_fun(scalar_t x) {
    scalar_t alpha_ = static_cast<scalar_t>(alpha);
    scalar_t beta_ = static_cast<scalar_t>(beta);
    scalar_t t_ = static_cast<scalar_t>(t);
    scalar_t slope_ = static_cast<scalar_t>(slope);
    scalar_t neg_slope_ = static_cast<scalar_t>(neg_slope);
    scalar_t a_ = static_cast<scalar_t>(a);
    scalar_t b_ = static_cast<scalar_t>(b);
    scalar_t c_ = static_cast<scalar_t>(c);

    if(x <= -alpha_) {
        return neg_slope_ * x + t_ + static_cast<scalar_t>(neg_slope * alpha);
    }
    else if(x <= beta_) {
        return a_ * x * x + b_ * x + c_;
    }
    else {
        return slope_ * x + t_ + static_cast<scalar_t>((alpha + beta) / 2.0 * neg_slope + (alpha - beta) / 2.0 * slope);
    }
}

template<typename scalar_t>
static __forceinline__ __device__ scalar_t gsmelu_fun_backward(scalar_t x) {
    scalar_t alpha_ = static_cast<scalar_t>(alpha);
    scalar_t beta_ = static_cast<scalar_t>(beta);
    scalar_t slope_ = static_cast<scalar_t>(slope);
    scalar_t neg_slope_ = static_cast<scalar_t>(neg_slope);
    scalar_t b_ = static_cast<scalar_t>(b);

    if(x <= -alpha_) {
        return neg_slope_;
    }
    else if(x <= beta_) {
        return static_cast<scalar_t>(2.0 * a) * x + b_;
    }
    else {
        return slope_;
    }
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gsmelu_kernel(const int nthreads, TensorInfo<scalar_t, int> input, TensorInfo<scalar_t, int> output) {
    const int inp_sN = input.strides[0];
    const int out_sN = output.strides[0];

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int n = index;

        const scalar_t *const __restrict__ inp_ptr = input.data + n * inp_sN;
        scalar_t *const __restrict__ out_ptr = output.data + n * out_sN;

        *out_ptr = gsmelu_fun(*inp_ptr);
    }
}

template<typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gsmelu_backward_input_kernel(const int nthreads, TensorInfo<scalar_t, int> grad_output, TensorInfo<scalar_t, int> input,
                                             TensorInfo<scalar_t, int> grad_input // initialized to empty
) {
    const int inp_sN = input.strides[0];
    const int gOut_sN = grad_output.strides[0];
    const int gInp_sN = grad_input.strides[0];

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int n = index;

        const scalar_t *const __restrict__ gOut_ptr = grad_output.data + n * gOut_sN;
        const scalar_t *const __restrict__ inp_ptr = input.data + n * inp_sN;
        scalar_t *const __restrict__ gInp_ptr = grad_input.data + n * gInp_sN;

        *gInp_ptr = gsmelu_fun_backward(*inp_ptr) * (*gOut_ptr);
    }
}

Tensor gsmelu_cuda(const Tensor &input) {
    auto N = input.size(0);
    auto output = at::empty({N}, input.options());

    int count = static_cast<int>(N);
    if(count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "gsmelu_cuda", [&] {
            gsmelu_kernel<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(count, getTensorInfo<scalar_t, int>(input),
                                                                                                                  getTensorInfo<scalar_t, int>(output));
        });
    }

    return output;
}

Tensor gsmelu_backward_cuda(const Tensor &grad_output, const Tensor &input) {
    auto N = grad_output.size(0);

    auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int count = static_cast<int>(N);
    if(count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "gsmelu_backward_input_cuda", [&] {
            gsmelu_backward_input_kernel<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count, getTensorInfo<scalar_t, int>(grad_output), getTensorInfo<scalar_t, int>(input), getTensorInfo<scalar_t, int>(grad_input));
        });
    }

    return grad_input;
}
