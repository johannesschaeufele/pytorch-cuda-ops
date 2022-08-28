#include <torch/types.h>
#include <ATen/ATen.h>

#include <type_traits>

template<typename scalar_t>
static __forceinline__ __device__ scalar_t warpReduceSum(scalar_t val) {
    using T = typename std::conditional_t<std::is_same_v<scalar_t, at::Half>, __half, scalar_t>;

    for(int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, static_cast<T>(val), offset);
    }

    return val;
}

template<typename T>
constexpr T constexpr_max(T a, T b) {
    return a < b ? b : a;
}

template<typename T>
constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}
