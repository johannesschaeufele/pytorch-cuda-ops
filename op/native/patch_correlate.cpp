#include "util/host_util.h"

#include <torch/extension.h>

using torch::Tensor;

Tensor patch_correlate_cuda(const Tensor &base, const Tensor &input, const Tensor &grid);
std::tuple<Tensor, Tensor, Tensor> patch_correlate_backward_cuda(const Tensor &grad_output, const Tensor &base, const Tensor &input, const Tensor &grid);

Tensor patch_correlate(const Tensor &base, const Tensor &input, const Tensor &grid) {
    CHECK_INPUT_TENSOR(base, 4);
    CHECK_INPUT_TENSOR(input, 4);
    CHECK_INPUT_TENSOR(grid, 4);

    return patch_correlate_cuda(base, input, grid);
}

std::tuple<Tensor, Tensor, Tensor> patch_correlate_bw(const Tensor &grad_output, const Tensor &base, const Tensor &input, const Tensor &grid) {
    CHECK_INPUT_TENSOR(base, 4);
    CHECK_INPUT_TENSOR(input, 4);
    CHECK_INPUT_TENSOR(grid, 4);

    return patch_correlate_backward_cuda(grad_output, base, input, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("patch_correlate", &patch_correlate, "patch_correlate forward (CUDA)");
    m.def("patch_correlate_bw", &patch_correlate_bw, "patch_correlate backward (CUDA)");
}
