#include "util/host_util.h"

#include <torch/extension.h>

using torch::Tensor;

Tensor involution_cuda(const Tensor &input, const Tensor &weight);
std::tuple<Tensor, Tensor> involution_backward_cuda(const Tensor &grad_output, const Tensor &input, const Tensor &weight);

Tensor involution(const Tensor &input, const Tensor &weight) {
    CHECK_INPUT_TENSOR(input, 4);
    CHECK_INPUT_TENSOR(weight, 4);

    return involution_cuda(input, weight);
}

std::tuple<Tensor, Tensor> involution_bw(const Tensor &grad_output, const Tensor &input, const Tensor &weight) {
    CHECK_INPUT_TENSOR(grad_output, 4);
    CHECK_INPUT_TENSOR(input, 4);
    CHECK_INPUT_TENSOR(weight, 4);

    return involution_backward_cuda(grad_output, input, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("involution", &involution, "involution forward (CUDA)");
    m.def("involution_bw", &involution_bw, "involution backward (CUDA)");
}
