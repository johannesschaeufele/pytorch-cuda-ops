#include "util/host_util.h"

#include <torch/extension.h>

using torch::Tensor;

Tensor gsmelu_cuda(const Tensor &input);
Tensor gsmelu_backward_cuda(const Tensor &grad_output, const Tensor &input);

Tensor gsmelu(const Tensor &input) {
    CHECK_INPUT_TENSOR(input, 1);

    return gsmelu_cuda(input);
}

Tensor gsmelu_bw(const Tensor &grad_output, const Tensor &input) {
    CHECK_INPUT_TENSOR(grad_output, 1);
    CHECK_INPUT_TENSOR(input, 1);

    return gsmelu_backward_cuda(grad_output, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gsmelu", &gsmelu, "gsmelu forward (CUDA)");
    m.def("gsmelu_bw", &gsmelu_bw, "gsmelu backward (CUDA)");
}
