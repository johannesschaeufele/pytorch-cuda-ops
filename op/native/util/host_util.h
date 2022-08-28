#define CHECK_DIMENSIONS(x, d) TORCH_CHECK(x.sizes().size() == d, #x " must have exactly " #d " dimension(s)")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_INPUT_TENSOR_BASE(x, d, contiguous) \
    CHECK_DIMENSIONS(x, d)                        \
    if(contiguous) {                              \
        CHECK_CONTIGUOUS(x)                       \
    }                                             \
    CHECK_CUDA(x)

#define CHECK_INPUT_TENSOR(x, d) CHECK_INPUT_TENSOR_BASE(x, d, true)
#define CHECK_INPUT_TENSOR_NC(x, d) CHECK_INPUT_TENSOR_BASE(x, d, false)
