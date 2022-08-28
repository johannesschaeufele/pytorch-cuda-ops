import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

from pathlib import Path

import threading

involution_lock = threading.Lock()
involution_op_cache = {}


def get_involution_op(kernel_size, channels, groups=1, stride_x=1, stride_y=1):
    assert kernel_size >= 1
    assert channels >= 1
    assert groups >= 1
    assert groups <= channels
    assert stride_x >= 1
    assert stride_y >= 1

    with involution_lock:
        key = "involution_" + str(kernel_size) + "_" + str(stride_x) + "_" + str(stride_y) + "_" + str(channels) + "_" + str(groups)
        if key in involution_op_cache:
            return involution_op_cache[key]
        else:
            print("Compiling", key)

            flags = ""
            flags += " -DINVOLUTION_KERNEL_SIZE=" + str(kernel_size)
            flags += " -DINVOLUTION_STRIDE_X=" + str(stride_x)
            flags += " -DINVOLUTION_STRIDE_Y=" + str(stride_y)
            flags += " -DINVOLUTION_CHANNEL_COUNT=" + str(channels)
            flags += " -DINVOLUTION_GROUP_COUNT=" + str(groups)

            native_sources_dir = Path(__file__).parent / "native"
            op = load(key,
                      sources=[native_sources_dir / "involution.cpp", native_sources_dir / "involution.cu"],
                      extra_cflags=["-std=c++17 -O3"],
                      extra_cuda_cflags=["-std=c++17 -O3 -use_fast_math" + flags])
            involution_op_cache[key] = op

            return op


class InvolutionFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, kernel_size: int, stride_x: int, stride_y: int, channels: int, groups: int):
        ctx.save_for_backward(input, weight)
        ctx.kernel_size = kernel_size
        ctx.stride_x = stride_x
        ctx.stride_y = stride_y
        ctx.channels = channels
        ctx.groups = groups

        op = get_involution_op(kernel_size=kernel_size, stride_x=stride_x, stride_y=stride_y, channels=channels, groups=groups)

        return op.involution(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        kernel_size = ctx.kernel_size
        stride_x = ctx.stride_x
        stride_y = ctx.stride_y
        channels = ctx.channels
        groups = ctx.groups
        op = get_involution_op(kernel_size=kernel_size, stride_x=stride_x, stride_y=stride_y, channels=channels, groups=groups)

        gInput, gWeight = op.involution_bw(grad_output, input, weight)

        return gInput, gWeight, None, None, None, None, None


@torch.jit.ignore
def involution_wrapper(input, weight, kernel_size: int, stride_x: int, stride_y: int, channels: int, groups: int):
    return InvolutionFunction.apply(input, weight, kernel_size, stride_x, stride_y, channels, groups)


def involution(input, weight, kernel_size: int, stride_x: int, stride_y: int, channels: int, groups: int = 1):
    """
    Performs the involution operation with the given inputs
    Involution works much like convolution, except that the kernel is different for each location (weight input)
    Different kernel sizes and strides can be chosen and weight sharing via groups is supported

    Args:
        input: Input tensor of shape (N, channels, H, W)
        weight: Weight tensor containing kernels for each location of shape (N, kernel_size * kernel_size * (channels + groups - 1) // groups), H, W)
        kernel_size: Involution kernel size
        stride_x: Involution horizontal stride
        stride_y: Involution vertical stride
        channels: Number of channels
        groups: Number of groups

    Returns:
        Involution result
    """

    return involution_wrapper(input, weight, kernel_size, stride_x, stride_y, channels, groups)
