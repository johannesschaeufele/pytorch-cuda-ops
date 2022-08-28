import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

from pathlib import Path

import threading

gsmelu_lock = threading.Lock()
gsmelu_op_cache = {}


def serialize_float(f):
    return str(f).replace(".", "_").replace("-", "m")


def get_gsmelu_op(slope: float, neg_slope: float, alpha: float, beta: float, t: float):
    with gsmelu_lock:
        key = "gsmelu__" + serialize_float(slope) + "__" + serialize_float(neg_slope) + "__" + serialize_float(
            alpha) + "__" + serialize_float(beta) + "__" + serialize_float(t)
        if key in gsmelu_op_cache:
            return gsmelu_op_cache[key]
        else:
            print("Compiling", key)

            flags = ""
            flags += " -DGSMELU_SLOPE=" + str(slope)
            flags += " -DGSMELU_NEG_SLOPE=" + str(neg_slope)
            flags += " -DGSMELU_ALPHA=" + str(alpha)
            flags += " -DGSMELU_BETA=" + str(beta)
            flags += " -DGSMELU_T=" + str(t)

            native_sources_dir = Path(__file__).parent / "native"
            op = load(key,
                      sources=[native_sources_dir / "gsmelu.cpp", native_sources_dir / "gsmelu.cu"],
                      extra_cflags=["-std=c++17 -O3"],
                      extra_cuda_cflags=["-std=c++17 -O3 -use_fast_math" + flags])
            gsmelu_op_cache[key] = op

            return op


class GSmeLuFunction(Function):

    @staticmethod
    def forward(ctx, input, slope: float, neg_slope: float, alpha: float, beta: float, delta_t: float):
        t = -neg_slope * alpha + delta_t

        ctx.save_for_backward(input)
        ctx.slope = slope
        ctx.neg_slope = neg_slope
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.t = t

        op = get_gsmelu_op(slope=slope, neg_slope=neg_slope, alpha=alpha, beta=beta, t=t)

        return op.gsmelu(input.view(-1).contiguous()).view(input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        slope = ctx.slope
        neg_slope = ctx.neg_slope
        alpha = ctx.alpha
        beta = ctx.beta
        t = ctx.t

        op = get_gsmelu_op(slope=slope, neg_slope=neg_slope, alpha=alpha, beta=beta, t=t)

        gInput = op.gsmelu_bw(grad_output.view(-1).contiguous(), input.view(-1).contiguous()).view(input.shape)

        return gInput, None, None, None, None, None


@torch.jit.ignore
def gsmelu_wrapper(input, slope: float, neg_slope: float, alpha: float, beta: float, delta_t: float):
    return GSmeLuFunction.apply(input, slope, neg_slope, alpha, beta, delta_t)


def gsmelu(input, slope: float = 1.0, neg_slope: float = 1E-2, alpha: float = 0.1, beta: float = 0.1, delta_t: float = 0.0):
    """
    Computes GSmeLU activation function
    """

    return gsmelu_wrapper(input, slope, neg_slope, alpha, beta, delta_t)


class GSmeLU(nn.Module):
    """
    GSmeLU activation function module
    """

    def __init__(self, slope: float = 1.0, negative_slope: float = 1E-2, alpha: float = 0.1, beta: float = 0.1, delta_t: float = 0.0):
        super().__init__()
        self.slope = slope
        self.neg_slope = negative_slope
        self.alpha = alpha
        self.beta = beta
        self.delta_t = delta_t

    def forward(self, input):
        return gsmelu(input, slope=self.slope, neg_slope=self.neg_slope, alpha=self.alpha, beta=self.beta, delta_t=self.delta_t)
