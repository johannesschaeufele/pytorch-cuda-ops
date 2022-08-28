import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

from pathlib import Path

import threading
import warnings

patch_correlate_lock = threading.Lock()
patch_correlate_op_cache = {}


def get_patch_correlate_op(l1: bool = False, feature_dim: int = 256, r: int = 4):
    with patch_correlate_lock:
        key = "patch_correlate_" + str(l1) + "_" + str(feature_dim) + "_" + str(r)
        if key in patch_correlate_op_cache:
            return patch_correlate_op_cache[key]
        else:
            print("Compiling", key)

            flags = " "
            flags += "-DPATCH_CORRELATE_FEATURE_DIM=" + str(feature_dim) + " "
            flags += "-DPATCH_CORRELATE_RADIUS=" + str(r) + " "
            if l1:
                flags += "-DPATCH_CORRELATE_L1=1 "

            native_sources_dir = Path(__file__).parent / "native"
            op = load(key,
                      sources=[native_sources_dir / "patch_correlate.cpp", native_sources_dir / "patch_correlate.cu"],
                      extra_cflags=["-std=c++17 -O3"],
                      extra_cuda_cflags=["-std=c++17 -O3 -use_fast_math" + flags])
            patch_correlate_op_cache[key] = op

            return op


class PatchCorrelationFunction(Function):

    @staticmethod
    def forward(ctx, base, input, coords, r, l1: bool = False, feature_dim: int = 256):
        ctx.save_for_backward(base, input, coords)
        ctx.r = r
        ctx.l1 = l1
        ctx.feature_dim = feature_dim

        op = get_patch_correlate_op(l1=l1, feature_dim=feature_dim, r=r)

        return op.patch_correlate(base, input, coords)

    @staticmethod
    def backward(ctx, grad_output):
        base, input, coords = ctx.saved_tensors

        op = get_patch_correlate_op(l1=ctx.l1, feature_dim=ctx.feature_dim, r=ctx.r)

        gb, gi, gc = op.patch_correlate_bw(grad_output, base, input, coords)

        return gb, gi, gc, None, None, None


@torch.jit.ignore
def patch_correlate_wrapper(base, input, coords, r: int, l1: bool):
    assert base.shape[0] == input.shape[0]
    assert base.shape[1] == input.shape[1]

    return PatchCorrelationFunction.apply(base, input, coords, r, l1, base.shape[1])


def patch_correlate(base, input, coords, r: int = 4, l1: bool = False):
    """
    Performs the patch_correlate operation with the given inputs

    Args:
        base: Feature tensor in reference space
        input: Feature tensor in corresponding space
        coords: Coordinates to use for sampling input
        r: Correlation radius
        l1: If true, use L1 distance instead of dot product

    Returns:
        Computed correlation volume
    """

    if r <= 0:
        return torch.empty_like(base)

    if not base.is_contiguous(memory_format=torch.channels_last):
        warnings.warn("patch_correlate parameter base should be in channels last format to ensure good memory access patterns")
    if not input.is_contiguous(memory_format=torch.channels_last):
        warnings.warn("patch_correlate parameter input should be in channels last format to ensure good memory access patterns")

    return patch_correlate_wrapper(base, input, coords, r, l1)
