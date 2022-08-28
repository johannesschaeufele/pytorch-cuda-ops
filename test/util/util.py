import torch
import torch.nn.functional as F

import pytest


@pytest.fixture(scope="module")
def pytorch_test_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.jit.optimized_execution(True):
        with torch.autograd.detect_anomaly():
            yield device


def pytorch_test_setup(detect_anomaly: bool = True):
    state = {}

    torch.set_printoptions(threshold=1000000000)

    state["restore_optimize"] = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(True)

    state["restore_anomaly"] = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(detect_anomaly)

    return state


def pytorch_test_teardown(state):
    if "restore_optimize" in state:
        torch._C._set_graph_executor_optimize(state["restore_optimize"])

    if "restore_anomaly" in state:
        torch.set_anomaly_enabled(state["restore_anomaly"])


def check_forward_agreement(a, b):
    assert a.shape == b.shape
    assert torch.isclose(a, b, equal_nan=True, rtol=1E-04, atol=1E-06).all()


def bilinear_sampler(img, coords, mode: str = "bilinear", padding_mode: str = "zeros"):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    assert H > 1
    assert W > 1

    xgrid, ygrid = coords.split([1, 1], dim=-1)

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    output = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    return output
