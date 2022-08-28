import torch
from torch import autograd

from op.gsmelu import gsmelu

import pytest
from util.util import pytorch_test_device, check_forward_agreement


def gsmelu_alt(x, slope: float = 1.0, neg_slope: float = 1E-2, alpha: float = 0.1, beta: float = 0.1, delta_t: float = 0.0):
    t = -neg_slope * alpha + delta_t

    a = (slope - neg_slope) / (2 * (alpha + beta))
    b = (alpha * slope + beta * neg_slope) / (alpha + beta)
    c = t + (alpha * alpha * (slope + neg_slope) + 2 * alpha * beta * neg_slope) / (2 * (alpha + beta))

    linear_neg = neg_slope * x + t + neg_slope * alpha
    quadratic = a * torch.square(x) + b * x + c
    linear_pos = slope * x + t + neg_slope * (alpha + beta) / 2 + slope * (alpha - beta) / 2

    leq_mask = (x <= -alpha).float()
    geq_mask = (x >= beta).float()
    quadratic_mask = (1 - leq_mask) * (1 - geq_mask)

    output = linear_neg * leq_mask + quadratic * quadratic_mask + linear_pos * geq_mask

    return output


@pytest.mark.parametrize("slope, neg_slope", [(1.0, 0.2), (1.0, 0.1)])
@pytest.mark.parametrize("alpha, beta", [(0.1, 0.2), (0.1, 0.3)])
@pytest.mark.parametrize("delta_t", (0.0, -0.1, 0.3))
def test_gsmelu_forward(pytorch_test_device, slope, neg_slope, alpha, beta, delta_t):
    with torch.no_grad():
        batch_size: int = 2
        channels: int = 5
        size: int = 8

        input = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device)

        check_forward_agreement(gsmelu(input, slope=slope, neg_slope=neg_slope, alpha=alpha, beta=beta, delta_t=delta_t),
                                gsmelu_alt(input, slope=slope, neg_slope=neg_slope, alpha=alpha, beta=beta, delta_t=delta_t))


@pytest.mark.parametrize("slope, neg_slope", [(1.0, 0.2), (1.0, 0.1)])
@pytest.mark.parametrize("alpha, beta", [(0.1, 0.2), (0.1, 0.3)])
@pytest.mark.parametrize("delta_t", (0.0, -0.1, 0.3))
def test_gsmelu_backward(pytorch_test_device, slope, neg_slope, alpha, beta, delta_t):
    batch_size: int = 2
    channels: int = 5
    size: int = 8

    input = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)

    # Check backward pass
    autograd.gradcheck(lambda x: gsmelu(x, slope=slope, neg_slope=neg_slope, alpha=alpha, beta=beta, delta_t=delta_t), [input],
                       raise_exception=True)
