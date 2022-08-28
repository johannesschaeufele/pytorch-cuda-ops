import torch
from torch import autograd
import torch.nn.functional as F

from op.involution import involution

import pytest
from util.util import pytorch_test_device, check_forward_agreement


def involution_alt(input, weight, kernel_size: int, stride_x: int, stride_y: int, channels: int, groups: int = 1):
    assert channels % groups == 0

    unfolded = F.unfold(input, kernel_size=kernel_size, stride=(stride_y, stride_x))

    weight = weight.view(weight.shape[0], channels // groups, 1, kernel_size * kernel_size, weight.shape[2], weight.shape[3])
    unfolded = unfolded.view(input.shape[0], channels // groups, groups, kernel_size * kernel_size,
                             (input.shape[2] - 2 * (kernel_size // 2) - 1) // stride_y + 1,
                             (input.shape[3] - 2 * (kernel_size // 2) - 1) // stride_x + 1)

    convolved = (weight * unfolded).sum(dim=3)
    output = convolved.view(convolved.shape[0], convolved.shape[1] * convolved.shape[2], convolved.shape[3], convolved.shape[4])

    return output


@pytest.mark.parametrize("channels, groups", [(6, 3), (8, 2)])
@pytest.mark.parametrize("kernel_size", (3, 5))
@pytest.mark.parametrize("stride_x, stride_y", [(1, 1), (2, 3)])
def test_involution_forward(pytorch_test_device, channels, groups, kernel_size, stride_x, stride_y):
    with torch.no_grad():
        batch_size: int = 2
        size: int = 8

        effective_count = (channels + groups - 1) // groups

        input = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device)
        pad: int = kernel_size // 2
        input = F.pad(input, [pad, pad, pad, pad], mode="reflect")
        weight = torch.randn(batch_size,
                             effective_count * kernel_size * kernel_size, (size + stride_y - 1) // stride_y,
                             (size + stride_x - 1) // stride_x,
                             dtype=torch.double,
                             device=pytorch_test_device,
                             requires_grad=True)

        check_forward_agreement(
            involution_alt(input, weight, kernel_size=kernel_size, stride_x=stride_x, stride_y=stride_y, channels=channels, groups=groups),
            involution(input, weight, kernel_size=kernel_size, stride_x=stride_x, stride_y=stride_y, channels=channels, groups=groups))


@pytest.mark.parametrize("channels, groups", [(5, 3), (7, 2)])
@pytest.mark.parametrize("kernel_size", (3, 5))
@pytest.mark.parametrize("stride_x, stride_y", [(1, 1), (2, 3)])
def test_involution_backward(pytorch_test_device, channels, groups, kernel_size, stride_x, stride_y):
    batch_size: int = 2
    size: int = 8

    effective_count = (channels + groups - 1) // groups

    input = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)
    pad: int = kernel_size // 2
    input = F.pad(input, [pad, pad, pad, pad], mode="reflect")
    weight = torch.randn(batch_size,
                         effective_count * kernel_size * kernel_size, (size + stride_y - 1) // stride_y, (size + stride_x - 1) // stride_x,
                         dtype=torch.double,
                         device=pytorch_test_device,
                         requires_grad=True)

    # Check backward pass
    autograd.gradcheck(
        lambda x, y: involution(x, y, kernel_size=kernel_size, stride_x=stride_x, stride_y=stride_y, channels=channels, groups=groups),
        [input, weight],
        raise_exception=True)
