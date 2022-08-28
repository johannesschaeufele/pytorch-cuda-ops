import torch
from torch import autograd

from op.patch_correlate import patch_correlate

import pytest
from util.util import pytorch_test_device, check_forward_agreement, bilinear_sampler


def alt_corr(fmap1, fmap2):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    return corr / torch.sqrt(torch.tensor(dim).float())


def patch_correlate_alt(base, input, coords, r):
    batch, h1, w1, _ = coords.shape

    dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
    dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

    centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl

    corr = alt_corr(base, input)
    _, _, _, dim, h2, w2 = corr.shape
    corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
    corr = bilinear_sampler(corr, coords_lvl)
    corr = corr.view(batch, h1, w1, -1)

    corr = corr.permute(0, 3, 1, 2)

    return corr


@pytest.mark.parametrize("r", (1, 3, 5))
def test_patch_correlate_forward(pytorch_test_device, r):
    batch_size: int = 1
    channels: int = 6
    size: int = 4
    f1 = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)
    f2 = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)
    coords = (torch.randn(batch_size, size, size, 2, dtype=torch.double, device=pytorch_test_device, requires_grad=True) + 1.0) / 2.0 * size

    check_forward_agreement(patch_correlate_alt(f1, f2, coords, r=r), patch_correlate(f1, f2, coords, r=r))


@pytest.mark.parametrize("r", (1, 3, 5))
@pytest.mark.parametrize("l1", (False, True))
def test_patch_correlate_backward(pytorch_test_device, r, l1):
    batch_size: int = 1
    channels: int = 6
    size: int = 4
    f1 = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)
    f2 = torch.randn(batch_size, channels, size, size, dtype=torch.double, device=pytorch_test_device, requires_grad=True)
    coords = (torch.randn(batch_size, size, size, 2, dtype=torch.double, device=pytorch_test_device, requires_grad=True) + 1.0) / 2.0 * size

    # Check backward pass
    autograd.gradcheck(lambda x, y, z: patch_correlate(x, y, z, r=r, l1=l1), [f1, f2, coords], raise_exception=True)
