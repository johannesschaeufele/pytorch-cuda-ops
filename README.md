# pytorch-cuda-ops

CUDA implementations of various operations along with PyTorch wrappers for them

## Overview

### Operations

`gsmelu` (**G**eneralized **Sm**ooth R**eLU**) is a smooth activation function modelled as a generalized smooth equivalent of ReLU,
as described in equation (11) of [Smooth Activations and Reproducibility in Deep Networks](https://arxiv.org/abs/2010.09931) and equation (9) of [Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations](https://arxiv.org/abs/2202.06499).

`involution` is a parameter-efficient convolution-like operation with differing kernels per pixel that can be seen to incorporate an attention mechanism,
adapted from [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Involution_Inverting_the_Inherence_of_Convolution_for_Visual_Recognition_CVPR_2021_paper.html).

The `patch_correlate` operation is adapted from the computation of and indexing into the correlation volume introduced in [RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow](https://arxiv.org/abs/2003.12039),
used to solve the correspondence problem of optical flow estimation.

### Structure

The PyTorch wrappers for the operations can be found in the `op/` directory, with the CUDA implementations being contained in `op/native`.

Tests can be found in the `test/` directory. These check the correctness of the operations' forward pass using non-CUDA reference implementations
and that of the backward pass using finite difference approximations of derivatives.

## Requirements

### Software
 * Git (for checkout only)
 * GCC (g++)
 * CUDA, cuDNN, CUDA Toolkit
 * Python 3.7+

### Python packages

 * PyTorch 1.7+
 * pytest (testing only)

## Licensing
The file `op/native/patch_correlate.cu` is based on code that is subject to the license found in `op/native/PyTorch_LICENSE` and is thus subject to the conditions of that license.
Note that the modified file is not released under that license with no further conditions.
