# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient MMD implementation."""

import torch

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def _kernel_mean(a, b, a_sq, b_sq, gamma, block):
    total = a.new_zeros(())
    for i in range(0, a.shape[0], block):
        d = -2 * a[i : i + block] @ b.T + a_sq[i : i + block, None] + b_sq[None, :]
        total += torch.exp(-gamma * d).sum()
    return total / (a.shape[0] * b.shape[0])


def mmd(x, y, block=4096):
    """
    Memory-efficient MMD implementation.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    The kernel means are accumulated over `block`-row chunks so the full n-by-n
    Gram matrices are never materialized, which would OOM on large sets.

    The computation runs in float64: the final MMD cancels ~5 digits between the
    three kernel means (each ~O(1)), so float32 would lose most of its
    significant figures and make the result depend on `block`.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (m, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    dtype = x.dtype
    x, y = x.double(), y.double()
    x_sqnorms = (x * x).sum(1)
    y_sqnorms = (y * y).sum(1)

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = _kernel_mean(x, x, x_sqnorms, x_sqnorms, gamma, block)
    k_xy = _kernel_mean(x, y, x_sqnorms, y_sqnorms, gamma, block)
    k_yy = _kernel_mean(y, y, y_sqnorms, y_sqnorms, gamma, block)

    return (_SCALE * (k_xx + k_yy - 2 * k_xy)).to(dtype)
