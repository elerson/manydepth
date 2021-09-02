# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
r"""Implements the distribution corresponding to the loss function.

This library implements the parts of Section 2 of "A General and Adaptive Robust
Loss Function", Jonathan T. Barron, https://arxiv.org/abs/1701.03077, that are
required for evaluating the negative log-likelihood (NLL) of the distribution
and for sampling from the distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from pkg_resources import resource_stream
import mpmath
import numpy as np
import torch
from manydepth.robust_loss_pytorch import cubic_spline
from manydepth.robust_loss_pytorch import general
from manydepth.robust_loss_pytorch import util
import ast

def partition_spline_curve(alpha):
  """Applies a curve to alpha >= 0 to compress its range before interpolation.

  This is a weird hand-crafted function designed to take in alpha values and
  curve them to occupy a short finite range that works well when using spline
  interpolation to model the partition function Z(alpha). Because Z(alpha)
  is only varied in [0, 4] and is especially interesting around alpha=2, this
  curve is roughly linear in [0, 4] with a slope of ~1 at alpha=0 and alpha=4
  but a slope of ~10 at alpha=2. When alpha > 4 the curve becomes logarithmic.
  Some (input, output) pairs for this function are:
    [(0, 0), (1, ~1.2), (2, 4), (3, ~6.8), (4, 8), (8, ~8.8), (400000, ~12)]
  This function is continuously differentiable.

  Args:
    alpha: A numpy array or tensor (float32 or float64) with values >= 0.

  Returns:
    An array/tensor of curved values >= 0 with the same type as `alpha`, to be
    used as input x-coordinates for spline interpolation.
  """
  alpha = torch.as_tensor(alpha)
  x = torch.where(alpha < 4, (2.25 * alpha - 4.5) /
                  (torch.abs(alpha - 2) + 0.25) + alpha + 2,
                  5. / 18. * util.log_safe(4 * alpha - 15) + 8)
  return x


def inv_partition_spline_curve(x):
  """The inverse of partition_spline_curve()."""
  x = torch.as_tensor(x)
  assert (x >= 0).all()
  alpha = torch.where(
      x < 8,
      0.5 * x + torch.where(x <= 4, 1.25 - torch.sqrt(1.5625 - x + .25 * x**2),
                            -1.25 + torch.sqrt(9.5625 - 3 * x + .25 * x**2)),
      3.75 + 0.25 * util.exp_safe(x * 3.6 - 28.8))
  return alpha


class DistributionSkewed():
  # This is only a class so that we can pre-load the partition function spline.

  def __init__(self):
    # Load the values, tangents, and x-coordinate scaling of a spline that
    # approximates the partition function. This was produced by running
    # the script in fit_partition_spline.py
    with resource_stream(__name__, 'resources/skewd_file.npz') \
      as spline_file:
      with np.load(spline_file, allow_pickle=True) as f:
        self._spline_2d_value = torch.tensor(f['value'])

        self._spline_2d_info = ast.literal_eval(str(f['info']))

    self.x_min, self.x_max = self._spline_2d_info['x_min'], self._spline_2d_info['x_max']
    self.b_min, self.b_max = self._spline_2d_info['b_min'], self._spline_2d_info['b_max']
    self.N_b  , self.N = self._spline_2d_info['N_b'], self._spline_2d_info['N']
    self.L  = (self.N)/(self.x_max-self.x_min)
    self.Lb = (self.N_b)/(self.b_max-self.b_min)
    


    #print(self._spline_2d_value)

    with resource_stream(__name__, 'resources/partition_spline.npz') \
      as spline_file:
      with np.load(spline_file, allow_pickle=False) as f:
        self._spline_x_scale = torch.tensor(f['x_scale'])
        self._spline_values = torch.tensor(f['values'])
        self._spline_tangents = torch.tensor(f['tangents'])


  def Z_integral(self, x, alpha, beta):

    
    #print('info', alpha, beta)
    

    new_x = (x*alpha-self.x_min)*self.L
    #new_x = torch.flatten(new_x)
    
    alpha = torch.as_tensor(alpha)
    beta = (torch.as_tensor(beta)-self.b_min)*self.Lb

    beta = torch.flatten(beta)[0]
    #print(self._spline_2d_value[:,1024])

    
    
    r = cubic_spline.interpolate2d(beta.to(x),
                                      new_x.to(x),
                                      self._spline_2d_value.to(x))

    #print(r)
    return r
    pass


  def log_base_partition_function(self, beta):
    r"""Approximate the distribution's log-partition function with a 1D spline.

    Because the partition function (Z(\beta) in the paper) of the distribution
    is difficult to model analytically, we approximate it with a (transformed)
    cubic hermite spline: Each beta is pushed through a nonlinearity before
    being used to interpolate into a spline, which allows us to use a relatively
    small spline to accurately model the log partition function over the range
    of all non-negative input values.

    Args:
      beta: A tensor or scalar of single or double precision floats containing
        the set of beta for which we would like an approximate log partition
        function. Must be non-negative, as the partition function is undefined
        when beta < 0.

    Returns:
      An approximation of log(Z(beta)) accurate to within 1e-6
    """
    beta = torch.as_tensor(beta)
    assert (beta >= 0).all()
    # Transform `beta` to the form expected by the spline.
    x = partition_spline_curve(beta)
    # Interpolate into the spline.
    return cubic_spline.interpolate1d(x * self._spline_x_scale.to(x),
                                      self._spline_values.to(x),
                                      self._spline_tangents.to(x))

  def nllfun(self, x, alpha, beta, scale):
    r"""Implements the negative log-likelihood (NLL).

    Specifically, we implement -log(p(x | 0, \beta, c) of Equation 16 in the
    paper as nllfun(x, beta, shape).

    Args:
      x: The residual for which the NLL is being computed. x can have any shape,
        and beta and scale will be broadcasted to match x's shape if necessary.
        Must be a tensor or numpy array of floats.
      beta: The shape parameter of the NLL (\beta in the paper), where more
        negative values cause outliers to "cost" more and inliers to "cost"
        less. beta can be any non-negative value, but the gradient of the NLL
        with respect to beta has singularities at 0 and 2 so you may want to
        limit usage to (0, 2) during gradient descent. Must be a tensor or numpy
        array of floats. Varying beta in that range allows for smooth
        interpolation between a Cauchy distribution (beta = 0) and a Normal
        distribution (beta = 2) similar to a Student's T distribution.
      scale: The scale parameter of the loss. When |x| < scale, the NLL is like
        that of a (possibly unnormalized) normal distribution, and when |x| >
        scale the NLL takes on a different shape according to beta. Must be a
        tensor or numpy array of floats.

    Returns:
      The NLLs for each element of x, in the same shape and precision as x.
    """
    # `scale` and `beta` must have the same type as `x`.
    x = torch.as_tensor(x)
    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)
    scale = torch.as_tensor(scale)
    assert (beta >= 0).all()
    assert (scale >= 0).all()
    float_dtype = x.dtype
    assert beta.dtype == float_dtype
    assert scale.dtype == float_dtype

    #l = torch.as_tensor(0.65)#self.Z_integral(x, alpha, beta)
    #print(l)
    loss = general.lossfun(x, beta, scale, approximate=True)
    #print(torch.log(scale))
    log_partition = torch.log(scale/2.0) + self.log_base_partition_function(beta) + self.Z_integral(x, alpha, beta)
    nll = loss + log_partition
    return nll

  def draw_samples(self, beta, scale):
    r"""Draw samples from the robust distribution.

    This function implements Algorithm 1 the paper. This code is written to
    allow
    for sampling from a set of different distributions, each parametrized by its
    own beta and scale values, as opposed to the more standard approach of
    drawing N samples from the same distribution. This is done by repeatedly
    performing N instances of rejection sampling for each of the N distributions
    until at least one proposal for each of the N distributions has been
    accepted.
    All samples are drawn with a zero mean, to use a non-zero mean just add each
    mean to each sample.

    Args:
      beta: A tensor/scalar or numpy array/scalar of floats where each element
        is the shape parameter of that element's distribution.
      scale: A tensor/scalar or numpy array/scalar of floats where each element
        is the scale parameter of that element's distribution. Must be the same
        shape as `beta`.

    Returns:
      A tensor with the same shape and precision as `beta` and `scale` where
      each element is a sample drawn from the distribution specified for that
      element by `beta` and `scale`.
    """
    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)
    scale = torch.as_tensor(scale)
    assert (alpha >= -20).all()
    assert (alpha <= 20).all()
    assert (beta >= -2).all()
    assert (scale >= 0).all()
    float_dtype = beta.dtype
    assert scale.dtype == float_dtype

    cauchy = torch.distributions.cauchy.Cauchy(0., np.sqrt(2.))
    uniform = torch.distributions.uniform.Uniform(0, 1)
    samples = torch.zeros_like(beta)
    accepted = torch.zeros(beta.shape).type(torch.bool)
    while not accepted.type(torch.uint8).all():
      # Draw N samples from a Cauchy, our proposal distribution.
      cauchy_sample = torch.reshape(
          cauchy.sample((np.prod(beta.shape),)), beta.shape)
      cauchy_sample = cauchy_sample.type(beta.dtype)

      # Compute the likelihood of each sample under its target distribution.
      nll = self.nllfun(cauchy_sample,
                        torch.as_tensor(alpha).to(cauchy_sample),
                        torch.as_tensor(beta).to(cauchy_sample),
                        torch.tensor(1).to(cauchy_sample))

      # Bound the NLL. We don't use the approximate loss as it may cause
      # unpredictable behavior in the context of sampling.
      nll_bound = general.lossfun(
          cauchy_sample,
          torch.tensor(0., dtype=cauchy_sample.dtype),
          torch.tensor(0., dtype=cauchy_sample.dtype),
          torch.tensor(1., dtype=cauchy_sample.dtype),
          approximate=False) + self.log_base_partition_function(beta) + self.Z_integral(cauchy_sample, alpha, beta)

      # Draw N samples from a uniform distribution, and use each uniform sample
      # to decide whether or not to accept each proposal sample.
      uniform_sample = torch.reshape(
          uniform.sample((np.prod(beta.shape),)), beta.shape)
      uniform_sample = uniform_sample.type(beta.dtype)
      accept = uniform_sample <= torch.exp(nll_bound - nll)

      # If a sample is accepted, replace its element in `samples` with the
      # proposal sample, and set its bit in `accepted` to True.
      samples = torch.where(accept, cauchy_sample, samples)
      accepted = accepted | accept

    # Because our distribution is a location-scale family, we sample from
    # p(x | 0, \beta, 1) and then scale each sample by `scale`.
    samples *= scale
    return samples

