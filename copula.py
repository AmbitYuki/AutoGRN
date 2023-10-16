from __future__ import division
from __future__ import print_function

import torch.nn as nn

from torch.autograd import Function

from scipy.special import gammainc
from scipy.stats import poisson
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson

import math
import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.pytorch.cell import DCGRUCell


EPS = 1e-3

def _standard_normal_quantile(u):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return math.sqrt(2) * torch.erfinv(2 * u - 1)


def _standard_normal_cdf(x):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class GaussianCopula(Distribution):
    r"""
    A Gaussian copula.

    Args:
        covariance_matrix (torch.Tensor): positive-definite covariance matrix
    """
    arg_constraints = {"covariance_matrix": constraints.positive_definite}
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(self, covariance_matrix=None, validate_args=None):
        # convert the covariance matrix to the correlation matrix
        # self.covariance_matrix = covariance_matrix.clone()
        # batch_diag = torch.diagonal(self.covariance_matrix, dim1=-1, dim2=-2).pow(-0.5)
        # self.covariance_matrix *= batch_diag.unsqueeze(-1)
        # self.covariance_matrix *= batch_diag.unsqueeze(-2)
        diag = torch.diag(covariance_matrix).pow(-0.5)
        self.covariance_matrix = (
            torch.diag(diag)).matmul(covariance_matrix).matmul(
            torch.diag(diag))

        batch_shape, event_shape = (
            covariance_matrix.shape[:-2],
            covariance_matrix.shape[-1:],
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.multivariate_normal = MultivariateNormal(
            loc=torch.zeros(event_shape),
            covariance_matrix=self.covariance_matrix,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value_x = _standard_normal_quantile(value)
        half_log_det = (
            self.multivariate_normal._unbroadcasted_scale_tril.diagonal(
                dim1=-2, dim2=-1
            )
            .log()
            .sum(-1)
        )
        M = _batch_mahalanobis(
            self.multivariate_normal._unbroadcasted_scale_tril, value_x
        )
        M -= value_x.pow(2).sum(-1)
        return -0.5 * M - half_log_det

    def conditional_sample(
        self, cond_val, sample_shape=torch.Size([]), cond_idx=None, sample_idx=None
    ):
        """
        Draw samples conditioning on cond_val.

        Args:
            cond_val (torch.Tensor): conditional values. Should be a 1D tensor.
            sample_shape (torch.Size): same as in
                `Distribution.sample(sample_shape=torch.Size([]))`.
            cond_idx (torch.LongTensor): indices that correspond to cond_val.
                If None, use the last m dimensions, where m is the length of cond_val.
            sample_idx (torch.LongTensor): indices to sample from. If None, sample
                from all remaining dimensions.

        Returns:
            Generates a sample_shape shaped sample or sample_shape shaped batch of
                samples if the distribution parameters are batched.
        """
        m, n = *cond_val.shape, *self.event_shape

        if cond_idx is None:
            cond_idx = torch.arange(n - (m/2), n, dtype=torch.int64).to(device)
        if sample_idx is None:
            sample_idx = torch.tensor(
                [i for i in range(n) if i not in set(cond_idx.tolist())]
            ).to(device)

        assert (
            len(cond_idx) == (m/2)
            and len(sample_idx) + len(cond_idx) <= n
            and not set(cond_idx.tolist()) & set(sample_idx.tolist())
        )

        cov_00 = self.covariance_matrix.index_select(
            dim=0, index=sample_idx
        ).index_select(dim=1, index=sample_idx)

        cov_01 = self.covariance_matrix.index_select(
            dim=0, index=sample_idx
        ).index_select(dim=1, index=cond_idx)

        cov_10 = self.covariance_matrix.index_select(
            dim=0, index=cond_idx
        ).index_select(dim=1, index=sample_idx)

        cov_11 = self.covariance_matrix.index_select(
            dim=0, index=cond_idx
        ).index_select(dim=1, index=cond_idx)

        cond_val_nscale = _standard_normal_quantile(cond_val)  # Phi^{-1}(u_cond)
        reg_coeff, _ = torch.solve(cov_10, cov_11)  # Sigma_{11}^{-1} Sigma_{10}
        cond_mu = torch.mv(reg_coeff.t(), cond_val_nscale)
        cond_sigma = cov_00 - torch.mm(cov_01, reg_coeff)
        cond_normal = MultivariateNormal(loc=cond_mu, covariance_matrix=cond_sigma)

        samples_nscale = cond_normal.sample(sample_shape)
        samples_uscale = _standard_normal_cdf(samples_nscale)

        return samples_uscale


def grad_x_gammainc(a, x, grad_output):
    temp = -x + (a-1)*torch.log(x) - torch.lgamma(a)
    temp = torch.where(temp > -25, torch.exp(temp), torch.zeros_like(temp))  # avoid underflow
    return temp * grad_output  # everything is element-wise


class GammaIncFunc(Function):
    '''Regularized lower incomplete gamma function.'''
    @staticmethod
    def forward(ctx, a, x):
        # detach so we can cast to NumPy
        a = a.detach()
        x = x.detach()
        result = gammainc(a.cpu().numpy(), x.cpu().numpy())
        result = torch.as_tensor(result, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(a, x, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        a, x, result = ctx.saved_tensors
        grad_a = torch.zeros_like(a)  # grad_a is never needed
        grad_x = grad_x_gammainc(a, x, grad_output)
        return grad_a, grad_x


def _batch_normal_icdf(loc, scale, value):
    return loc[None, :] + scale[None, :] * torch.erfinv(2 * value - 1) * math.sqrt(2)


class CopulaModel(nn.Module):

    def __init__(self, marginal_type = "Normal", eps=EPS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marginal_type = marginal_type
        self.eps = eps

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(3.0))
        self.S = None
        self.I = None

    # covariance matrix
    def get_prec(self, inputs, adj):
        if self.S is None:
            degree = adj.sum(axis=0)
            degree[degree==0] = 1
            D = np.diag(degree**(-0.5))
            S = D.dot(adj).dot(D)
            self.S = torch.tensor(S, dtype=torch.float32).to(inputs.device)
            self.I = torch.eye(self.S.size(0)).to(inputs.device)
        prec = torch.exp(self.beta) * (self.I - torch.tanh(self.alpha) * self.S)
        return prec

    # precision matrix
    def get_cov(self, inputs, adj):
        return torch.inverse(self.get_prec(inputs, adj))


    def marginal(self, logits, cov=None):
        if self.marginal_type == "Normal":
            return Normal(loc=logits, scale=torch.diag(cov).pow(0.5))
        elif self.marginal_type == "Poisson":
            return Poisson(rate=torch.exp(logits) + self.eps)
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))

    def cdf(self, marginal, labels):
        if self.marginal_type == "Normal":
            res = marginal.cdf(labels)
        elif self.marginal_type == "Poisson":
            cdf_left = 1 - GammaIncFunc.apply(labels, marginal.mean)
            cdf_right = 1 - GammaIncFunc.apply(labels + 1, marginal.mean)
            res = (cdf_left + cdf_right) / 2
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))
        return torch.clamp(res, self.eps, 1-self.eps)

    def icdf(self, marginal, u):
        if self.marginal_type == "Normal":
            res = _batch_normal_icdf(marginal.mean, marginal.stddev, u)
        elif self.marginal_type == "Poisson":
            mean = marginal.mean.detach().cpu().numpy()
            q = u.detach().cpu().numpy()
            res = poisson.ppf(q, mean)
            res = torch.as_tensor(res, dtype=u.dtype, device=u.device)
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))
        if (res == float("inf")).sum() + (res == float("nan")).sum() > 0:
            # remove the rows containing inf or nan values
            inf_mask = res.sum(dim=-1) == float("inf")
            nan_mask = res.sum(dim=-1) == float("nan")
            res[inf_mask] = 0
            res[nan_mask] = 0
            valid_num = inf_mask.size(0) - inf_mask.sum() - nan_mask.sum()
            return res.sum(dim=0) / valid_num
        return res.mean(dim=0)


    # def get_cov(self, inputs, adj):
    #     raise NotImplementedError("`get_cov` not implemented.")
    #
    # def get_prec(self, inputs, adj):
    #     raise NotImplementedError("`get_prec` not implemented.")



    def nll(self, x, y, adj, logits):
        '''
        :param x: shape (seq_len, batch_size, num_sensor*input_dim)
        :param y: shape (horizon, batch_size, num_sensor*input_dim)
        '''
        inputs = x.mean(1)
        # inputs = input.mean(axis=-1, keepdim=True)
        cov = self.get_cov(inputs, adj)
        # cov = cov[inputs.type(torch.long), :]
        # cov = cov[:, inputs.type(torch.long)]
        # logits = self.forward(inputs)[inputs]
        labels = y.mean(1)
        # C = np.linalg.cholesky(cov.cpu().detach().numpy())
        # print("c",C)
        copula = GaussianCopula(cov)
        marginal = self.marginal(logits.mean(1), cov)
        u = self.cdf(marginal, labels)
        # us = u.mean(1)
        nll_copula = - torch.sum(copula.log_prob(u))
        # print("nll_copula", nll_copula)
        nll_marginal = - torch.sum(marginal.log_prob(labels))
        # l = (nll_copula + nll_marginal) / labels.size(0)
        return (nll_copula + nll_marginal) / labels.size(0)
    '''
        def nll(self, data):
            cov = self.get_cov(data)
            cov = cov[data.train_mask, :]
            cov = cov[:, data.train_mask]
            logits = self.forward(data)[data.train_mask]
            labels = data.y[data.train_mask]

            copula = GaussianCopula(cov)
            marginal = self.marginal(logits, cov)

            u = self.cdf(marginal, labels)
            nll_copula = - copula.log_prob(u)
            nll_marginal = - torch.sum(marginal.log_prob(labels))
            return (nll_copula + nll_marginal) / labels.size(0)
    '''

    def predict(self, x, y, adj, logits, num_samples=500):

        '''
        input = x.mean(1)
        cov = self.get_cov(input, adj)
        labels = y.mean(1)
        copula = GaussianCopula(cov)
        marginal = self.marginal(logits.mean(1), cov)
        '''

        # cond_mask = data.train_mask
        # eval_mask = torch.logical_xor(
        #     torch.ones_like(inputs).to(dtype=torch.bool),
        #     inputs)
        inputs = x.mean(1)
        cov = self.get_cov(inputs, adj)
        # logits = self.forward(inputs, adj)
        copula = GaussianCopula(cov)

        # cond_cov = (cov[inputs, :])[:, inputs]
        cond_marginal = self.marginal(logits.mean(1), cov)
        # eval_cov = (cov[eval_mask, :])[:, eval_mask]
        eval_marginal = self.marginal(logits.mean(1), cov)

        cond_u = torch.clamp(
            self.cdf(cond_marginal, y.mean(1)), self.eps, 1-self.eps)
        cond_val = cond_u.mean(1)
        cond_idx = torch.where(x)[0]
        sample_idx = torch.where(x)[0]
        eval_u = copula.conditional_sample(
            cond_val=cond_val, sample_shape=[num_samples, ])
            # , cond_idx=cond_idx,
            # sample_idx=sample_idx)
        eval_u = torch.clamp(eval_u, self.eps, 1-self.eps)
        eval_y = self.icdf(eval_marginal, eval_u)

        # pred_y = y.clone()
        # pred_y[eval_mask] = eval_y
        return eval_y
    '''
        def predict(self, data, num_samples=500):
            cond_mask = data.train_mask
            eval_mask = torch.logical_xor(
                torch.ones_like(data.train_mask).to(dtype=torch.bool),
                data.train_mask)
            cov = self.get_cov(data)
            logits = self.forward(data)
            copula = GaussianCopula(cov)

            cond_cov = (cov[cond_mask, :])[:, cond_mask]
            cond_marginal = self.marginal(logits[cond_mask], cond_cov)
            eval_cov = (cov[eval_mask, :])[:, eval_mask]
            eval_marginal = self.marginal(logits[eval_mask], eval_cov)

            cond_u = torch.clamp(
                self.cdf(cond_marginal, data.y[cond_mask]), self.eps, 1-self.eps)
            cond_idx = torch.where(cond_mask)[0]
            sample_idx = torch.where(eval_mask)[0]
            eval_u = copula.conditional_sample(
                cond_val=cond_u, sample_shape=[num_samples, ], cond_idx=cond_idx,
                sample_idx=sample_idx)
            eval_u = torch.clamp(eval_u, self.eps, 1-self.eps)
            eval_y = self.icdf(eval_marginal, eval_u)

            pred_y = data.y.clone()
            pred_y[eval_mask] = eval_y
            return pred_y
    '''
