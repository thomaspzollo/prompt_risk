import math
import numpy as np
import scipy
from scipy.stats import ksone, ks_1samp, norm
from scipy.special import betaincinv
from scipy.optimize import bisect
import pandas as pd
from dataclasses import asdict, dataclass
from torch.utils.data import TensorDataset
import crossprob
import torch
from bisect import bisect_left
import argparse
from argparse import Namespace
import os
import pickle as pkl
from typing import Callable
import numpy.typing as npt


def mean_quantile_weight(p):
    return 1.0

def cvar_quantile_weight(p, beta_min):
    if p >= beta_min:
        return 1.0 / (1.0 - beta_min)
    else:
        return 0.0
    
def interval_quantile_weight(p, beta_min, beta_max):
    if beta_min <= p and p <= beta_max:
        return 1.0 / (beta_max - beta_min)
    else:
        return 0.0
    
def integrate_quantiles(X, b, beta_min=0.0, beta_max=1.0):
    dist_max = 1.0
    X_sorted = np.sort(X, axis=-1)
    b_lower = np.concatenate([np.zeros(1), b], -1)
    b_upper = np.concatenate([b, np.ones(1)], -1)
    
    # clip bounds to [beta_min, 1]
    b_lower = np.maximum(b_lower, beta_min)
    b_upper = np.maximum(b_upper, b_lower)
    
    # clip bounds to [0, beta_max]
    b_upper = np.minimum(b_upper, beta_max)
    b_lower = np.minimum(b_upper, b_lower)

    heights = b_upper - b_lower
    widths = np.concatenate([X_sorted, np.full((X_sorted.shape[0], 1), dist_max)], -1)
    return np.sum(heights * widths, -1) / (beta_max - beta_min)


def ks_bound(n, delta):
    i = np.arange(1, n + 1)
    c = scipy.stats.ksone.isf(delta, n=n)
    return np.clip(i / n - c, 0.0, 1.0)


def ecdf(xs, data):
    return np.mean(np.expand_dims(data, -1) <= xs, -2)


@dataclass(frozen=True)
class OrderStatsBound:
    """Class for representing an CDF lower bounds based on order statistics."""
    q: npt.ArrayLike
    delta: float # probability of failure
        
    @property
    def n(self) -> int:
        return self.q.shape[0]
        
    def __post_init__(self):
        assert self.q.ndim == 1, "q must be 1d"
        assert (0 <= np.min(self.q)) and (np.max(self.q) <= 1.0), "elements of q must be in [0, 1]"
        assert np.all((self.q[1:] - self.q[:-1]) >= 0.0), "elements of q must be monotone nondecreasing"
        assert 0 <= self.delta <= 1, "delta must be in [0, 1]"


def bisect_proposal(proposal : Callable, delta : float):
    def f(c):
        return crossprob.ecdf1_new_b(proposal(c)) - (1 - delta)
    
    c_opt = scipy.optimize.bisect(f, 0.0, 1.0)
    
    return OrderStatsBound(proposal(c_opt), delta)

def berk_jones(n : int, delta : float) -> OrderStatsBound:
    def proposal(c : float):
        i = np.arange(1, n + 1)
        return scipy.special.betaincinv(i, n - i + 1, c)
    
    return bisect_proposal(proposal, delta).q

def berk_jones_trunc(n : int, delta : float, k : int) -> OrderStatsBound:
    def proposal(c : float):
        i = np.arange(1, n + 1)
        b = scipy.special.betaincinv(i, n - i + 1, c)
        b[:k] = 0.0
        return b
    
    return bisect_proposal(proposal, delta)

def berk_jones_trunc2(n : int, delta : float, k_min : int, k_max : int) -> OrderStatsBound:
    assert k_max > k_min, "k_max must be greater than k_min. got {:f}, {:f}".format(k_min, k_max)
    def proposal(c : float):
        i = np.arange(1, n + 1)
        b = scipy.special.betaincinv(i, n - i + 1, c)
        b[:k_min] = 0.0
        b[k_max:] = b[k_max-1]
        return b
    
    return bisect_proposal(proposal, delta)    

def berk_jones_one_sided(n : int, delta : float, q_min : float) -> OrderStatsBound:
    def get_bound(k):
        return berk_jones_trunc(n, delta, k)
    
    k_opt = bisect_left(
        np.arange(n).tolist(),
        q_min,
        key=lambda k: get_bound(k).q[k]
    )
    
    return get_bound(k_opt).q

def berk_jones_two_sided(n : int, delta : float, q_min : float, q_max : float) -> OrderStatsBound:
    def get_bound(k):
        return berk_jones_trunc(n, delta, k)
    
    k_min_opt = bisect_left(
        np.arange(n).tolist(),
        q_min,
        key=lambda k: get_bound(k).q[k]
    )
    
    def get_bound2(k_max):
        return berk_jones_trunc2(n, delta, k_min_opt, k_max)
        
    def key_fun(k):
        k_clip = max(k_min_opt+1, k)
        return get_bound2(k_clip).q[k_clip]
        
    k_max_opt = bisect_left(
        np.arange(n).tolist(),
        q_max,
        key=key_fun
    )
    
    return get_bound2(k_max_opt).q


def invert_lcb(t, lcb, beta_vals):
    inds = np.apply_along_axis(np.searchsorted, 1, lcb, beta_vals)
    t_len = t.shape[0]
    return np.where(inds == t_len, np.inf, t[np.minimum(t_len - 1, inds)])


class Bound:

    def __init__(self, name, bound_fn, p_ignore=None):
        self.name = name
        self.bound_fn = bound_fn
        self.p_ignore = p_ignore
        self.b = None


def calc_gini(X, L, beta_min=0.0, beta_max=1.0):

    b = L

    dist_max = 1.0
    X_sorted = np.sort(X, axis=-1)
    
    b_lower = np.concatenate([np.zeros(1), b], -1)
    b_upper = np.concatenate([b, np.ones(1)], -1)
    
    # clip bounds to [beta_min, 1]
    b_lower = np.maximum(b_lower, beta_min)
    b_upper = np.maximum(b_upper, b_lower)
    
    # clip bounds to [0, beta_max]
    b_upper = np.minimum(b_upper, beta_max)
    b_lower = np.minimum(b_upper, b_lower)
    
    # heights = b_upper - b_lower
    widths = np.concatenate([X_sorted, np.full((X_sorted.shape[0], 1), dist_max)], -1)
    
    num = np.sum((b_upper**2-b_lower**2)*widths, -1)
    den = np.sum((np.flip(b_lower[1:])-np.flip(b_lower[:-1]))*X_sorted, -1)
    res = (num/den)-1.0
    res = np.minimum(res, np.ones_like(res))
    return res
