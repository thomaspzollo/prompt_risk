import numpy as np
import scipy
import argparse
from argparse import Namespace
import os
import pickle as pkl
from tqdm import tqdm
from collections import OrderedDict


def ecdf(xs, data):
    return np.mean(np.expand_dims(data, -1) <= xs, -2)

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