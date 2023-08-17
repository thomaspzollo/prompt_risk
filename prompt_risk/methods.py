import math
from dataclasses import dataclass
from tkinter import W
from typing import Optional, Tuple
from typing import Callable
import torch
from torch import Tensor
import scipy.optimize
import scipy.stats
import scipy.special
import numpy as np
from scipy.stats import binom, norm
import pickle as pkl

from scipy.stats import binom
from scipy.optimize import brentq
from scipy.special import rel_entr
import bisect

# from var_control.methods_monotonic import hb_p_value, hoeffding_ucb, hoeffding_bentkus_ucb, find_rcps_index


def is_sorted(x: Tensor) -> bool:
    """Check if a tensor is sorted.

    Args:
        x (Tensor): A 1-dimensional tensor.

    Returns:
        bool: True if x is sorted low to high and False otherwise.
    """
    assert x.dim() == 1
    return bool(torch.all(x[:-1].le(x[1:])).item())


def hoeffding_ucb(loss: Tensor, delta: float) -> float:
    """The Hoeffding upper confidence bound for losses in [0, 1].

    Args:
        loss (Tensor): A tensor of loss values with shape (N), where N is the
                       number of examples.
        delta (float): The probability of error for the UCB.

    Returns:
        float: The UCB value.
    """
    assert loss.dim() == 1
    assert (loss.min().item() >= 0.0) and (loss.max().item() <= 1.0), "Loss values must be in the range [0, 1]"
    num_train = loss.size(-1)
    mean_loss = torch.mean(loss, dim=-1).item()
    # compute upper confidence bound on risk for each hypothesis
    return mean_loss + math.sqrt(math.log(1.0 / delta) / (2 * num_train))


def WSR_mu_plus(x, delta, maxiters=1000) -> float:  # this one is different.
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1].clone()
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1)

    def _Kn(mu):
        kn = np.cumsum(np.log(1 - nu * (x - mu))).max() + np.log(delta)
        return kn

    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1 - 1e-10, maxiter=maxiters)


def wsr_ucb(loss: Tensor, delta: float) -> float:
    """The WSR upper confidence bound.

    Args:
        loss (Tensor): A tensor of loss values with shape (N,), where N is the
                       number of examples.
        delta (float): The probability of error for the UCB.

    Returns:
        float: The UCB value.
    """
    assert loss.dim() == 1
    assert (loss.min().item() >= 0.0) and (loss.max().item() <= 1.0), "Loss values must be in the range [0, 1]"
    return float(WSR_mu_plus(loss, delta))


def hoeffding_bentkus_ucb(loss: Tensor, delta: float, maxiters: int=1000) -> float:
    """The Hoeffding-Bentkus upper confidence bound for losses in [0, 1].

    Args:
        loss (Tensor): A tensor of loss values with shape (N), where N is the
                       number of examples.
        delta (float): The probability of error for the UCB.
        maxiters (int, optional): The maximum number of iterations for brentq. Defaults to 1000.

    Returns:
        float: The UCB value.
    """
    assert loss.dim() == 1
    assert (loss.min().item() >= 0.0) and (loss.max().item() <= 1.0), "Loss values must be in the range [0, 1]"

    n = loss.size(-1)

    r_hat = torch.mean(loss)

    def g_hb(r):
        h_value = np.exp(-n * h1(r_hat, r))
        b_value = np.e * binom.cdf(np.ceil(n * r_hat), n, r)
        return np.minimum(b_value, h_value)

    def g_hb_diff(r):
        return g_hb(r) - delta

    if g_hb_diff(1.0) > 0.0:
        return 1.0
    else:
        return brentq(g_hb_diff, r_hat, 1.0, maxiter=maxiters)


def h1(y, mu):
    return rel_entr(y, mu) + rel_entr(1 - y, 1 - mu)


def hb_p_value(r_hat, n, alpha):
    h_p_value = np.exp(-n * h1(np.minimum(r_hat, alpha), alpha))
    b_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    return np.minimum(b_p_value, h_p_value)


def WSR_mu_plus(x, delta, maxiters=1000) -> float:  # this one is different.
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1].clone()
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1)

    def _Kn(mu):
        kn = np.cumsum(np.log(1 - nu * (x - mu))).max() + np.log(delta)
        return kn

    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1 - 1e-10, maxiter=maxiters)


def wsr_ucb(loss: Tensor, delta: float) -> float:
    """The WSR upper confidence bound.
    Args:
        loss (Tensor): A tensor of loss values with shape (N,), where N is the
                       number of examples.
        delta (float): The probability of error for the UCB.
    Returns:
        float: The UCB value.
    """
    assert loss.dim() == 1
    assert (loss.min().item() >= 0.0) and (loss.max().item() <= 1.0), "Loss values must be in the range [0, 1]"
    return float(WSR_mu_plus(loss, delta))


def find_rcps_index(loss: Tensor, delta: float, target: float, ucb_fun: Callable[[Tensor, float], float]) -> int:
    """Template for RCPS that takes an arbitrary UCB function.

    Args:
        loss (Tensor): A tensor of loss values with shape (H, N), where H is
                       the number of hypotheses and N is the number of examples.
        delta (float): The probability of error for the UCB.
        target (float): The target maximum acceptable loss value.
        ucb_fun (Callable[[Tensor, float], Tensor]): An arbitrary UCB function.

    Returns:
        int: The largest hypothesis index that can be guaranteed to satisfy risk control.
    """
    assert loss.dim() == 2

    def compute_ucb(x: Tensor) -> float:
        # squeezes x and calls ucb_fun, with some sanity checks
        assert x.dim() == 2
        assert x.size(0) == 1
        ucb_value = ucb_fun(x.squeeze(0), delta)
        # assert ucb_value >= torch.mean(x).item(), "UCB must be at least the sample mean"
        return ucb_value

    return bisect.bisect_right(torch.split(loss, 1, dim=0), target, key=compute_ucb) - 1


def find_rcps_ucb(loss: Tensor, delta: float, target: float, ucb_fun: Callable[[Tensor, float], float]) -> int:
    """Template for RCPS that takes an arbitrary UCB function.

    Args:
        loss (Tensor): A tensor of loss values with shape (H, N), where H is
                       the number of hypotheses and N is the number of examples.
        delta (float): The probability of error for the UCB.
        target (float): The target maximum acceptable loss value.
        ucb_fun (Callable[[Tensor, float], Tensor]): An arbitrary UCB function.

    Returns:
        int: The largest hypothesis index that can be guaranteed to satisfy risk control.
    """
    assert loss.dim() == 2
    def compute_ucb(x: Tensor) -> float:
        # squeezes x and calls ucb_fun, with some sanity checks
        assert x.dim() == 2
        assert x.size(0) == 1
        ucb_value = ucb_fun(x.squeeze(0), delta)
        # assert ucb_value >= torch.mean(x).item(), "UCB must be at least the sample mean"
        return ucb_value
    
    alpha_vals = torch.Tensor(
            [compute_ucb(loss_vec) for loss_vec in
             loss.split(1)])
    return alpha_vals


@dataclass(frozen=True)
class RcpsWSR:
    @staticmethod
    def fit_risk(
            loss: Tensor,
            target: float,
            delta: float,
    ) -> int:

        ucb = find_rcps_ucb(loss, delta, target, wsr_ucb)
        if (ucb < target).sum(-1) == 0:
            hypothesis_ind = None
            alpha = None
        else:
            hypothesis_ind = torch.argmin(ucb)
            alpha = torch.min(ucb)
            
        return hypothesis_ind, alpha.item()


@dataclass(frozen=True)
class Rcps:
    @staticmethod
    def fit_target_risk(
            loss: Tensor,
            delta: float,
            target: float,
    ) -> int:
        return find_rcps_index(loss, delta, target, hoeffding_ucb)


@dataclass(frozen=True)
class RcpsHB:
    @staticmethod
    def fit_target_risk(
            loss: Tensor,
            delta: float,
            target: float,
    ) -> int:
        return find_rcps_index(loss, delta, target, hoeffding_bentkus_ucb)


def binomial_ucb(alpha, loss_vec, delta):
    # max possible loss
    ALPHA_MAX = 1.0

    N = loss_vec.size(-1)
    S_alpha = loss_vec.gt(alpha).sum()

    ret = 1 - scipy.special.betaincinv(N - S_alpha, S_alpha + 1, delta)
    if torch.isnan(ret).any():
        return ALPHA_MAX
    else:
        return ret


def rcps_binomial_alpha(loss_vec, beta, delta):
    alpha_vals = torch.unique(loss_vec, sorted=True)
    ucb_vals = torch.Tensor([binomial_ucb(alpha, loss_vec, delta) for alpha in alpha_vals.tolist()])
    return alpha_vals[torch.nonzero(ucb_vals < 1 - beta).reshape(-1)[0].item()]


@dataclass(frozen=True)
class RcpsBinomialBonferroni:
    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)

        correction = delta / (num_hypotheses)

        alpha_vals = torch.Tensor(
            [rcps_binomial_alpha(loss_vec.squeeze(0), beta=beta, delta=correction) for loss_vec in
             loss.split(1)])
        hypothesis_ind = torch.argmin(alpha_vals, -1)
        alpha = alpha_vals[hypothesis_ind]

        return int(hypothesis_ind.item()), float(alpha.item())


@dataclass(frozen=True)
class RcpsBinomial:
    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)

        alpha_vals = torch.Tensor(
            [rcps_binomial_alpha(loss_vec.squeeze(0), beta=beta, delta=(delta / num_hypotheses)) for loss_vec in
             loss.split(1)])
        hypothesis_ind = torch.argmin(alpha_vals, -1)
        alpha = alpha_vals[hypothesis_ind]

        return int(hypothesis_ind.item()), float(alpha.item())


@dataclass(frozen=True)
class Ltt:
    @staticmethod
    def fit_target_risk(
            loss: Tensor,
            delta: float,
            target: float,
    ) -> int:
        n = loss.shape[1]
        # compute mean risk per threhold
        risk = loss.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = torch.exp(-2 * n * (torch.relu(target - risk) ** 2))

        hypothesis_ind = int(
            torch.nonzero(pvals < delta / loss.shape[0]).view(-1).max().item()
        )

        return hypothesis_ind

    @staticmethod
    def fit_risk(
            metric: Tensor,
            target: float,
            delta: float,
    ) -> Optional[Tensor]:

        n = metric.shape[1]
        # compute mean risk per threhold
        risk = metric.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = torch.exp(-2 * n * (torch.relu(target - risk) ** 2))
        # perform multiple testing with Bonferroni correction
        lambda_hats = risk[pvals < delta / risk.shape[0]]
        if lambda_hats.nelement() == 0:
            hypothesis_ind = None
        else:
            hypothesis_ind = torch.argmin(risk)

        return hypothesis_ind


@dataclass(frozen=True)
class LttHB:
    @staticmethod
    def fit_target_risk(
            loss: Tensor,
            delta: float,
            target: float,
    ) -> int:
        n = loss.shape[1]
        # compute mean risk per threhold
        risk = loss.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = hb_p_value(risk, n, target)

        hypothesis_ind = int(
            torch.nonzero(pvals < delta / loss.shape[0]).view(-1).max().item()
        )

        return hypothesis_ind

    @staticmethod
    def fit_risk(
            metric: Tensor,
            target: float,
            delta: float,
    ) -> Optional[Tensor]:
        n = metric.shape[1]
        # compute mean risk per threhold
        risk = metric.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = hb_p_value(risk, n, target)

        # perform multiple testing with Bonferroni correction
        lambda_hats = risk[pvals < delta / risk.shape[0]]
        if lambda_hats.nelement() == 0:
            hypothesis_ind = None
            ucb = None
        else:
            hypothesis_ind = torch.argmin(risk)
            ucb = hoeffding_bentkus_ucb(metric[hypothesis_ind], delta / risk.shape[0])

        return hypothesis_ind, ucb
    
    @staticmethod
    def fit_risk_inflate(
            metric: Tensor,
            target: float,
            delta: float,
    ) -> Optional[Tensor]:
        n = metric.shape[1]
        # compute mean risk per threhold
        risk = metric.permute(1, 0).mean(dim=0)
        # compute p-value
        pvals = hb_p_value(risk, n, target)

        # perform multiple testing with Bonferroni correction
        lambda_hats = risk[pvals < delta / risk.shape[0]]
        if lambda_hats.nelement() == 0:
            hypothesis_ind = None
            mean_alpha = None
        else:
            hypothesis_ind = torch.argmin(risk)
            ucb = hoeffding_bentkus_ucb(metric[hypothesis_ind], delta / risk.shape[0])

        return hypothesis_ind, ucb


@dataclass(frozen=True)
class VarControl:
    @staticmethod
    def fit_target_var(
            loss: Tensor,
            delta: float,
            beta: float,
            target: float,
    ) -> int:
        # assumption is that loss increases across the rows, so choose the largest
        # hypothesis index that satisfies the quantile constraint
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)
        inflated_beta = beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        inflated_quantile = torch.quantile(loss, inflated_beta, dim=-1, interpolation="higher")

        # select hypothesis based on quantile
        hypothesis_ind = int(
            torch.nonzero(inflated_quantile <= target).view(-1).max().item()
        )

        return hypothesis_ind

    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        num_hypotheses = loss.size(0)
        num_train = loss.size(-1)
        inflated_beta = beta + math.sqrt(
            math.log(num_hypotheses / delta) / (2 * num_train)
        )
        inflated_quantile = torch.quantile(loss, inflated_beta, dim=-1, interpolation="higher")

        hypothesis_ind = torch.argmin(inflated_quantile, -1)
        alpha = inflated_quantile[hypothesis_ind]

        return int(hypothesis_ind.item()), float(alpha.item())


def calculate_k(n, delta, beta, method):
    if method == "DKW":
        k = np.ceil(n * (beta + np.sqrt(-np.log(delta) / (2 * n)))) - 1
    elif method == "OrderStats":
        k = binom.ppf(1 - delta, n, beta) + 1 - 1
    elif method == "Inflation":
        k = np.ceil((1 - (1 - beta) * delta) * (n + 1)) - 1
    else:
        raise ValueError
    return k


def get_beta_var_monotonic(k, train_loss, target):
    sorted_loss, _ = torch.sort(train_loss, 1)
    alpha_vals = sorted_loss[:, k]
    hypothesis_ind = int(
        torch.nonzero(alpha_vals <= target).view(-1).max().item()
    )
    return hypothesis_ind


def get_beta_var_general(k, train_loss):
    sorted_loss, _ = torch.sort(train_loss, 1)
    alpha_vals = sorted_loss[:, k]
    hypothesis_ind = torch.argmin(alpha_vals, -1)
    alpha = alpha_vals[hypothesis_ind]
    return int(hypothesis_ind.item()), float(alpha.item())

        
def calc_front(k, loss, no_h, n):        
    max_k = k.max().item()
    sorted_loss, _ = torch.sort(loss, 1)
    if max_k >= n:
        extra_col = torch.ones((no_h,int(max_k-n+1)))
        sorted_loss = torch.concat([sorted_loss, extra_col], 1)
    alpha_vals = sorted_loss[:, k]
    return alpha_vals


@dataclass(frozen=True)
class DKW:

    @staticmethod
    def fit_target_var(
            loss: Tensor,
            delta: float,
            beta: float,
            target: float,
    ) -> int:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "DKW")
        return get_beta_var_monotonic(int(k), loss, target)

    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "DKW")
        return get_beta_var_general(int(k), loss)
    
    @staticmethod
    def fit_front(
            loss: Tensor,
            delta: float,
            beta_vals: float,
    ) -> Tuple[int, float]:
        n = loss.shape[1]
        no_h = loss.shape[0]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta_vals, "DKW")
        return calc_front(k, loss, no_h, n)


@dataclass(frozen=True)
class OrderStats:

    @staticmethod
    def fit_target_var(
            loss: Tensor,
            delta: float,
            beta: float,
            target: float,
    ) -> int:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "OrderStats")
        return get_beta_var_monotonic(int(k), loss, target)

    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "OrderStats")
        return get_beta_var_general(int(k), loss)
    
    @staticmethod
    def fit_front(
            loss: Tensor,
            delta: float,
            beta_vals: float,
    ) -> Tuple[int, float]:
        n = loss.shape[1]
        no_h = loss.shape[0]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta_vals, "OrderStats")
        return calc_front(k, loss, no_h, n)


class Inflation:

    @staticmethod
    def fit_target_var(
            loss: Tensor,
            delta: float,
            beta: float,
            target: float,
    ) -> int:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "Inflation")
        return get_beta_var_monotonic(int(k), loss, target)

    @staticmethod
    def fit_var(
            loss: Tensor,
            delta: float,
            beta: float,
    ) -> Tuple[int, float]:
        n = loss.shape[1]
        delta = delta / loss.shape[0]
        k = calculate_k(n, delta, beta, "Inflation")
        return get_beta_var_general(int(k), loss)