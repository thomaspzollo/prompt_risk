import argparse
from argparse import Namespace
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Tuple, Optional
import math
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from scipy.stats import binom, norm
import scipy.stats
import pickle as pkl
import os

from prompt_risk.methods import (
    VarControl,
    RcpsBinomial,
    DKW,
    OrderStats,
    Inflation,
    LttHB,
    RcpsHB
)

from prompt_risk.utils import *
from prompt_risk.bounds import *


def load_loss(args):

    dataset = args.dataset
    model_size = args.model_size
    loss_fn = args.loss_fn

    load_folder = "../output/{}".format(
        dataset, 
    )
    load_root = "{}/{}_model_{}_{}_loss_dist.pkl".format(
        load_folder,
        dataset, 
        model_size, 
        loss_fn
    )
    print("loading from", load_root)
    
    with open(load_root, 'rb') as file:
        res = pkl.load(file)

    instructions = [r[0] for r in res]
    X = np.array([r[1] for r in res])
    return instructions, X


@dataclass(frozen=True)
class GeneralTrialResult:
    quantile_loss: float
    # mean_in_quantile_loss: float
    # mean_in_quantile_pred_size: float
    # quantile_satisfied: bool
    # mean_loss: float
    # mean_pred_size: float
    # mean_satisfied: bool
    # mean_alpha: float
    quantile_alpha: float


def run_general_trial(
    X,
    X_test,
    method_dict,
    alpha,
    beta,
    delta,
    markov_scaling,
):
    def get_result(method):
        if hasattr(method, "fit_var"):
            hypothesis_ind, quantile_alpha = method.fit_var(
                X, delta, beta
            )
            mean_alpha = math.nan
        else:
            if markov_scaling:
                hypothesis_ind = method.fit_risk(
                    X, alpha * (1 - beta), delta
                )
                mean_alpha = math.nan
                quantile_alpha = alpha
            else:
                hypothesis_ind, mean_alpha = method.fit_risk(X, alpha, delta)
                if not mean_alpha:
                    mean_alpha = alpha
                # mean_alpha = alpha
                # quantile_alpha = math.nan
                quantile_alpha = mean_alpha/(1-beta)

        # compute the test loss
        test_loss = X_test[hypothesis_ind]

        quantile_loss = torch.quantile(test_loss, beta).item()
        quantile_satisfied = quantile_loss <= quantile_alpha

        inds = quantile_indices(test_loss, beta)
        mean_in_quantile_loss = test_loss[inds].mean(-1).item()

        return GeneralTrialResult(
            quantile_loss=quantile_loss,
            quantile_alpha=quantile_alpha,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


def calculate_oracle(test_loss, beta, loss_type="general", alpha=None):

    quantile = torch.quantile(test_loss, beta, dim=-1, interpolation="higher")
    if loss_type == "general":
        hypothesis_ind = torch.argmin(quantile, -1)
        oracle_var = quantile[hypothesis_ind].item()
    else:
        hypothesis_ind = int(
            torch.nonzero(quantile <= alpha).view(-1).max().item()
        )
        oracle_var = quantile[hypothesis_ind].item()
    return hypothesis_ind, oracle_var


def quantile_indices(x: Tensor, beta: float) -> Tensor:
    assert x.dim() == 1
    n = x.size(0)
    n_trunc = math.floor(n * beta)
    _, inds = torch.sort(x)
    return inds[:n_trunc]


def main(args: Namespace):

    torch.manual_seed(args.seed)
    
    output_dir = "../output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)
    
    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
    ]

    methods = [
        DKW,
        OrderStats,
        LttHB,
    ]

    instructions, loss = load_loss(args)
    args.num_hypotheses = loss.shape[0]

    print("loss shape", loss.shape)

    save_string = "{}_{}_var_beta_{}_no_data_{}".format(args.dataset, args.loss_fn, args.beta, args.num_val_datapoints)
    print(save_string)
    
    method_dict = OrderedDict([(method.__name__, method) for method in methods])

    trial_results = []
    oracle_vars = []
    
    for trial_idx in tqdm(range(args.num_trials)):

        rand_idx = torch.randperm(loss.shape[1])
        train_idx = rand_idx[:args.num_val_datapoints]
        test_idx = rand_idx[args.num_val_datapoints:]
        
        X = torch.Tensor(loss[:, train_idx])
        X_test = torch.Tensor(loss[:, test_idx])

        correction = X.shape[0]
        n = X.shape[-1]

        
        oracle_hypothesis_ind, oracle_var = calculate_oracle(X_test, args.beta, "general")
        oracle_loss = X_test[oracle_hypothesis_ind]
        # oracle_pred_size = test_batch.pred_size[oracle_hypothesis_ind]
        oracle_inds = quantile_indices(oracle_loss, args.beta)
        # oracle_mean_in_quantile_pred_size = oracle_pred_size[oracle_inds].float().mean(-1).item()
        oracle_vars.append(oracle_var)
        # oracle_set_sizes.append(oracle_mean_in_quantile_pred_size)

        trial_results.append(
            run_general_trial(
                X,
                X_test,
                method_dict,
                alpha=args.alpha,
                beta=args.beta,
                delta=args.delta,
                markov_scaling=args.markov_scaling,
            )
        )

        n = X.shape[-1]
        bound_res = OrderedDict()

        for bound_item in bound_list:

            bound_name = bound_item.name
            bound_fn = bound_item.bound_fn

            if bound_item.b is not None:
                b = bound_item.b
            else:
                if bound_name in ["KS", "BJ"]:
                    b = bound_fn(n, args.delta/correction)
                else:
                    raise ValueError
                bound_item.b = b
                
            X_sorted = np.sort(X, axis=-1)
            quantile_alphas = X_sorted[:, (b < args.beta).astype(int).sum()]
            hyp_ind = np.argmin(quantile_alphas)
            alpha = np.min(quantile_alphas)
            test_loss = X_test[hyp_ind].numpy()
            quantile_loss = np.quantile(test_loss, args.beta).item()
            res = GeneralTrialResult(
                quantile_loss=quantile_loss,
                quantile_alpha=alpha,
            )
            bound_res[bound_name]=res

        trial_results[-1].update(bound_res)

    rows = []
    for trial_ind, trial_result in enumerate(trial_results):
        for k, v in trial_result.items():
            rows.append({"trial": trial_ind + 1, "method": k} | asdict(v))

    results_df = pd.DataFrame(rows)
    avg_results_df = results_df.drop(columns="trial").groupby(["method"]).mean()
    avg_results_df['oracle_var'] = sum(oracle_vars)/len(oracle_vars)
    
    if args.save_csv:
        print("saving...")
        results_df.to_csv("{}/{}_full_results.csv".format(output_dir, save_string))
        avg_results_df.to_csv("{}/{}.csv".format(output_dir, save_string))
        args_dict = vars(args)
        with open("{}/{}.pkl".format(output_dir, save_string), "wb") as handle:
            pkl.dump(args_dict, handle)
        print(args_dict)
        
    print(avg_results_df)
    if args.show_latex:
        print(avg_results_df.to_latex(float_format="%.3f"))

    if args.show_std:
        std_results_df = results_df.drop(columns="trial").groupby(["method"]).std()
        print(std_results_df)
        if args.show_latex:
            print(std_results_df.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VaR experiments")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="number of random splits (default: 1000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="target loss value (default: 0.25)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.8, help="target quantile level (default: 0.9)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    parser.add_argument(
        "--markov_scaling",
        action="store_true",
        help="use Markov's inequality to convert risk controlling to var controlling",
    )
    parser.add_argument(
        "--show_latex",
        action="store_true",
        help="display latex output"
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="display standard deviation results"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true"
    )
    parser.add_argument(
        "--dataset",
        default="red_team_chat",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--model_size",
        default="base",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--loss_fn",
        default="toxicity",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--num_val_datapoints",
        type=int,
        default=1000,
        help="number of validation datapoints",
    )
    args = parser.parse_args()
    main(args)