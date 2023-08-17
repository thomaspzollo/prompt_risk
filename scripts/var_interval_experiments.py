import math
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

from prompt_risk.methods import (
    DKW,
    OrderStats,
    LttHB,
    RcpsWSR
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


def main(args):
    
    output_dir = "../output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)
    
    beta_lo = args.beta_lo
    beta_hi = args.beta_hi

    torch.manual_seed(args.seed)

    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
        Bound("One-sided-BJ", berk_jones_one_sided)
    ] 
    
    if beta_hi < 1.0:
        bound_list.append(
            Bound("Two-sided-BJ", berk_jones_two_sided)
        )

    methods = [
        DKW,
        OrderStats,
    ]

    save_dir = f"output/{args.dataset}_experiments"

    instructions, loss = load_loss(args)

    print("loss shape", loss.shape)
    args.num_hypotheses = loss.shape[0]
        
    if args.fixed_pred:
        aucs = integrate_quantiles(
                    loss.T,
                    np.arange(1, loss.shape[0] + 1) / loss.shape[0], beta_min=beta_lo, beta_max=beta_hi
                )
        best_hyp = np.argmin(aucs)
        thresholds = thresholds[best_hyp:best_hyp+1]
        loss = loss[:, best_hyp:best_hyp+1]
        preds = preds[:, best_hyp:best_hyp+1]
        args.num_hypotheses = 1
        
    beta_grid_size = args.grid_size
    tolerance = -1e-6

    save_string = "{}_{}_beta_lo_{}_beta_hi_{}_no_data_{}_grid_size_{}".format(args.dataset, args.loss_fn, beta_lo, beta_hi, args.num_val_datapoints, beta_grid_size)
    if args.fixed_pred:
        save_string += "_fixed_pred"
    print(save_string)

    method_dict = OrderedDict([(method.__name__, method) for method in methods])
    trial_results = []
    trial_idx = 0

    for trial_idx in tqdm(range(args.num_trials)):

        rand_idx = torch.randperm(loss.shape[1])
        train_idx = rand_idx[:args.num_val_datapoints]
        test_idx = rand_idx[:args.num_val_datapoints]
        
        X = torch.Tensor(loss[:, train_idx])
        X_test = torch.Tensor(loss[:, test_idx])

        correction = X.shape[0]
        n = X.shape[-1]

        for bound_item in bound_list:

            bound_name = bound_item.name
            bound_fn = bound_item.bound_fn

            if bound_item.b is not None:
                b = bound_item.b
            else:
                if bound_name in ["KS", "BJ"]:
                    b = bound_fn(n, args.delta/correction)
                elif bound_name == "One-sided-BJ":
                    b = bound_fn(n, args.delta/correction, q_min=beta_lo)
                elif bound_name == "Two-sided-BJ":
                    b = bound_fn(n, args.delta/correction, q_min=beta_lo, q_max=beta_hi)
                else:
                    raise ValueError
                bound_item.b = b

            aucs = integrate_quantiles(X, b, beta_min=beta_lo, beta_max=beta_hi)
            hyp_ind = np.argmin(aucs)
            auc = np.min(aucs)

            test_loss = X_test[hyp_ind]
            X_sorted = np.sort(X, axis=-1)
            test_cdf = ecdf(X_sorted[hyp_ind], test_loss)
            violation = int(np.any(
                test_cdf < b,
                axis=-1
            ))
            
            test_int_auc = integrate_quantiles(
                torch.Tensor(np.expand_dims(test_loss, 0)),
                np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta_lo, beta_max=beta_hi
            )[0]
            
            trial_results.append((trial_idx, bound_name, auc, test_int_auc, violation))

            x = list(X_sorted[hyp_ind])+[1.0,1.0]
            b_0 = list(b)
            b_0.extend([b_0[-1],1.0])

        for method_name, method in method_dict.items():

            beta_vals = np.linspace(beta_lo, beta_hi, beta_grid_size)

            if hasattr(method, "fit_front"):
                bounded_region = method.fit_front(
                    X, args.delta/beta_grid_size, beta_vals
                ).numpy()
                mean_alpha = math.nan
            else:
                raise NotImplementedError

            assert bounded_region.shape[-1] == beta_grid_size
            assert bounded_region.max().item()<=1.0
            aucs = integrate_quantiles(bounded_region, beta_vals, beta_min=beta_lo, beta_max=beta_hi)
            hyp_ind = np.argmin(aucs)
            auc = aucs[hyp_ind].item()

            test_loss = X_test[hyp_ind]
            test_cdf = ecdf(bounded_region[hyp_ind], test_loss)
            violation = int(np.any(
                test_cdf < beta_vals,
                axis=-1
            ))
            
            test_int_auc = integrate_quantiles(
                torch.Tensor(np.expand_dims(test_loss, 0)),
                np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta_lo, beta_max=beta_hi
            )[0]
            
            trial_results.append((trial_idx, method_name+"Bonferroni", auc, test_int_auc, violation))

    results_df = pd.DataFrame(
        trial_results, 
        columns=["trial", "method", "guaranteed_auc", "empirical_auc", "lcb_violation"]
    )
    average_df = results_df.drop(columns="trial").groupby(["method"]).mean()
    if args.save_csv:
        print("saving df to csv...")
        results_df.to_csv("{}/{}_full_results.csv".format(output_dir, save_string))
        average_df.to_csv("{}/{}.csv".format(output_dir, save_string))
        args_dict = vars(args)
        with open("{}/{}.pkl".format(output_dir, save_string), "wb") as handle:
            pkl.dump(args_dict, handle)
        print(args_dict)
    print(average_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interval experiments")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="number of random splits (default: 1000)",
    )
    parser.add_argument(
        "--num_val_datapoints",
        type=int,
        default=1000,
        help="number of validation datapoints",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=100,
        help="size of beta grid",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    parser.add_argument(
        "--beta_lo",
        type=float,
        default=0.85,
        help="minimum quantile in interval",
    )
    parser.add_argument(
        "--beta_hi",
        type=float,
        default=0.95,
        help="maximum quantile in interval",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="store results"
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
        "--fixed_pred",
        action="store_true",
        help="fix predictor"
    )
    args = parser.parse_args()
    main(args)