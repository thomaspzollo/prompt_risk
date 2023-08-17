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

    torch.manual_seed(args.seed)

    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
    ] 
    methods = [
        LttHB,
        RcpsWSR
    ]

    instructions, loss = load_loss(args)

    print("loss shape", loss.shape)

    save_string = "{}_{}_mean_no_data_{}".format(args.dataset, args.loss_fn, args.num_val_datapoints)
    print(save_string)

    method_dict = OrderedDict([(method.__name__, method) for method in methods])
    trial_results = []

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
                else:
                    raise ValueError
                bound_item.b = b

            aucs = integrate_quantiles(X, b, beta_min=0.0, beta_max=1.0)
            hyp_ind = np.argmin(aucs)
            auc = np.min(aucs)

            test_loss = X_test[hyp_ind].numpy()
            mean_loss = np.mean(test_loss)

            trial_results.append((trial_idx, bound_name, auc, mean_loss))

        for method_name, method in method_dict.items():

            if hasattr(method, "fit_risk"):
                hyp_ind, mean_alpha = method.fit_risk(X, 1.0, args.delta)
            else:
                raise NotImplementedError
                
            test_loss = X_test[hyp_ind].numpy()
            mean_loss = np.mean(test_loss)

            trial_results.append((trial_idx, method_name, mean_alpha, mean_loss))

    results_df = pd.DataFrame(trial_results, columns=["trial", "method", "alpha", "mean loss"])
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
    parser = argparse.ArgumentParser(description="Run mean bounding experiments")

    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="random seed (default: 0)"
    )
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=50,
        help="number of hypotheses (default: 500)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="number of random splits (default: 1000)",
    )
    parser.add_argument(
        "--num_val_datapoints",
        type=int,
        default=500,
        help="number of validation points",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
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
    args = parser.parse_args()
    main(args)