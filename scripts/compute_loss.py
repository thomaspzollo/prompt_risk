"""This module produces loss values for model generated outputs.

Examples:
    $ python -m scripts.compute_loss \
        --output-dir output \
        --datasets red_team_chat full_chat \
        --loss-fn weqweasdas/hh_rlhf_rm_open_llama_3b \
        --batch-size 30 \
        --eval-models google/flan-t5-xxl
    
    $ python -m scripts.compute_loss \
        --output-dir output \
        --datasets mbpp \
        --loss-fn pass@k
"""
import argparse
import math
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import evaluate
from transformers import AutoTokenizer, pipeline
import torch
from detoxify import Detoxify

from scripts.generate_outputs import prepare_test_code
from scripts.args import parse_args
from .code_eval.evaluation import evaluate_functional_correctness, calculate_pass_at_k

def get_scorer(args):
    if args.loss_fn == "bleu":
        scorer = evaluate.load("bleu")
    elif args.loss_fn == "bertscore":
        scorer = evaluate.load("bertscore")
    elif args.loss_fn == "rouge":
        scorer = evaluate.load("rouge")
    elif args.loss_fn == "toxicity":
        scorer = Detoxify(device=args.device)
    elif args.loss_fn == "accuracy":
        scorer = None
    elif args.loss_fn == "weqweasdas/hh_rlhf_rm_open_llama_3b":
        # TODO: determine if this can be run on multiple GPUs, this can take about 15 minutes for 100,000 examples
        rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
        scorer = pipeline(
            "sentiment-analysis",
            model="weqweasdas/hh_rlhf_rm_open_llama_3b",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            tokenizer=rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
    elif args.loss_fn == "pass@k":
        scorer = None
    else:
        raise ValueError
    return scorer

def get_scores(args, scorer, chat_out, batch):

    if args.loss_fn == "toxicity":
        tox_out = scorer.predict(chat_out)
        scores = torch.hstack([torch.Tensor(v).unsqueeze(1) for v in tox_out.values()])
        scores = torch.max(scores,-1)[0].detach().cpu().tolist()

    elif args.loss_fn == "bertscore":
        if args.dataset == "healthcare":
            refs = batch["output"]
        else:
            refs = batch["summary"]

        for k in range(len(chat_out)):
            print(k)
            print()
            print(refs[k])
            print()
            print(chat_out[k])
            e += 7
        scores = scorer.compute(
            predictions=chat_out,
            references=refs,
            lang="en",
            device=args.device,
        )["f1"]
        scores = list(1-np.array(scores))
        scores = list(scores)

    elif args.loss_fn == "rouge":
        scores = scorer.compute(
            predictions=chat_out,
            references=batch["output"],
            use_aggregator=False
        )["rougeL"]
        print(scores)
        e += 7
        scores = list(1-np.array(scores))
        scores = list(scores)

    elif args.loss_fn == "accuracy":

        preds = [c.lower() for c in chat_out]
        labels = [b.split("$")[1].lower() for b in batch]
        assert len(preds) == len(labels)
        # print(preds, labels)
        scores = []
        for i in range(len(labels)):

            if preds[i] != labels[i]:
                scores.append(1.0)
            else:
                scores.append(0.0)

            if preds[i] not in ["yes", "no"]:
                print("Warning, bad output:", preds[i])

    else:
        raise NotImplementedError

    return scores

def compute_chat_loss(df, reward_model_pipeline, args):
    final_loss_col = "weqweasdas/hh_rlhf_rm_open_llama_3b_eval_reward_med_sigmoid"
    dataset = df['dataset'].iloc[0]
    # check if we've already computed loss for all rows in this df
    if final_loss_col in df.columns and (df[final_loss_col].isna().sum() == 0):
        print(f"Loss ({final_loss_col}) already computed for {df.shape[0]} rows in {dataset}. Skipping.")
        return df, reward_model_pipeline
    
    if reward_model_pipeline is None:
        # lazy load the reward model pipeline
        print("Loading reward model pipeline...")
        reward_model_pipeline = get_scorer(args)
        
    # human input is between Here is a human input: and \nChatbot Response: 
    df['human_input'] = df['text'].str.split('Here is a human input: ').str[1].str.split('\nChatbot Response:').str[0].str.strip()
    # generated_text is model output
    # prepend 'human_input' column with '###Human: ', prepend 'generated_text' column with '###Assistant: ', then concatenate
    df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_text'] = '###Human: ' + df['human_input'].astype(str) + ' ###Assistant: ' + df['generated_text'].astype(str)

    # compute loss
    pipe_kwargs = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": args.batch_size # 65 # tuned for a single 4090
    }
    print(f"Warning: long sequences will be truncated to {reward_model_pipeline.tokenizer.model_max_length} tokens. Consider using a different reward model if this is a problem.")
    tokenizer_kwargs={'truncation': True}
    pipe_kwargs.update(tokenizer_kwargs)
    start = time.perf_counter()
    # TODO: identify rows that have already been computed and only compute loss for the remaining rows
    pipe_outputs = reward_model_pipeline(df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_text'].tolist(), **pipe_kwargs)
    end = time.perf_counter()
    print(f"Time to compute loss for {len(df):,} rows: {end-start:.2f} seconds. Average time per example: {(end-start)/len(df):.2f} seconds.")
    rewards = [output[0]["score"] for output in pipe_outputs]
    df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_reward'] = rewards
    # examine distribution of loss values
    print(df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_reward'].describe())
    # pass through a shifted sigmoid with center at median and scale to [0,1]
    median = df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_reward'].median()
    df[final_loss_col] = 1 / (1 + df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_reward'].apply(lambda x: math.exp(-1*(x-median))))
    return df, reward_model_pipeline


def score_mbpp_predictions(df):
    # TODO: create unified interface for loading data the same way in compute_loss.py and generate_outputs.py
    dataset = load_dataset("mbpp")
    # concat train, validation, and test portions
    dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    label_df = dataset.to_pandas()
    # drop the text column because it will conflict df's text column
    label_df = label_df.drop(columns=["text"])
    # merge with predictions on task_id
    df = df.merge(label_df, on="task_id")
    # extract python code from 'generated_text' column
    df['generated_code'] = df['generated_text'].apply(lambda x: x.split('[PYTHON]')[1].strip().split('[/PYTHON]')[0].strip())
    df['test_code'] = df.apply(lambda x: prepare_test_code(x), axis=1)
    
    # need to add a completion_id column to df for a unique identifier while we're in this function
    df['completion_id'] = range(len(df))
    # identify which rows have already been scored for functional correctness and only score the remaining rows
    if 'passed' not in df.columns:
        # all rows have not been scored for functional correctness
        score_df = df[['completion_id', 'generated_code', 'test_code']]
    else:
        # some rows have already been scored for functional correctness
        score_df = df[df['passed'].isna()][['completion_id', 'generated_code', 'test_code']]
    # convert to a list of dictionaries
    scores_needed = score_df.to_dict(orient="records")
    print(f"Calculating functional correctness for {len(scores_needed):,}/{len(df):,} rows. {len(df)-len(scores_needed):,} rows already scored.")
    if len(scores_needed) > 0:
        workers = mp.cpu_count()
        start = time.perf_counter()
        results = evaluate_functional_correctness(scores_needed, n_workers=workers, timeout=3.0)
        end = time.perf_counter()
        print(f"Time to compute functional correctness for {len(scores_needed):,} rows: {end-start:.2f} seconds. Average time per example: {(end-start)/len(scores_needed):.2f} seconds.")
        # convert results to a dataframe
        results_df = pd.DataFrame(results)
        # identify rows in df that were affected by the functional correctness check
        # so that we can merge the results back into this particular portion of the df
        results_df.set_index('completion_id', inplace=True)
        df = df.set_index('completion_id')
        df.loc[results_df.index, ['passed', 'result']] = results_df[['passed', 'result']]
        # reset index and drop
        df = df.reset_index(drop=True)
    # drop unnecessary columns from mbpp 'code', 'test_list', 'test_setup_code', 'challenge_test_list'
    df = df.drop(columns=['code', 'test_list', 'test_setup_code', 'challenge_test_list', 'completion_id'], errors='ignore')
    return df

def main(args):
    # TODO: extract this into a function that can be called from other scripts

     # allow user to specify a collection of datasets and then loop over them
    if not args.datasets:
        args.datasets = [args.dataset]

    for dataset in args.datasets:
        # lazily load the reward model pipeline (we might have already computed loss scores, in which case, save time on the load time)
        scorer = None
        args.dataset = dataset
        # load generated outputs for specified models
        models = [x.replace("/","-") for x in args.eval_models]
        # if no models specified, load all models
        if not models:
            files = os.listdir(os.path.join(args.output_dir, args.dataset))
            files = [f for f in files if f.endswith(".csv")]
        else:
            files = [f"{m}_predictions.csv" for m in models]
        dfs = {}
        for f in files:
            model_id = f.split("_predictions.csv")[0]
            df = pd.read_csv(os.path.join(args.output_dir, args.dataset, f))
            dfs[model_id] = df

            # check if all models have the expected number of rows populated with text
            if dfs[model_id].shape[0] != 50*2000:
                print(f"Warning: {model_id} has {dfs[model_id].shape[0]} rows, expected 50*2000=100,000")
            # check for any nans in generated_text
            assert dfs[model_id].generated_text.isna().sum() == 0

            # compute loss for each model
            print(f"Computing loss for {model_id}")
            if 'chat' in args.dataset:
                dfs[model_id], scorer = compute_chat_loss(df, scorer, args)
            if args.dataset == 'mbpp':
                # score mbpp predictions
                dfs[model_id] = score_mbpp_predictions(dfs[model_id])
                # calculate pass@k for each hypothesis
                df = dfs[model_id]
                hypothesis_dfs = list(df.groupby(['hypothesis']))
                for h, sample_df in hypothesis_dfs:
                    print(f'Hypothesis: {h}')
                    pass_at_k = calculate_pass_at_k(sample_df.to_dict(orient="records"), k=[1, 10])
                    print(pass_at_k)


            # save back to the same file
            dfs[model_id].to_csv(os.path.join(args.output_dir, args.dataset, f"{model_id}_predictions.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
