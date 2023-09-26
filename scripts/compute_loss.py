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
    
    $ python -m scripts.compute_loss \
        --output-dir output \
        --datasets bigbio/meqsum \
        --loss-fn rouge

    $ python -m scripts.compute_loss \
        --output-dir output \
        --datasets cnn_dailymail \
        --loss-fn rouge
    
    $ python -m scripts.compute_loss \
        --output-dir output \
        --datasets cnn_dailymail \
        --loss-fn bertscore \
        --batch-size 400
"""
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
from prompt_risk.instructions import MEQSUM_PROMPT_FILES

from scripts.generate_outputs import extract_user_portion, prepare_test_code, subset_ensure_unique
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
    
    # load original dataset to get human input
    if args.dataset == 'full_chat':
        dataset = load_dataset("Anthropic/hh-rlhf")["test"]
        # create an id column ranging from 0 to len(dataset)
        ids = list(range(len(dataset)))
        dataset = dataset.add_column('task_id', ids)
        dataset = subset_ensure_unique(dataset, n_total=None, text_col_name='rejected')
        # supplement with train dataset
        train_dataset = load_dataset("Anthropic/hh-rlhf")["train"]
        # create an id column ranging from len(dataset) to len(dataset) + len(train_dataset)
        ids = list(range(len(dataset), len(dataset) + len(train_dataset)))
        train_dataset = train_dataset.add_column('task_id', ids)
        train_dataset = subset_ensure_unique(train_dataset, n_total=None, text_col_name='rejected')
        dataset = concatenate_datasets([dataset, train_dataset])
        dataset = dataset.map(lambda x: {'human_input': extract_user_portion(x['rejected'])}, batched=False)
        label_df = dataset.to_pandas()[['task_id', 'human_input']]
    elif args.dataset == 'red_team_chat':
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
        # create an id column ranging from 0 to len(dataset)
        ids = list(range(len(dataset)))
        dataset = dataset.add_column('task_id', ids)
        dataset = subset_ensure_unique(dataset, None, text_col_name='transcript')
        dataset = dataset.map(lambda x: {'human_input': extract_user_portion(x['transcript'])}, batched=False)
        label_df = dataset.to_pandas()[['task_id', 'human_input']]
    # join on task_id
    original_len = len(df)
    df = df.merge(label_df, on="task_id", how="left")
    assert len(df) == original_len, f"Expected {original_len} rows, got {len(df)} rows"
    assert df['human_input'].isna().sum() == 0, "Some rows are missing human input"

    # generated_text is model output
    # prepend 'human_input' column with '###Human: ', prepend 'generated_text' column with '###Assistant: ', then concatenate
    df['weqweasdas/hh_rlhf_rm_open_llama_3b_eval_text'] = '###Human: ' + df['human_input'].astype(str).str.strip() + ' ###Assistant: ' + df['generated_text'].astype(str).str.strip()

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
    df = df.drop(columns=['human_input'], errors='ignore')
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

def extract_summary(text):
    parts = text.split('[SUMMARY]')
    if len(parts) == 1:
        return parts[0]
    parts = parts[1].split('[/SUMMARY]')
    if len(parts) == 1:
        return parts[0]
    return parts[0].strip()

def score_meqsum_predictions(df, scorer):
    # load reference summaries
    dataset = load_dataset("bigbio/meqsum", "meqsum_source")['train']
    dataset = dataset.filter(lambda x: x['File'] not in MEQSUM_PROMPT_FILES)
    # rename 'File' column to 'task_id' to match df
    label_df = dataset.to_pandas().rename(columns={'File': 'task_id'})
    # merge with predictions on task_id
    df = df.merge(label_df, on="task_id", how="left")
    # extract prediction from 'generated_text' column
    df['generated_summary'] = df['generated_text'].apply(lambda x: extract_summary(x))
    # compute rouge scores
    rouge_scores = scorer.compute(predictions=df['generated_summary'].tolist(), references=df['Summary'].tolist(), use_aggregator=False)
    # extract rougeL scores
    rougeL_scores = rouge_scores['rougeL']
    # add rougeL scores to df
    df['rougeL'] = rougeL_scores
    # drop unnecessary columns from meqsum 'CHQ', 'Summary'
    df = df.drop(columns=['CHQ', 'Summary'], errors='ignore')
    return df

def score_summarization_predictions(df, scorer, dataset='cnn_dailymail'):
    # if we've already computed scores, skip
    if scorer.name == 'rouge' and 'rougeL' in df.columns and (df['rougeL'].isna().sum() == 0):
        print(f"rougeL already computed for {df.shape[0]} rows in {dataset}. Skipping.")
        return df
    if scorer.name == 'bert_score' and 'bertscore_f1' in df.columns and (df['bertscore_f1'].isna().sum() == 0):
        print(f"bertscore already computed for {df.shape[0]} rows in {dataset}. Skipping.")
        return df
    
    # load reference summaries
    if dataset == 'cnn_dailymail':
        dataset = load_dataset('cnn_dailymail', name='3.0.0', split='train')
        dataset = subset_ensure_unique(dataset, n_total=None, text_col_name='article')
        dataset = dataset.rename_column('id', 'task_id')
        dataset = dataset.rename_column('highlights', 'summary')
        label_df = dataset.to_pandas()[['task_id', 'summary']]
    elif dataset == 'xsum':
        dataset = load_dataset("xsum", split="train")
        dataset = subset_ensure_unique(dataset, n_total=None, text_col_name='document')
        dataset = dataset.rename_column('id', 'task_id')
        label_df = dataset.to_pandas()[['task_id', 'summary']]
        label_df['task_id'] = label_df['task_id'].astype(int)
    # merge with predictions on task_id
    original_len = len(df)
    df = df.merge(label_df, on="task_id", how="left")
    assert len(df) == original_len, f"Expected {original_len} rows, got {len(df)} rows"
    assert df['summary'].isna().sum() == 0, "Some rows are missing human input"
    # compute rouge scores
    if scorer.name == 'rouge':
        rouge_scores = scorer.compute(predictions=df['generated_text'].tolist(), references=df['summary'].tolist(), use_aggregator=False)
        # extract rougeL scores
        rougeL_scores = rouge_scores['rougeL']
        # add rougeL scores to df
        df['rougeL'] = rougeL_scores
    # compute bertscore
    if scorer.name == 'bert_score':
        start = time.time()
        bert_scores = scorer.compute(predictions=df['generated_text'].tolist(), references=df['summary'].tolist(), device=args.device, lang='en', batch_size=args.batch_size)
        end = time.time()
        print(f"Time to compute bertscore for {len(df):,} rows: {end-start:.2f} seconds. Average time per example: {(end-start)/len(df):.2f} seconds.")
        # extract bertscore scores
        bert_scores_df = pd.DataFrame(bert_scores).rename(columns={'f1': 'bertscore_f1', 'precision': 'bertscore_precision', 'recall': 'bertscore_recall', 'hashcode': 'bertscore_hashcode'})
        # add bertscore scores to df
        df = pd.concat([df, bert_scores_df], axis=1)

    # drop 'summary' column
    df = df.drop(columns=['summary'], errors='ignore')

    # score with 
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
        dataset_dir = args.dataset.replace("/","_") # handle datasets like bigbio/meqsum
        # load generated outputs for specified models
        models = [x.replace("/","-") for x in args.eval_models]
        # if no models specified, load all models
        if not models:
            files = os.listdir(os.path.join(args.output_dir, dataset_dir))
            files = [f for f in files if f.endswith(".csv")]
        else:
            files = [f"{m}_predictions.csv" for m in models]
        # filter out any embeddings files
        files = [f for f in files if not f.endswith("embeddings.csv")]
        dfs = {}
        for f in files:
            model_id = f.split("_predictions.csv")[0]
            df = pd.read_csv(os.path.join(args.output_dir, dataset_dir, f))
            dfs[model_id] = df

            # check if all models have the expected number of rows populated with text
            if dfs[model_id].shape[0] != 50*2000:
                print(f"Warning: {model_id} has {dfs[model_id].shape[0]} rows, expected 50*2000=100,000")
            # it's possible that the model simply predicts EOS, which means generated text will be nan
            valid_empty_rows = (dfs[model_id]['finish_reason'] == 'eos_token') & dfs[model_id]['generated_text'].isna()
            dfs[model_id].loc[valid_empty_rows, 'generated_text'] = ''
            # check for any nans in generated_text
            assert dfs[model_id].generated_text.isna().sum() == 0

            # compute loss for each model
            print(f"Computing loss for {model_id}")
            if 'chat' in args.dataset:
                dfs[model_id], scorer = compute_chat_loss(df, scorer, args)
            elif args.dataset == 'mbpp':
                # score mbpp predictions
                dfs[model_id] = score_mbpp_predictions(dfs[model_id])
                # calculate pass@k for each hypothesis
                df = dfs[model_id]
                hypothesis_dfs = list(df.groupby(['hypothesis']))
                for h, sample_df in hypothesis_dfs:
                    if sample_df['passed'].isna().any():
                        continue
                    print(f'Hypothesis: {h}')
                    pass_at_k = calculate_pass_at_k(sample_df.to_dict(orient="records"), k=[1, 10])
                    print(pass_at_k)
            elif args.dataset == 'bigbio/meqsum':
                # score meqsum predictions
                scorer = get_scorer(args)
                dfs[model_id] = score_meqsum_predictions(dfs[model_id], scorer)
                scorer = None # in case we need to load a different scorer for the next dataset
                print(f'Average rougeL score on {args.dataset} - {model_id}: {dfs[model_id]["rougeL"].mean():.4f}')
            elif args.dataset in ['cnn_dailymail', 'xsum']:
                # score summarization predictions
                scorer = get_scorer(args)
                dfs[model_id] = score_summarization_predictions(dfs[model_id], scorer, dataset=args.dataset)
                scorer = None
                if 'rougeL' in dfs[model_id].columns:
                    print(f'Average rougeL score on {args.dataset} - {model_id}: {dfs[model_id]["rougeL"].mean():.4f}')
                if 'bertscore_f1' in dfs[model_id].columns:
                    print(f'Average bertscore_f1 score on {args.dataset} - {model_id}: {dfs[model_id]["bertscore_f1"].mean():.4f}')
            
            # save back to the same file
            dfs[model_id].to_csv(os.path.join(args.output_dir, dataset_dir, f"{model_id}_predictions.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
