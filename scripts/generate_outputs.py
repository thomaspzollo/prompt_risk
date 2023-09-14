"""This module produces loss scores for the specified dataset and loss function.

Note that if you're using the text-generation-inference server, you should reserve
a GPU for the evaluation models (e.g. BERTScore, Detoxify, etc.) to avoid memory issues.

Examples:
    $ python -m scripts.generate_outputs \
        --datasets red_team_chat full_chat \
        --use-tgi \
        --model-name-or-path google/flan-t5-xxl \
        --num-gpus 4 \
        --server-port 8081 \
        --dtype float16 \
        --print-container-logs \
        --n-total 2000 \
        --num-hypotheses 50
"""
import argparse
from argparse import Namespace
import random
import os
from tqdm import tqdm
import scipy
import numpy as np
import pickle as pkl

import evaluate
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import pipeline, AutoModel, GPTJModel, GPTJConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from tgi.call_server import get_batch_size, make_predictions

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prompt_risk.utils import *
from prompt_risk.bounds import *
from prompt_risk.instructions import instruction_sets

from text_generation import Client
from tgi.docker_utils import server_live, set_hf_token, start_server, stop_server
from tgi.args import parse_args as tgi_parse_args
from scripts.args import parse_args as scripts_parse_args


def get_instructions(args, instruction_sets):
    if args.dataset in ["xsum", "sumedh/MeQSum"]:
        return instruction_sets["summarization"]
    elif args.dataset in ["red_team_chat", "full_chat"]:
        return instruction_sets["chat"]
    elif args.dataset in ["pubmed_qa"]:
        return instruction_sets["pubmed"]
    elif args.dataset in ["healthcare"]:
        return instruction_sets["healthcare"]


def set_seeds(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def get_chat_model(args):
    if args.dataset in ["med_sum", "chat_doctor_sum"]:
        model_name = "GanjinZero/biobart-large"
        # model_name = "microsoft/biogpt"
        chat_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            # device_map="auto",
            # torch_dtype=torch.float16
        ).to(args.device)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # chat_tokenizer.pad_token = chat_tokenizer.eos_token
    else:
        model_name = "google/flan-t5-{}".format(args.model_size)
        chat_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16
        ).to(args.device)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)

    chat_model.eval()
    print("chat model loaded")
    return chat_model, chat_tokenizer


def get_instruction_root(args, instruction):
    if args.dataset in ["xsum", "sumedh/MeQSum"]:
        ins_root = "Your goal is to summarize a document. " + instruction + " Summarize the following document: \nDocument: "
    elif args.dataset in ["red_team_chat", "full_chat"]:
        ins_root = instruction + "\nHere is a human input: "
    elif args.dataset in ["pubmed_qa", "healthcare"]:
        ins_root = "You are a helpful medical chatbot. " + instruction + "\n\nHere is a medical query: "
    else:
        raise ValueError
    return ins_root


def get_data(args, ins_root):
    if args.dataset == "xsum":
        dataset = load_dataset("xsum")["test"]
        def prepend(batch):
            batch["text"] = [ins_root + t + " \nSummary: " for t in batch["document"]]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]

    elif args.dataset == "full_chat":
        dataset = load_dataset("Anthropic/hh-rlhf")["test"]
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] + " \nChatbot Response:" for t in batch["rejected"]]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]

    elif args.dataset == "red_team_chat":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] + " \nChatbot Response:" for t in batch["transcript"]]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["train"]["text"]
        
    elif args.dataset == "pubmed_qa":
        dataset = load_dataset("pubmed_qa", "pqa_artificial")["train"]
        def prepend(batch):
            data = dict()
            data["text"] = [ins_root + batch["question"][t] + " \nAnswer 'no' or 'yes': " + "$" + batch["final_decision"][t] for t in range(len(batch["question"]))]
            data["text"] = [t.split("$")[0] for t in data["text"]]
            return data

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")
        dataset = dataset["text"]

    elif args.dataset == "healthcare":
        dataset = load_dataset("wangrongsheng/HealthCareMagic-100k-en")["train"]
        def prepend(batch):
            data = dict()
            data["text"] = [
                (ins_root + b + "\nProvide a detailed response in one paragraph: ")
                for b in batch["input"]
            ]
            data["output"] = batch["output"]
            return data

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]
    
    elif args.dataset == "sumedh/MeQSum":
        # TODO: parse this with an excel parser
        dataset = load_dataset("sumedh/MeQSum")["train"]
        def prepend(batch):
            data = dict()
            data["text"] = [
                (ins_root + b + "\nProvide a detailed response in one paragraph: ")
                for b in batch["input"]
            ]
            data["output"] = batch["output"]
            return data
        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    return dataset, dataloader


def get_datasets_dataloaders(instructions, cache_df, args, return_dataloaders=False):
    datasets = []
    dataloaders = []
    for instruction in tqdm(instructions, desc="instructions"):
        set_seeds(args.random_seed)
        ins_root = get_instruction_root(args, instruction)
        # dataset is a list of text examples, dataloader is a DataLoader
        dataset, dataloader = get_data(args, ins_root)
        # check if any of the text examples are duplicated
        # if so, warn user that they will be removed
        unique = set(dataset)
        n_unique = len(unique)
        n_total = len(dataset)
        if n_unique != n_total:
            print(f"Warning: {n_total - n_unique}/{n_total} examples are duplicates and will be removed.")
            dataset = sorted(list(unique))
        
        # clip to requested number of examples
        dataset_slice = slice(0, args.n_total)
        dataset = dataset[dataset_slice]
        
        # check if any of the text examples + instruction have already been processed
        # if so, remove them from the list
        if not cache_df.empty:
            text_len = len(dataset)
            temp_df = cache_df[(cache_df["hypothesis"] == instruction) & (cache_df["dataset"] == args.dataset) & (cache_df["model_name_or_path"] == args.model_name_or_path)]
            final_dataset = []
            already_ran_texts = set(temp_df["text"].tolist())
            for text in dataset:
                if text not in already_ran_texts:
                    final_dataset.append(text)
            dataset = final_dataset
            print(f"Skipping {text_len - len(dataset)}/{text_len} examples that were already run.")
        
        # retain metadata and clip to requested number of examples
        dataset_dict = {"dataset": args.dataset, "hypothesis": instruction, "model_name_or_path": args.model_name_or_path, "text": dataset}
        datasets.append(dataset_dict)
        if return_dataloaders:
            dataloaders.append(dataloader)
    return datasets, dataloaders

def tgi_prediction_pipeline(dataset, cache_df, args):
    # process text examples with the TGI server
    # returns [id, text, generated_text, ...]
    df = make_predictions(dataset['text'],
                            args,
                            num_threads=args.max_batch_size)
    df['hypothesis'] = dataset['hypothesis']
    df['dataset'] = dataset['dataset']
    df['model_name_or_path'] = dataset['model_name_or_path']
    cache_df = pd.concat([cache_df, df], ignore_index=True)
    return cache_df


def main(args):
    # TODO: extract this into a generate_outputs function that can be called from other scripts
    print(args, "\n")
    set_seeds(args.random_seed)

    # set HF token to ensure access to llama 2 models
    set_hf_token()

    # allow user to specify a collection of datasets and then loop over them
    if not args.datasets:
        args.datasets = [args.dataset]

    # TODO: add support for multiple models
    container = None
    for dataset_ in args.datasets:
        args.dataset = dataset_
        # define output folder and cache csv path
        save_folder = os.path.join(args.output_dir, args.dataset)
        os.makedirs(save_folder, exist_ok=True)
        model_name = args.model_name_or_path.replace("/", "-")
        csv_path = os.path.join(save_folder, f'{model_name}_predictions.csv')

        # load the cache
        if os.path.exists(csv_path) and not args.no_cache:
            # load the existing csv
            cache_df = pd.read_csv(csv_path)
        else:
            cache_df = pd.DataFrame()
        
        # prepare the datasets
        instructions = get_instructions(args, instruction_sets)[:args.num_hypotheses]
        assert len(instructions) == args.num_hypotheses
        # dataloaders might be useful if we need to run native PyTorch models
        datasets, dataloaders = get_datasets_dataloaders(instructions, cache_df, args)
        
        # if all datasets are empty (i.e. all text examples have already been processed), exit
        if all([len(d["text"]) == 0 for d in datasets]):
            print(f"All {args.n_total} requested text examples have already been processed for dataset {args.dataset}")
            continue

        # tokenize the dataset and check the token distribution
        # so we can set the max input length and max total tokens appropriately
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # only retain datasets that have at least one text example
        datasets = [d for d in datasets if len(d['text']) > 0]
        # get max length of the dataset among all text examples
        max_len = 0
        for idx, d in enumerate(datasets):
            sample_dataset = d['text']
            tokenized = tokenizer(sample_dataset, padding=False, truncation=False,)
            lens = [len(t) for t in tokenized['input_ids']]
            desc_df = pd.DataFrame({'len': lens})
            stats = desc_df.describe()
            max_len = max(max_len, int(stats.loc['max', 'len']))
            if idx == 0:
                print(f'Token distribution for dataset {args.dataset}:')
                print(stats)
        print(f"Max length across all prompts in datasets: {max_len}")
        # set max_input_tokens to max length + 100 to account for special tokens and any variation in prompts
        args.max_input_length = max_len + 100
        # set max_total_tokens min of 3x max_input_tokens and 4096
        args.max_total_tokens = min(args.max_input_length * 3, 4096)
        args.max_new_tokens = args.max_total_tokens - args.max_input_length
        print(f"Max input length: {args.max_input_length}")
        print(f"Max total tokens: {args.max_total_tokens}")

        if args.use_tgi:
            args.max_concurrent_requests = 128 # default
            if 'flan-t5' in args.model_name_or_path:
                # set max batch total tokens manually because TGI won't do it automatically for T5 models
                # this was set somewhat experimentally
                args.max_concurrent_requests = 200
                args.max_batch_total_tokens = min(args.max_total_tokens*args.max_concurrent_requests, 50_000)
            # start the text-generation-inference server with the specified model
            container = start_server(args)
            # get the max batch size
            args.max_batch_size = get_batch_size(container, args.max_total_tokens, args.max_concurrent_requests)

        # run predictions for all datasets
        for dataset in datasets:
            cache_df = tgi_prediction_pipeline(dataset, cache_df, args)
            # save csv the predictions
            cache_df.to_csv(csv_path, index=False)

        # stop the server so we can reconfigure it for the next dataset
        if args.use_tgi:
            # stop the server
            stop_server(container)

if __name__ == "__main__":
    # merge the Docker/TGI args with the main args
    args, unknown = scripts_parse_args(known_only=True)
    docker_args, remaining_unknown = tgi_parse_args(unknown, known_only=True)
    # check if there are any remaining unknown args
    if remaining_unknown:
        raise argparse.ArgumentError(None, f'Unknown arguments: {remaining_unknown}')

    # merge the two args objects
    args = vars(args)
    args.update(vars(docker_args))
    args = Namespace(**args)
    main(args)