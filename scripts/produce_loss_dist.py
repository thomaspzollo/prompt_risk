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
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from detoxify import Detoxify

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prompt_risk.utils import *
from prompt_risk.bounds import *
from prompt_risk.instructions import instruction_sets


def get_instructions(args, instruction_sets):

    if "sum" in args.dataset:
        return instruction_sets["summarization"]
    elif "chat" in args.dataset:
        return instruction_sets["chat"]


def set_seeds(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def get_chat_model(args):

    if args.dataset in ["med_sum", "chat_doctor_sum"]:
        model_name = "stanford-crfm/BioMedLM"
        # model_name = "microsoft/biogpt"
        chat_model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # chat_tokenizer.pad_token = chat_tokenizer.eos_token
    else:
        model_name = "google/flan-t5-{}".format(args.model_size)
        chat_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    chat_model.eval()
    print("chat model loaded")
    return chat_model, chat_tokenizer


def get_scorer(args):
    if args.loss_fn == "bleu":
        scorer = evaluate.load("bleu")
    elif args.loss_fn == "bertscore":
        scorer = evaluate.load("bertscore")
    elif args.loss_fn == "toxicity":
        scorer = Detoxify(device=args.device)
    else:
        raise ValueError
    return scorer


def get_instruction_root(args, instruction):

    if "sum" in args.dataset:
        ins_root = instruction + " Summarize the following document: \n(Document): "
    elif "chat" in args.dataset:
        ins_root = instruction + " Here is the question: \n(Question): "
    else:
        raise ValueError
    return ins_root


def get_data(args, ins_root):

    if args.dataset == "xsum":
        dataset = load_dataset("xsum")["test"]
        def prepend(batch):
            batch["text"] = [ins_root + t for t in batch["document"]]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")

    elif args.dataset == "med_sum":
        dataset = load_dataset("ccdv/pubmed-summarization")["test"]

        def prepend(batch):
            batch["text"] = [ins_root + t for t in batch["article"]]
            batch["summary"] = batch["abstract"]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")

    elif args.dataset == "chat_doctor_sum":
        dataset = load_dataset("LinhDuong/chatdoctor-200k")["train"]

        def prepend(batch):
            print(batch)
            batch["text"] = [(t["instruction"] + " " + t["input"]) for t in batch]
            batch["summary"] = batch["abstract"]
            return batch

        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")
        
    elif args.dataset == "full_chat":
        dataset = load_dataset("Anthropic/hh-rlhf")["test"]
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] for t in batch["rejected"]]
            return batch
    
        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]

    elif args.dataset == "red_team_chat":

        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] for t in batch["transcript"]]
            return batch
            
        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["train"]["text"]
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return dataset, dataloader


def get_scores(args, scorer, chat_out, batch):

    if args.loss_fn == "toxicity":
        tox_out = scorer.predict(chat_out)
        scores = torch.hstack([torch.Tensor(v).unsqueeze(1) for v in tox_out.values()])
        scores = torch.max(scores,-1)[0].detach().cpu().tolist()
        
    elif args.loss_fn == "bertscore":
        scores = scorer.compute(
            predictions=chat_out, 
            references=batch["summary"], 
            lang="en"
        )["f1"]
        scores = list(1-np.array(scores))
        scores = list(scores)
        
    else:
        raise NotImplementedError
        
    return scores


def scoring_pipeline(
    chat_model, 
    chat_tokenizer, 
    instructions, 
    args
):

    scorer = get_scorer(args)
    res = []

    for instruction in instructions:

        set_seeds(args.random_seed)
        ins_root = get_instruction_root(args, instruction)
        dataset, dataloader = get_data(args, ins_root)

        X = []
        with torch.no_grad():
    
            for batch_idx, batch in enumerate(dataloader):

                if "sum" in args.dataset:
                    text = batch["text"]
                elif "chat" in args.dataset:
                    text = batch

                if batch_idx == 0:
                    print(text[0])

                inputs = chat_tokenizer(
                    text, 
                    padding=True, #(args.dataset != "med_sum"), 
                    truncation=True, 
                    return_tensors="pt"
                ).to(args.device)
                
                outputs = chat_model.generate(
                    **inputs, 
                    max_length=args.max_gen_len
                )
                
                chat_out = chat_tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )

                scores = get_scores(args, scorer, chat_out, batch)
                X.extend(scores)

                if len(X) > args.n_total:
                    break
                    
        X = np.array(X)
    
        print("X", X.shape)
        print(X)
        print()
        res.append([instruction, X])

    return res
  

def main(args):
    
    print(args, "\n")
    instructions = get_instructions(args, instruction_sets)[:args.num_hypotheses]
    assert len(instructions) == args.num_hypotheses

    set_seeds(args.random_seed)

    chat_model, chat_tokenizer = get_chat_model(args)

    res = scoring_pipeline(
        chat_model, 
        chat_tokenizer, 
        instructions, 
        args
    )

    save_folder = "../output/{}".format(
        args.dataset, 
    )
    os.makedirs(save_folder, exist_ok=True)
    save_root = "{}/{}_model_{}_{}_loss_dist.pkl".format(
        save_folder,
        args.dataset, 
        args.model_size, 
        args.loss_fn
    )
    print("saving to", save_root)
    
    with open(save_root, 'wb') as file:
        pkl.dump(res, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Produce loss distribution")

    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="random seed"
    )
    parser.add_argument(
        "--max_gen_len", 
        type=int, 
        default=50, 
        help="max length of LLM generations"
    )
    parser.add_argument(
        "--n_total",
        type=int,
        default=2000,
        help="number of evaluated datapoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dataset",
        default="xsum",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--model_size",
        default="base",
        help="FLAN T5 model size"
    )
    parser.add_argument(
        "--loss_fn", 
        default="bertscore", 
        type=str, 
        help="Outer loss function"
    )
    parser.add_argument(
        "--num_hypotheses", 
        type=int, 
        default=5, 
        help="no. of hypotheses"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="gpu device"
    )

    args = parser.parse_args()
    main(args)