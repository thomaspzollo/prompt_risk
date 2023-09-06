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

from text_generation import Client


def get_instructions(args, instruction_sets):

    if args.dataset in ["xsum"]:
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

    if args.dataset in ["xsum"]:
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
        dataset = dataset.with_format("torch")
        
    elif args.dataset == "full_chat":
        dataset = load_dataset("Anthropic/hh-rlhf")["test"]
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] + " \nChatbot Response: " for t in batch["rejected"]]
            return batch
    
        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["text"]

    elif args.dataset == "red_team_chat":

        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        def prepend(batch):
            batch["text"] = [ins_root + t.split("Human: ")[1].split("\n")[0] + " \nChatbot Response: " for t in batch["transcript"]]
            return batch
            
        dataset = dataset.map(prepend, batched=True)
        dataset = dataset.with_format("torch")["train"]["text"]

    elif args.dataset == "pubmed_qa":
        dataset = load_dataset("pubmed_qa", "pqa_artificial")["train"]
        def prepend(batch):
            data = dict()
            data["text"] = [ins_root + batch["question"][t] + " \nAnswer 'no' or 'yes': " + "$" + batch["final_decision"][t] for t in range(len(batch["question"]))]
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
        dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return dataset, dataloader


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
            lang="en"
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

        if args.with_text:
            query_texts = []
            chat_responses = []

        client = Client("http://0.0.0.0:8080")
        
        with torch.no_grad():
    
            for batch_idx, batch in enumerate(tqdm(dataloader, total=int(np.ceil(args.n_total/args.batch_size)))):

                if ("sum" in args.dataset) or (args.dataset == "healthcare"):
                    text = batch["text"]
                elif ("pubmed" in args.dataset):
                    text = [t.split("$")[0] for t in batch]
                elif "chat" in args.dataset:
                    text = batch

                # print(text)

                if batch_idx == 0:
                    print(text[0])
                    print()

                if args.model_size != "xl":

                    inputs = chat_tokenizer(
                        text, 
                        padding=True, #(args.dataset != "med_sum"), 
                        truncation=True, 
                        return_tensors="pt",
                        # device_map="auto"
                    ).to("cuda")
                    
                    outputs = chat_model.generate(
                        **inputs, 
                        max_length=args.max_gen_len
                    )
    
                    chat_out = chat_tokenizer.batch_decode(
                        outputs, 
                        skip_special_tokens=True
                    )

                else:

                    chat_out = [client.generate(t, max_new_tokens=args.max_gen_len).generated_text for t in text]
                    # chat_out = client.generate(text, max_new_tokens=50).generated_text

                scores = get_scores(args, scorer, chat_out, batch)
                X.extend(scores)

                if args.with_text:
                    query_texts.extend(text)
                    chat_responses.extend(chat_out)

                if len(X) > args.n_total:
                    break
                    
        X = np.array(X)
    
        print("X", X.shape)
        print(X)
        print()

        r = [instruction, X]
        if args.with_text:
            r.extend([query_texts, chat_responses])
            assert X.shape[0] == len(query_texts) and len(query_texts) == len(chat_responses)
        
        res.append(r)

        if args.save_results and len(res) > args.save_after:

            print("saving intermediate result", len(res), "to", args.save_root)
            with open(args.save_root, 'wb') as file:
                pkl.dump(res, file)

    return res
  

def main(args):

    args.save_results = not args.no_save
    
    print(args, "\n")
    instructions = get_instructions(args, instruction_sets)[:args.num_hypotheses]
    assert len(instructions) == args.num_hypotheses

    set_seeds(args.random_seed)

    if args.model_size != "xl":
        chat_model, chat_tokenizer = get_chat_model(args)
    else:
        chat_model = None
        model_name = "google/flan-t5-{}".format(args.model_size)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)

    save_folder = "../output/{}".format(
        args.dataset, 
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.with_text:
        save_root = "{}/{}_model_{}_{}_loss_dist_with_text.pkl".format(
            save_folder,
            args.dataset, 
            args.model_size, 
            args.loss_fn
        )    
    else:
        save_root = "{}/{}_model_{}_{}_loss_dist.pkl".format(
            save_folder,
            args.dataset, 
            args.model_size, 
            args.loss_fn
        )
    args.save_root = save_root

    res = scoring_pipeline(
        chat_model, 
        chat_tokenizer, 
        instructions, 
        args
    )

    if args.save_results:
        print("saving final result to", args.save_root)
        with open(args.save_root, 'wb') as file:
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
        default=50, 
        help="no. of hypotheses"
    )
    parser.add_argument(
        "--save_after", 
        type=int, 
        default=0, 
        help="when to start saving intermediate results (will overwrite previous final results)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="gpu device"
    )
    parser.add_argument(
        "--with_text",
        action="store_true"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)