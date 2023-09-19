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
        --num-hypotheses 50 \
        --seed 42
    
    $ python -m scripts.generate_outputs \
        --datasets mbpp \
        --use-tgi \
        --model-name-or-path codellama/CodeLlama-7b-Instruct-hf \
        --num-gpus 1 \
        --server-port 8081 \
        --dtype float16 \
        --print-container-logs \
        --n-total 103 \
        --num-hypotheses 2 \
        --num-return-sequences 10 \
        --seed 42 \
        --do-sample
    
    $ python -m scripts.generate_outputs \
        --datasets bigbio/meqsum \
        --use-tgi \
        --model-name-or-path tiiuae/falcon-7b-instruct \
        --num-gpus 1 \
        --server-port 8081 \
        --dtype float16 \
        --print-container-logs \
        --n-total 100 \
        --num-hypotheses 21 \
        --seed 42
"""
import argparse
import os
import random
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM,
                        AutoTokenizer)

from prompt_risk.instructions import instruction_sets, MEQSUM_PROMPT_FILES
from scripts.llama_utils import prepare_chats
from scripts.args import parse_args as scripts_parse_args
from tgi.args import parse_args as tgi_parse_args
from tgi.call_server import get_batch_size, make_predictions
from tgi.docker_utils import set_hf_token, start_server, stop_server

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_k_shot_examples(k=3, n=50, idx_start=1, idx_end=11):
    """Selects between 1 and k examples without replacement from Task IDs [idx_start, idx_end).
    Returns n examples total. Note that we check that all examples are unique.
    """
    # NB: ensure random seeds are set before calling this function
    assert k >= 1 and k <= idx_end, f"k must be between 1 and {idx_end}"
    chosen_idxs = set()
    # crucially we need to preserve the order of the idxs to be able to build incrementally
    # i.e. num_hypotheses 2->20
    ordered_chosen_idxs = []
    for i in range(n):
        # ensure every example is unique
        while True:
            num_shots = random.randint(1, k)
            idxs = random.sample(range(idx_start, idx_end), k=num_shots)
            # note that we're preserving the order of the idxs (i.e. 1, 2 != 2, 1)
            idxs = tuple(idxs) # tuples are hashable
            if idxs not in chosen_idxs:
                chosen_idxs.add(idxs)
                ordered_chosen_idxs.append(idxs)
                break
    chosen_idxs = [list(x) for x in ordered_chosen_idxs]
    return chosen_idxs

def create_hypotheses(args, instruction_set, k_shot_idx_start=0, k_shot_idx_end=10):
    """Creates hypotheses for the specified instruction set, including choosing k-shot examples."""
    # select the k-shot examples
    variants_needed = max(0, args.num_hypotheses - len(instruction_sets[instruction_set]))
    k_shot_idxs = select_k_shot_examples(k=args.k_shots, n=variants_needed, idx_start=k_shot_idx_start, idx_end=k_shot_idx_end)
    instructions = []
    for i in range(args.num_hypotheses):
        # first add the instructions without the k-shot examples
        if i < len(instruction_sets[instruction_set]):
            instructions.append({'instruction': instruction_sets[instruction_set][i]})
        else:
            # then start adding the k-shot examples, while cycling through the list of prompts
            instruction_idx = i % len(instruction_sets[instruction_set])
            idx_into_kshots = i - len(instruction_sets[instruction_set])
            instructions.append({'instruction': instruction_sets[instruction_set][instruction_idx], 'k_shot_idxs': k_shot_idxs[idx_into_kshots]})
    return instructions

def get_instructions(args, instruction_sets):
    if args.dataset in ["xsum"]:
        return instruction_sets["summarization"]
    elif args.dataset in ['bigbio/meqsum']:
        instruction_set = "healthcare_question_summarization"
        instructions = create_hypotheses(args, instruction_set, k_shot_idx_start=0, k_shot_idx_end=len(MEQSUM_PROMPT_FILES))
        return instructions
    elif args.dataset in ["red_team_chat", "full_chat"]:
        return instruction_sets["chat"]
    elif args.dataset in ["pubmed_qa"]:
        return instruction_sets["pubmed"]
    elif args.dataset in ["healthcare"]:
        return instruction_sets["healthcare"]
    elif args.dataset in ["mbpp", "human_eval"]:
        # TODO: return a list of dicts containing prompts and chosen k-shot examples
        # will need to load mbpp dataset by this point
        if args.dataset == "mbpp":
            instruction_set = "code"
            # task_IDs range from 1-10
            instructions = create_hypotheses(args, instruction_set, k_shot_idx_start=1, k_shot_idx_end=11)
            return instructions

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
    elif args.dataset in ["mbpp", "human_eval", "bigbio/meqsum"]:
        # simply return the instruction because prompt preparation is handled elsewhere
        ins_root = instruction
    else:
        raise ValueError
    return ins_root

def prepare_test_code(example):
    setup_code = example['test_setup_code']
    tests = example['test_list']
    test_code = setup_code + '\n' if setup_code else ''
    test_code += '\n'.join(tests)
    return test_code

def prepare_mbpp_user_prompt(mbpp_example, prompt_dataset, k_shot_idxs):
    """Prepares user portion of the prompt."""
    # borrowed from https://arxiv.org/pdf/2308.12950.pdf
    template = '{}\nYour code should pass these tests:\n\n{}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.'
    prompt = []
    # add the k-shot examples
    for idx in k_shot_idxs:
        # row idx is task_id - 1
        idx = idx - 1
        example = prompt_dataset[idx]
        task = example['text']
        # task_id 927 for an example that uses a test setup
        test_text = prepare_test_code(example)
        prompt.append(template.format(task, test_text))
        # show correct answer for this example
        correct_answer = f'[PYTHON]\n{example["code"].strip()}\n[/PYTHON]'
        prompt.append(correct_answer)
    
    # add task to be completed
    task = mbpp_example['text']
    test_text = prepare_test_code(mbpp_example)
    prompt.append(template.format(task, test_text))
    prompt = '\n'.join(prompt)
    return {'prompt': prompt}

def prepare_meqsum_user_prompt(task_example, prompt_dataset, k_shot_idxs):
    """Prepares user portion of the prompt, potentially including k-shot examples.
    The system prompt will be prepended to the user prompt at a later stage.
    """
    template = 'Summarize the following user question:\n{}\n\nYour summary should start with a [SUMMARY] tag and end with a [/SUMMARY] tag.'
    prompt = []
    # add the k-shot examples
    for idx in k_shot_idxs:
        example = prompt_dataset[idx]
        task = example['CHQ']
        prompt.append(template.format(task))
        # show correct answer for this example
        correct_answer = f'[SUMMARY]\n{example["Summary"].strip()}\n[/SUMMARY]'
        prompt.append(correct_answer)
    
    # add task to be completed
    task = task_example['CHQ']
    prompt.append(template.format(task))
    prompt = '\n'.join(prompt)
    return {'prompt': prompt}

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
    
    elif args.dataset == "bigbio/meqsum":
        dataset = load_dataset("bigbio/meqsum", "meqsum_source")['train']
        prompt_dataset = dataset.filter(lambda x: x['File'] in MEQSUM_PROMPT_FILES)
        dataset = dataset.filter(lambda x: x['File'] not in MEQSUM_PROMPT_FILES)
        # take instruction and potential k-shots and format them into a formatted dialog
        system_prompt = ins_root['instruction']
        k_shot_idxs = ins_root['k_shot_idxs'] if 'k_shot_idxs' in ins_root else []
        dataset = dataset.map(prepare_meqsum_user_prompt, batched=False, fn_kwargs={'prompt_dataset': prompt_dataset, 'k_shot_idxs': k_shot_idxs})
        # if llama model, format accordingly
        if 'llama' in args.model_name_or_path:
            system_message = {'role': 'system', 'content': system_prompt}
            # prepare the dialogs by combining the system prompt with the user prompt
            dialogs = []
            for example in dataset:
                dialog = [system_message]
                dialog.append({'role': 'user', 'content': example['prompt']})
                dialogs.append(dialog)
            
            dialogs = prepare_chats(dialogs)
        else: # e.g. Falcon model
            dialogs = []
            for example in dataset:
                dialog = system_prompt.strip() + '\n\n' + example['prompt']
                dialogs.append(dialog)

        # retain the task_id (i.e. the File) for each example
        final_dataset = []
        for i, dialog in enumerate(dialogs):
            example = {'text': dialog, 'task_id': dataset[i]['File']}
            final_dataset.append(example)
        dataset = final_dataset

    elif args.dataset == "mbpp":
        dataset = load_dataset("mbpp")
        # task_id 1-10
        prompt_dataset = dataset['prompt']
        # concat train, validation, and test portions
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        if 'codellama' not in args.model_name_or_path:
            raise ValueError("mbpp dataset is only supported for codellama models. If other models are needed, format the prompts accordingly.")
        
        # take instruction and potential k-shots and format them into a formatted dialog for code llama
        system_prompt = ins_root['instruction']
        system_message = {'role': 'system', 'content': system_prompt}
        k_shot_idxs = ins_root['k_shot_idxs'] if 'k_shot_idxs' in ins_root else []
        dataset = dataset.map(prepare_mbpp_user_prompt, batched=False, fn_kwargs={'prompt_dataset': prompt_dataset, 'k_shot_idxs': k_shot_idxs})
        # prepare the dialogs by combining the system prompt with the user prompt
        dialogs = []
        for example in dataset:
            dialog = [system_message]
            dialog.append({'role': 'user', 'content': example['prompt']})
            dialogs.append(dialog)
        
        dialogs = prepare_chats(dialogs)
        # retain the task_id for each example
        final_dataset = []
        for i, dialog in enumerate(dialogs):
            example = {'text': dialog, 'task_id': dataset[i]['task_id']}
            final_dataset.append(example)
        dataset = final_dataset

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
        set_seeds(args.seed)
        ins_root = get_instruction_root(args, instruction)
        # dataset is a list of text examples, dataloader is a DataLoader
        dataset, dataloader = get_data(args, ins_root)
        # check if any of the text examples are duplicated
        # if so, warn user that they will be removed
        # can only check against hashale types (e.g., strings, etc.)
        if isinstance(dataset[0], str):
            unique = set(dataset)
            n_unique = len(unique)
            n_total = len(dataset)
            if n_unique != n_total:
                print(f"Warning: {n_total - n_unique}/{n_total} examples are duplicates and will be removed.")
                dataset = sorted(list(unique))
        
        # clip to requested number of examples
        if len(dataset) < args.n_total:
            print(f"Warning: {args.n_total:,} examples requested but only {len(dataset):,} examples available. Using {len(dataset):,} examples.")
        dataset_slice = slice(0, args.n_total)
        dataset = dataset[dataset_slice]
        
        # blow out dataset according to args.num_return_sequences
        complete_dataset = []
        original_seed = args.seed
        for i in range(args.num_return_sequences):
            seed = original_seed + i
            for example in dataset:
                example_dict = {} 
                if isinstance(example, dict):
                    # retain any existing values, including text, possibly task_id, etc.
                    example_dict.update(example)
                elif isinstance(example, str):
                    example_dict['text'] = example
                example_dict.update({'hypothesis': instruction, 'dataset': args.dataset, 'model_name_or_path': args.model_name_or_path, 'seed': seed})
                complete_dataset.append(example_dict)
        dataset = complete_dataset

        # check if any of the text examples + instruction + seed have already been processed
        # if so, remove them from the to-be-processed list
        if not cache_df.empty:
            num_examples = len(dataset)
            final_dataset = []
            for example in dataset:
                # check all columns except for the generated_text column
                mask = True
                for col in example:
                    mask &= (cache_df[col] == example[col])
                # if no row matches, add the example to the final dataset, which needs to be processed
                if not mask.any():
                    final_dataset.append(example)
            dataset = final_dataset
            print(f"Skipping {num_examples - len(dataset)}/{num_examples} examples that were already run.")
        
        datasets.append(dataset)
        if return_dataloaders:
            dataloaders.append(dataloader)
    return datasets, dataloaders

def tgi_prediction_pipeline(dataset, cache_df, args):
    # process text examples with the TGI server
    # returns [id, text, generated_text, ...]
    # give each example a unique id so we can merge the metadata back in later
    id_dataset = []
    for idx, example in enumerate(dataset):
        example['id'] = idx
        id_dataset.append(example)
    dataset = id_dataset
    df = make_predictions(dataset,
                            args,
                            num_threads=args.max_batch_size)
    # merge the metadata back in, assuming the metadata is the same for all examples
    cols = list(dataset[0].keys())
    # only retain columns not already in the df
    cols =  ['id'] + [col for col in cols if col not in df.columns]
    meta_df = pd.DataFrame(dataset)[cols]
    df = df.merge(meta_df, on='id')
    col_order = ['id', 'text', 'generated_text']
    col_order.extend([col for col in df.columns if col not in col_order])
    cache_df = pd.concat([cache_df, df], ignore_index=True)
    # retain any columns that were already in the cache
    cache_cols = cache_df.columns
    col_order = col_order + [col for col in cache_cols if col not in col_order]
    cache_df = cache_df[col_order]
    return cache_df


def main(args):
    # TODO: extract this into a generate_outputs function that can be called from other scripts
    print(args, "\n")
    set_seeds(args.seed)

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
        dataset_dir = args.dataset.replace("/", "_")
        save_folder = os.path.join(args.output_dir, dataset_dir)
        os.makedirs(save_folder, exist_ok=True)
        model_name = args.model_name_or_path.replace("/", "-")
        csv_path = os.path.join(save_folder, f'{model_name}_predictions.csv')

        # load the cache
        if os.path.exists(csv_path) and not args.no_cache:
            # load the existing csv
            cache_df = pd.read_csv(csv_path)
            # if hypothesis column is a dict string, convert it back to a dict
            # so we can properly check if any of the text examples have already been processed
            cache_df['hypothesis'] = cache_df['hypothesis'].apply(eval)
            if 'seed' not in cache_df.columns:
                # NB: this is a late addition in order to support multiple seeds
                # without having to rerun old experiments that didn't have a seed column
                # add a seed column
                cache_df['seed'] = args.seed
        else:
            cache_df = pd.DataFrame()
        
        # prepare the datasets
        instructions = get_instructions(args, instruction_sets)[:args.num_hypotheses]
        assert len(instructions) == args.num_hypotheses
        # dataloaders might be useful if we need to run native PyTorch models
        # this set will check if any of the text examples have already been
        # processed and only returns data that needs to be processed
        datasets, dataloaders = get_datasets_dataloaders(instructions, cache_df, args)
        
        # if all datasets are empty (i.e. all text examples have already been processed), exit
        if all([len(d) == 0 for d in datasets]):
            print(f"All {args.n_total} requested text examples have already been processed for dataset {args.dataset}")
            continue

        # tokenize the dataset and check the token distribution
        # so we can set the max input length and max total tokens appropriately
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # only retain datasets that have at least one text example
        datasets = [d for d in datasets if len(d) > 0]
        # get max length of the dataset among all text examples
        max_len = 0
        for idx, d in enumerate(datasets):
            sample_dataset = [x['text'] for x in d]
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
        print(f"Max input length: {args.max_input_length:,}")
        print(f"Max total tokens: {args.max_total_tokens:,}")
        
        if args.use_tgi:
            args.max_concurrent_requests = 128 # default
            if 'flan-t5' in args.model_name_or_path:
                # set max batch total tokens manually because TGI won't do it automatically for T5 models
                # this was set somewhat experimentally
                args.max_concurrent_requests = 200
                args.max_batch_total_tokens = min(args.max_total_tokens*args.max_concurrent_requests, 50_000)
            if 'codellama' in args.model_name_or_path:
                # may need to increase max length for particularly long prompts
                # see https://huggingface.co/docs/text-generation-inference/basic_tutorials/preparing_model
                if args.max_input_length + 512 > 4096:
                    new_max = args.max_input_length + 512
                    print(f"Warning: max input length of {args.max_input_length:,} leaves little (or no) room for generation. Increasing max_total_tokens to {new_max:,} with RoPE scaling.")
                    args.max_total_tokens = new_max
                    args.rope_scaling = 'dynamic'
                    args.rope_factor = new_max / 4096
            # start the text-generation-inference server with the specified model
            container = start_server(args)
            # get the max batch size
            args.max_batch_size = get_batch_size(container, args.max_total_tokens, args.max_concurrent_requests)

        # run predictions for all datasets
        for dataset in datasets:
            if 'codellama' in args.model_name_or_path:
                # include [/PYTHON] and </s> in the stop tokens
                args.stop = ['[/PYTHON]', '</s>']
                # following the codellama paper: https://arxiv.org/pdf/2308.12950.pdf
                args.top_p = 0.95
                args.temperature = 0.8
            if args.dataset == 'bigbio/meqsum':
                # include [/SUMMARY] and </s> in the stop tokens
                args.stop = ['[/SUMMARY]', tokenizer.eos_token]
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