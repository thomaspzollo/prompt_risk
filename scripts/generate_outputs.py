"""This module produces loss scores for the specified dataset and loss function.

Note that if you're using the text-generation-inference server, you should reserve
a GPU for the evaluation models (e.g. BERTScore, Detoxify, etc.) to avoid memory issues.

Examples:
    $ python -m scripts.generate_outputs \
        --datasets red_team_chat full_chat \
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
        --model-name-or-path tiiuae/falcon-7b-instruct \
        --num-gpus 1 \
        --server-port 8081 \
        --dtype float16 \
        --print-container-logs \
        --n-total 100 \
        --num-hypotheses 21 \
        --seed 42
    
    $ python -u -m scripts.generate_outputs \
        --datasets cnn_dailymail \
        --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
        --num-gpus 2 \
        --n-total 10000 \
        --batch-size 1000 \
        --seed 42
"""
import argparse
import os
import random
from argparse import Namespace
import time
import hashlib

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, Value
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM,
                        AutoTokenizer)

from prompt_risk.instructions import instruction_sets, MEQSUM_PROMPT_FILES
from .llama_utils import prepare_chats
from .args import parse_args as scripts_parse_args
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
    if args.dataset in ["xsum", "cnn_dailymail"]:
        # convert to dictionaries for consistency with other datasets
        instructions = [{'instruction': ins} for ins in instruction_sets["summarization"]]
        return instructions
    elif args.dataset in ['bigbio/meqsum']:
        instruction_set = "healthcare_question_summarization"
        instructions = create_hypotheses(args, instruction_set, k_shot_idx_start=0, k_shot_idx_end=len(MEQSUM_PROMPT_FILES))
        return instructions
    elif args.dataset in ["red_team_chat", "full_chat"]:
        return [{'instruction': ins} for ins in instruction_sets["chat"]]
    elif args.dataset in ["mbpp", "human_eval"]:
        # return a list of dicts containing prompts and chosen k-shot examples
        if args.dataset == "mbpp":
            instruction_set = "code"
            # task_IDs range from 1-10, per github --> [1, 11)
            instructions = create_hypotheses(args, instruction_set, k_shot_idx_start=1, k_shot_idx_end=11)
            return instructions

def set_seeds(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

def subset_ensure_unique(dataset, n_total, text_col_name='text', shuffle=True):
    """Subsets the dataset to n_total examples and ensures that all examples are
    unique."""
    # shuffle the dataset
    if shuffle:
        idxs = list(range(len(dataset)))
        random.shuffle(idxs) # assuming seed has been set
        dataset = dataset.select(idxs)

    if n_total is None:
        n_total = len(dataset)
    
    # clip to requested number of examples    
    if len(dataset) < n_total:
        print(f"Warning: {n_total:,} examples requested but only {len(dataset):,} examples available. Using {len(dataset):,} examples.")
        n_total = len(dataset)
    dataset_slice = slice(0, n_total)
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.iloc[dataset_slice]
    elif isinstance(dataset, list):
        dataset = dataset[dataset_slice]
    elif isinstance(dataset, Dataset):
        # the behavior of slicing a Dataset is different than a list
        # it returns a dictionary of columns mapped to lists as opposed to records format
        dataset = dataset.select(range(n_total))

    # check if any of the text examples are duplicated
    # if so, warn user
    unique = set(dataset[text_col_name])
    n_unique = len(unique)
    n_total = len(dataset)
    if n_unique != n_total:
        print(f"Warning: {n_total - n_unique}/{n_total} examples are duplicates.")
    # TODO: consider removing duplicates and adding a while condition to pad the dataset with more unique examples
    return dataset

def extract_user_portion(convo_string):
    return convo_string.split("Human: ")[1].split("\n")[0].strip()

def prepare_chat_user_prompt(task_example, chat_col_name='rejected'):
    """Format user portion of the chat. Include system prompt at a later stage."""
    template = 'Your goal is to respond to the following human input:\nHuman: {}\nChatbot Response:'
    conversation = task_example[chat_col_name]
    human_input = extract_user_portion(conversation)
    prompt = template.format(human_input)
    return {'prompt': prompt}


def prepare_complete_chat_input(args, dataset, instruction, id_col_name='task_id'):
    """Prepares the complete chat input, including the system prompt and user prompt."""
    system_prompt = instruction['instruction']
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

    # retain the task_id (e.g., id) for each example
    final_dataset = []
    for i, dialog in enumerate(dialogs):
        example = {'text': dialog, 'task_id': dataset[i][id_col_name]}
        final_dataset.append(example)
    dataset = final_dataset
    return dataset

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

def prepare_summarization_user_prompt(task_example, dataset='cnn_dailymail'):
    """Prepares user portion of the prompt. The system prompt will be prepended
    to the user prompt at a later stage.
    """
    template = 'Summarize the following document:\n{}'
    # add task to be completed
    if dataset == 'cnn_dailymail':
        doc_col = 'article'
    elif dataset == 'xsum':
        doc_col = 'document'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    task = task_example[doc_col]
    prompt = template.format(task)
    return {'prompt': prompt}

def get_data(args, instruction=None):
    if args.dataset == "full_chat":
        dataset = load_dataset("Anthropic/hh-rlhf")["test"]
        # create an id column ranging from 0 to len(dataset)
        ids = list(range(len(dataset)))
        dataset = dataset.add_column('task_id', ids)
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='rejected')
        # if embed only, extract user portion of the chat
        if args.embed:
            dataset = dataset.map(lambda x: {'text': extract_user_portion(x['rejected'])}, batched=False)
            return dataset, None
        dataset = dataset.map(prepare_chat_user_prompt, batched=False, fn_kwargs={'chat_col_name': 'rejected'})
        dataset = prepare_complete_chat_input(args, dataset, instruction)
    elif args.dataset == "red_team_chat":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
        # create an id column ranging from 0 to len(dataset)
        ids = list(range(len(dataset)))
        dataset = dataset.add_column('task_id', ids)
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='transcript')
        # if embed only, extract user portion of the chat
        if args.embed:
            dataset = dataset.map(lambda x: {'text': extract_user_portion(x['transcript'])}, batched=False)
            return dataset, None
        dataset = dataset.map(prepare_chat_user_prompt, batched=False, fn_kwargs={'chat_col_name': 'transcript'})
        dataset = prepare_complete_chat_input(args, dataset, instruction)
    elif args.dataset == "bigbio/meqsum":
        dataset = load_dataset("bigbio/meqsum", "meqsum_source", split="train")
        prompt_dataset = dataset.filter(lambda x: x['File'] in MEQSUM_PROMPT_FILES)
        dataset = dataset.filter(lambda x: x['File'] not in MEQSUM_PROMPT_FILES)
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='CHQ')
        # prepare user prompt, potentially prepending with k-shot examples
        k_shot_idxs = instruction['k_shot_idxs'] if 'k_shot_idxs' in instruction else []
        dataset = dataset.map(prepare_meqsum_user_prompt, batched=False, fn_kwargs={'prompt_dataset': prompt_dataset, 'k_shot_idxs': k_shot_idxs})
        dataset = prepare_complete_chat_input(args, dataset, instruction, id_col_name='File')
    elif args.dataset == "mbpp":
        if 'codellama' not in args.model_name_or_path:
            raise ValueError("mbpp dataset is only supported for codellama models. If other models are needed, format the prompts accordingly.")
        dataset = load_dataset("mbpp")
        # task_id 1-10
        prompt_dataset = dataset['prompt']
        # concat train, validation, and test portions
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='text')
        # prepare user prompt, potentially prepending with k-shot examples
        k_shot_idxs = instruction['k_shot_idxs'] if 'k_shot_idxs' in instruction else []
        dataset = dataset.map(prepare_mbpp_user_prompt, batched=False, fn_kwargs={'prompt_dataset': prompt_dataset, 'k_shot_idxs': k_shot_idxs})
        dataset = prepare_complete_chat_input(args, dataset, instruction)
    elif args.dataset == 'cnn_dailymail':
        dataset = load_dataset('cnn_dailymail', name='3.0.0', split='train')
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='article')
        # this is an embedding use case so simply return the dataset without further formatting
        if args.embed:
            # rename 'article' to 'text' for ease downstream
            dataset = dataset.rename_column('article', 'text')
            dataset = dataset.rename_column('id', 'task_id')
            return dataset, None
        dataset = dataset.map(prepare_summarization_user_prompt, batched=False, fn_kwargs={'dataset': 'cnn_dailymail'})
        dataset = prepare_complete_chat_input(args, dataset, instruction, id_col_name='id')
    elif args.dataset == 'xsum':
        dataset = load_dataset("xsum", split="train")
        # cast id column to int so we can ensure the cache works correctly
        new_features = dataset.features.copy()
        new_features['id'] = Value('int64')
        dataset = dataset.cast(new_features)
        dataset = subset_ensure_unique(dataset, args.n_total, text_col_name='document')
        if args.embed:
            # rename 'document' to 'text' for ease downstream
            dataset = dataset.rename_column('document', 'text')
            dataset = dataset.rename_column('id', 'task_id')
            return dataset, None
        dataset = dataset.map(prepare_summarization_user_prompt, batched=False, fn_kwargs={'dataset': 'xsum'})
        dataset = prepare_complete_chat_input(args, dataset, instruction, id_col_name='id')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    return dataset, dataloader

def hash_row(row, cols):
    """Hash row based on specified columns for faster cache checking."""
    if isinstance(row, dict):
        values = [row[col] for col in cols]
        row_str = '|'.join([str(x) for x in values])
    else:
        row = row[cols]
        row_str = '|'.join([str(x) for x in row.values])
    row_bytes = row_str.encode('utf-8')
    row_hash = hashlib.sha256(row_bytes).hexdigest()
    return row_hash

def get_datasets_dataloaders(instructions, cache_df, args, return_dataloaders=False):
    datasets = []
    dataloaders = []
    for instruction in tqdm(instructions, desc="Instructions"):
        set_seeds(args.seed)
        # dataset is a list of text examples, dataloader is a DataLoader
        dataset, dataloader = get_data(args, instruction)
        # check if any of the text examples are duplicated
        # if so, warn user that they will be removed
        # can only check against hashable types (e.g., strings, etc.)
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
            # assume all examples have the same columns
            cols = list(dataset[0].keys())
            if 'hash' not in cache_df.columns:
                cache_df['hash'] = cache_df.apply(lambda x: hash_row(x, cols), axis=1)
            hash_vals = set(cache_df['hash'].values)
            num_examples = len(dataset)
            data_df = pd.DataFrame(dataset)
            data_hash_vals = data_df.apply(lambda x: hash_row(x, cols), axis=1)
            final_dataset = []
            # check if any of the examples have already been processed
            for i, hash_val in enumerate(data_hash_vals):
                if hash_val not in hash_vals:
                    final_dataset.append(dataset[i])
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
    cols =  [col for col in cols if col not in df.columns]
    if 'id' not in cols:
        cols = ['id'] + cols
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

def embed(args, save_folder, model_name):
    # load the dataset
    dataset, _ = get_data(args, instruction=None)
    # check if we've already computed the embeddings
    csv_path = os.path.join(save_folder, f'{model_name}_embeddings.csv')
    if os.path.exists(csv_path) and not args.no_cache:
        cache_df = pd.read_csv(csv_path)
        if len(cache_df) == len(dataset):
            print(f"Embeddings for {model_name} already computed for dataset {args.dataset}. Skipping.")
            return
    
    # TODO: keep track of what we've already computed
    # though this might be so fast that it doesn't matter if we recompute
    model = SentenceTransformer(args.model_name_or_path)

    # encode documents with specified number of gpus
    target_devices = [f'cuda:{i}' for i in range(args.num_gpus)]
    pool = model.start_multi_process_pool(target_devices)

    # compute the embeddings using the multi-process pool
    start = time.perf_counter()
    emb = model.encode_multi_process(dataset['text'], pool)
    end = time.perf_counter()
    print(f"Embeddings computed embeddings with shape {emb.shape} in {end - start:.2f} seconds. Time per example: {(end - start) / len(dataset):.2f} seconds.")

    # stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
    
    # save the embeddings as csv
    df = pd.DataFrame()
    df['embedding'] = emb.tolist()
    df['task_id'] = dataset['task_id']
    df.to_csv(os.path.join(save_folder, f'{model_name}_embeddings.csv'), index=False)
    

def generate_outputs(args, save_folder, model_name):
    """This is the main pipeline for generating outputs for a given dataset and
    model."""
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
        return

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
    print(f"Max length across all prompts in datasets: {max_len:,}")
    # set max_input_tokens to max length + 10 to account for special tokens and any variation in prompts
    args.max_input_length = max_len + 10
    # set max_total_tokens min of 3x max_input_tokens and 4096
    args.max_total_tokens = min(args.max_input_length * 3, 4096)
    args.max_new_tokens = args.max_total_tokens - args.max_input_length
    print(f"Max input length: {args.max_input_length:,}")
    print(f"Max total tokens: {args.max_total_tokens:,}")
    
    args.max_concurrent_requests = 128 # default
    if 'flan-t5' in args.model_name_or_path:
        # set max batch total tokens manually because TGI won't do it automatically for T5 models
        # this was set somewhat experimentally
        args.max_concurrent_requests = 200
        max_tokens = (args.num_gpus // 4) * 50_000
        args.max_batch_total_tokens = min(args.max_total_tokens*args.max_concurrent_requests, max_tokens)
    if 'lama' in args.model_name_or_path:
        # may need to increase max length for particularly long prompts
        # see https://huggingface.co/docs/text-generation-inference/basic_tutorials/preparing_model
        if args.max_input_length + 512 > 4096:
            new_max = args.max_input_length + 512
            print(f"Warning: max input length of {args.max_input_length:,} leaves little (or no) room for generation. Increasing max_total_tokens to {new_max:,} with RoPE scaling.")
            args.max_total_tokens = new_max
            args.max_new_tokens = args.max_total_tokens - args.max_input_length
            args.rope_scaling = 'dynamic'
            args.rope_factor = new_max / 4096
            # was gettting error: ArgumentValidation("`max_batch_prefill_tokens` must be >= `max_input_length`. Given: 4096 and 4874")
            # so set max_batch_prefill_tokens to max_input_length
            args.max_batch_prefill_tokens = args.max_input_length*2
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
    # stop the server
    stop_server(container)

def main(args):
    set_seeds(args.seed)

    # set HF token to ensure access to llama 2 models
    set_hf_token()

    # allow user to specify a collection of datasets and then loop over them
    if not args.datasets:
        args.datasets = [args.dataset]

    # TODO: add support for multiple models
    for dataset_ in args.datasets:
        args.dataset = dataset_
        # define output folder and cache csv path
        dataset_dir = args.dataset.replace("/", "_")
        save_folder = os.path.join(args.output_dir, dataset_dir)
        os.makedirs(save_folder, exist_ok=True)
        model_name = args.model_name_or_path.replace("/", "-")

        # major branch in pipeline: either generate outputs or embed the dataset
        if args.embed:
            # embed the dataset
            embed(args, save_folder, model_name)
        else:
            # generate outputs
            generate_outputs(args, save_folder, model_name)

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