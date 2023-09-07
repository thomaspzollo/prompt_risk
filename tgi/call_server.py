"""This module implements the interface to the call the text-generation-inference server.

Examples:
    $ python -m tgi.call_server \
        --print-container-logs
"""
from queue import Queue
import os
import subprocess
import threading
import time
import pandas as pd

import requests
import torch
import gpustat
from text_generation import Client
from tqdm import tqdm  # InferenceAPIAsyncClient, InferenceAPIClient

from .args import parse_args
from .docker_utils import server_live, set_hf_token, start_server, stop_server


def get_gpu_memory():
    if not torch.cuda.is_available():
        return {
            "torch_allocated_GBs": 0,
            "total_GPU_memory_used_GBs": 0,
            "GPU_memory_used_GBs": 0,
        }
    torch_allocated_gbs = torch.cuda.memory_allocated() / 1e9
    stats = gpustat.GPUStatCollection.new_query()
    memory_used = {gpu.index: gpu.memory_used / 1e3 for gpu in stats}
    total_memory_used = sum(memory_used.values())
    return {
        "torch_allocated_GBs": torch_allocated_gbs,
        "total_GPU_memory_used_GBs": total_memory_used,
        "GPU_memory_used_GBs": memory_used,
    }


def call_server(prompt,
                url=None,
                timeout=1000,
                return_response_dict=False,
                generate_args={}):
    """Calls the HF inference server with the given prompt and return the response."""

    client = Client(url, timeout=timeout)

    # call server
    start = time.perf_counter()
    response = client.generate(prompt, **generate_args)
    text = response.generated_text
    end = time.perf_counter()

    # compute stats
    # NB: there may be more detailed logs in the server
    response_time = end - start
    token_count = len(response.details.tokens)
    tokens_per_second = token_count / response_time
    milliseconds_per_token = response_time / token_count * 1000

    # get GPU memory usage
    gpu_memory = get_gpu_memory()

    response_dict = {
        "generated_text": text,
        "response_time": response_time,
        "tokens_per_second": tokens_per_second,
        "milliseconds_per_token": milliseconds_per_token,
        "torch_allocated_GBs": gpu_memory["torch_allocated_GBs"],
        "total_GPU_memory_used_GBs": gpu_memory["total_GPU_memory_used_GBs"],
        "GPU_memory_used_GBs": gpu_memory["GPU_memory_used_GBs"],
        "response_object": response,
    }
    if return_response_dict:
        return text, response_dict
    return text

def call_server_thread(queue, output_queue, generate_args, args):
    while True:
        example = queue.get()
        if example is None:
            break
        text = example['text']
        url = f'http://{args.host}:{args.server_port}'
        text, response_dict = call_server(text, url, return_response_dict=True, generate_args=generate_args)
        # TODO: if we get an error, we should add a little sleep and retry
        # pop the response_object because we can't serialize it
        response_object = response_dict.pop('response_object')
        # retrieve the finish reason
        finish_reason = response_object.details.finish_reason
        response = {'id': example['id'], 'finish_reason': finish_reason}
        response.update(response_dict)
        output_queue.put(response)
        queue.task_done()
        # report on progress every 100 examples
        if example['id'] % 100 == 0:
            print(f"Index: {example['id']}\nGenerated: {response['generated_text']}")

def make_predictions(text_examples, args, num_threads=2):
    """Make predictions with the model in TGI server and save predictions to disk."""
    # TODO: consider adding support to format text examples for llama chat models
    # give examples an ID to track their order
    text_examples = [{'id': i, 'text': x} for i, x in enumerate(text_examples)]
    
    # parallelize API calls
    # TODO: switch over to asyncio and add error handling
    # create work queue
    queue = Queue()
    output_queue = Queue()

    # add all prompts to the queue
    for x in text_examples:
        queue.put(x)

    # add sentinel values to the queue
    sentinel_count = 1 if num_threads == 0 else num_threads 
    for _ in range(sentinel_count):
        queue.put(None)
    
    max_new_tokens = args.max_new_tokens
    if args.max_new_tokens is None:
        max_new_tokens = args.max_total_tokens - args.max_input_length
    generate_args = {'max_new_tokens': max_new_tokens,
                     'top_k': args.top_k,
                     'seed': args.seed,
                     'do_sample': not args.dont_sample}

    start = time.time()
    # useful to be able to call the server in the main thread for debugging
    if num_threads == 0:
        # run in the main thread
        call_server_thread(queue, output_queue, generate_args, args)
    else:
        threads = []
        for _ in range(num_threads):
            # call_server_thread(queue, output_queue, args)
            t = threading.Thread(target=call_server_thread,
                                args=(queue, output_queue, generate_args, args))
            t.start()
            threads.append(t)

        # wait for all threads to finish
        for t in threads:
            t.join()
    end = time.time()
    print(
        f"Completed {len(text_examples):,} examples in {end - start:,.2f} seconds. \
            Time per example: {(end - start) / len(text_examples):.2f} seconds.")
    
    # save the predictions
    predictions = []
    while not output_queue.empty():
        predictions.append(output_queue.get())

    df = pd.DataFrame(predictions)
    # sort by the original index
    df = df.sort_values(by='id')
    return df

def get_batch_size(container, max_total_tokens):
    # parse container logs to determine how many tokens we can support concurrently
    # then set the batch size accordingly
    logs = container.logs(stream=False).decode('utf-8')
    # grep for Setting max batch total tokens to
    search_string = 'Setting max batch total tokens to '
    line = [line for line in logs.split('\n') if search_string in line][0]
    max_batch_total_tokens = int(line.split(search_string)[1])
    max_batch_size = max_batch_total_tokens // max_total_tokens
    # model can only handle 128 concurrent requests
    if max_batch_size > 128:
        print(f"Warning: max batch size of {max_batch_size:,} is too large. Setting to 128.")
        print(f"Consider increasing the maximum number of concurrent requests in the TGI server.")
        max_batch_size = 128
    print(f"Setting max batch size to {max_batch_size:,} based on max batch total tokens of {max_batch_total_tokens:,} and input length of {max_total_tokens:,}.")
    return max_batch_size

def main(args):
    # set the environment variable for the HF inference server
    set_hf_token()

    container = start_server(args)
    # get the max batch size
    max_batch_size = get_batch_size(container, args.max_total_tokens)

    url = f'http://{args.host}:{args.server_port}'

    # configure decoding parameters
    # https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate
    # num_return_sequences=1, - unclear how to sample multiple sequences from server
    generate_args = {
        "do_sample": True,
        "top_k": 50,
        "seed": 42,
    }
    generate_args["max_new_tokens"] = 20

    # call the server
    text, response_dict = call_server(
        "Hello, world!",
        url=url,
        return_response_dict=True,
        generate_args=generate_args,
    )
    print(f'Generated text:\n{text}\n')
    print(response_dict)

    # now test with make_predictions
    text_examples = [
        "Hello, world!",
        "This is a test.",
        "This is a test. This is only a test.",
    ]
    # ensure we get fast responses
    args.max_new_tokens = 20
    df = make_predictions(text_examples, args, num_threads=0)
    print(df)
    # check that we can call the server in a multi-threaded fashion
    df = make_predictions(text_examples, args, num_threads=max_batch_size)
    print(df)

    # stop the server
    stop_server(container)


if __name__ == "__main__":
    args = parse_args()
    main(args)