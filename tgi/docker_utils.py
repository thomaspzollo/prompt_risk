"""This module uses the Docker Python library to start and manage Docker containers.

Examples:
    $ python -m tgi.docker \
        --hf-cache-dir data \
        --model-name-or-path meta-llama/Llama-2-7b-chat-hf \
        --num-gpus 1 \
        --server-port 8081 \
        --dtype bfloat16
"""
import argparse
import os
import subprocess
import time
import logging

import docker
import requests
from .args import parse_args

logging.basicConfig(level=logging.INFO)

def set_hf_token(token_path=None):
    """Set the environment variable for the HF inference server."""
    if token_path is None:
        # try to find the token in the default location
        token_path = os.path.expanduser("~/.cache/huggingface/token")
    token = subprocess.check_output(f"cat {token_path}",
                                    shell=True).decode("utf-8").strip()
    if token is None:
        logging.warning("HUGGING_FACE_HUB_TOKEN not found in environment, using default token")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

def server_live(host, server_port):
    """Check if the HF inference server is running."""
    try:
        response = requests.get(f"http://{host}:{server_port}/health")
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False


def start_server(args):
    """Start the HF inference server in Docker running in the background."""
    client = docker.from_env()

    environment = {}
    # get -e HUGGING_FACE_HUB_TOKEN from the environment
    hf_hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
    if hf_hub_token is None:
        logging.warning("HUGGING_FACE_HUB_TOKEN not found in environment, using default token")
    else:
        environment["HUGGING_FACE_HUB_TOKEN"] = hf_hub_token

    model_id = args.model_name_or_path
    num_shard = str(args.num_gpus)
    logging.info(f"Starting server with model {model_id} and {num_shard} shard(s)")
    start = time.perf_counter()
    # arguments passed to the main entrypoint of the container: text-generation-inference
    command = ["--model-id", model_id, "--num-shard", num_shard]
    # user can specify up to max_new_tokens = max_total_tokens - max_input_length in their request
    command.extend(["--max-input-length", str(args.max_input_length)])
    command.extend(["--max-total-tokens", str(args.max_total_tokens)])
    model_id_clean = model_id.replace("/", "-")
    container_name = f"text-generation-inference-{model_id_clean}-{num_shard}"
    if args.container_quantize:
        command.extend(["--quantize", "bitsandbytes"])
        container_name += "-quantized"

    # if name already in use, remove it
    try:
        container = client.containers.get(container_name)
        container.remove(force=True)
    except docker.errors.NotFound:
        pass

    # gpu list
    gpu_list = [str(i) for i in range(args.num_gpus)]
    container = client.containers.run(
        "ghcr.io/huggingface/text-generation-inference:latest",
        name=container_name,
        command=command,
        environment=environment,
        ports={"80/tcp": args.server_port},
        volumes={args.hf_cache_dir: {"bind": "/data", "mode": "rw"}},
        detach=True,
        # devices=["/dev/nvidia0:/dev/nvidia0"],
        # https://github.com/docker/docker-py/issues/2395#issuecomment-907243275
        # TODO: figure out how to control, which GPUs are available to the container
        # the following should correspond to --gpus all
        # if you want to restrict the number of GPUs you can use the following
        # in place of count=-1: device_ids=['0', '1']
        device_requests=[
            docker.types.DeviceRequest(device_ids=gpu_list, capabilities=[["gpu"]])
        ],
        shm_size="1g",
    )
    time.sleep(5)  # wait for container to start
    # print container output to be sure there are no errors
    logs = container.logs().decode('utf-8')
    if args.print_container_logs:
        print(logs)

    # loop until server is ready
    max_wait = 200
    total_wait = 0
    while total_wait < max_wait:
        server_is_live = server_live(args.host, args.server_port)
        if server_is_live:
            logging.info(f"Server {container_name} is ready!")
            break
        else:
            logging.info(f"Server {container_name} is not ready, waiting...")
            # get new logs
            new_logs = container.logs().decode('utf-8')
            if args.print_container_logs:
                incremental_logs = new_logs.replace(logs, "")
                print(incremental_logs)
            logs = new_logs 
            time.sleep(10)
            total_wait += 10
    if total_wait >= max_wait:
        logging.error(f"Server {container_name} is not ready after {max_wait} seconds")
        raise TimeoutError(f"Server {container_name} is not ready after {max_wait} seconds")
    end = time.perf_counter()
    logging.info(f"Server started in {end - start: .2f} seconds")
    container.reload()  # update container status
    return container


def stop_server(container=None):
    """Stop the HF inference server in Docker."""
    if container is None:
        return None
    logging.info(f"Stopping server..")
    container.stop()
    container.remove(force=True)
    logging.info(f"Server stopped")


def main(args):
    # set the environment variable for the HF inference server
    set_hf_token()
    
    container = start_server(args)
    stop_server(container)


if __name__ == "__main__":
    args = parse_args()
    main(args)
