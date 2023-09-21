# Prompt Risk Control
Prompt risk control is a framework for selecting prompts that minimize a risk criterion (e.g., error rate, toxicity, etc.). This framework accounts for more than just empirical average performance on a validation set when selecting a prompt. It takes into account the worst-case performance of a prompt through the use of metrics like conditional value at risk (CVaR). While problematic generations are rare for many language models, they can be catastrophic in real-world applications. This framework allows users to select prompts that minimize the risk of such generations.

## Environment Setup
To install the dependencies, run the following command:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installing crossprob requires a bit of manual effort. First, clone the repository:
```bash
git clone git@github.com:mosco/crossing-probability.git
``` 
If you don't have swig and libfftw3 installed, install it with `sudo apt update && sudo apt install -y swig libfftw3-dev`.

Note that I then had to add `#include <limits>` to `src/common.cc` and
```c++
#include <iostream>
#include <iterator>
```
to `src/common.hh` to get it to compile.

Make the C++ library:
```bash
cd crossing-probability
make
```

Then, install the Python package:
```bash
make python
python setup.py install
```

Finally, login to the HuggingFace hub with `huggingface-cli login` and enter your credentials. This is required to be able to pull down Llama models. If you don't yet have access to Llama 2 models, request access [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), followed by requesting access on Hugging Face.

Install Docker with NVIDIA support, if necessary, following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-docker).

## Manually running a text-generation-inference server
```
HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
model=meta-llama/Llama-2-7b-hf
model=google/flan-t5-xxl
num_shard=4
volume=$PWD/data
docker run \
    --gpus all \
    --shm-size 1g \
    -p 8081:80 \
    -v $volume:/data \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    ghcr.io/huggingface/text-generation-inference:latest \
        --model-id $model \
        --num-shard $num_shard \
        --max-input-length 512 \
        --max-total-tokens 1024 \
        --dtype float16
```

Benchmark and tune TGI to maximize throughput (optional):
```bash
docker-compose up -d
docker-compose exec benchmark /bin/bash
text-generation-benchmark --sequence-length 2048 --decode-length 200 --runs 5 --batch-size 4
```

## Make predictions
Run Red Team Chat on Flan-T5:
```bash
nohup python -u -m scripts.generate_outputs \
    --datasets red_team_chat full_chat \
    --use_tgi \
    --model-name-or-path google/flan-t5-xxl \
    --num-gpus 4 \
    --server-port 8081 \
    --dtype float16 \
    --print-container-logs \
    --n_total 2000 \
    --num_hypotheses 50 \
> generate_outputs.log 2>&1 &
```

Run Code Llama on MBPP:
```bash
nohup python -u -m scripts.generate_outputs \
    --datasets mbpp \
    --use-tgi \
    --model-name-or-path codellama/CodeLlama-7b-Instruct-hf \
    --num-gpus 1 \
    --server-port 8081 \
    --dtype float16 \
    --print-container-logs \
    --n-total 300 \
    --num-hypotheses 20 \
    --num-return-sequences 10 \
    --seed 42 \
    --do-sample \
> generate_outputs.log 2>&1 &
```

Run MeQSum on Falcon 7b:
```bash
nohup python -u -m scripts.generate_outputs \
    --datasets bigbio/meqsum \
    --use-tgi \
    --model-name-or-path tiiuae/falcon-7b-instruct \
    --num-gpus 1 \
    --server-port 8081 \
    --dtype float16 \
    --print-container-logs \
    --n-total 1000 \
    --num-hypotheses 50 \
    --seed 42 \
> generate_outputs.log 2>&1 &
```
Generate embeddings for full_chat/red_team_chat:
```bash
# red_team_chat
python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 2000 \
    --batch-size 1000 \
    --seed 42 \
    --embed
```
Generate embeddings for CNN Daily Mail corpus:
```bash
python -u -m scripts.generate_outputs \
    --datasets cnn_dailymail \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 10000 \
    --batch-size 1000 \
    --seed 42 \
    --embed
```

Generate embeddings for XSUM:
```bash
python -u -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 10000 \
    --batch-size 1000 \
    --seed 42 \
    --embed
```

## Upload outputs to S3
First configure your AWS credentials with `aws configure`. Follow [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#cliv2-linux-install) if you need to install the aws CLI.

Then run:
```bash
python -m scripts.upload_to_s3 \
    --output-dir ./output \
    --s3-bucket-name prompt-risk-control \
```