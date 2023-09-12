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
num_shard=1
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
        --max-input-length 2048 \
        --max-total-tokens 4096 \
        --dtype bfloat16
```

## Benchmark to maximize throughput
Install the tool locally (this is probably easier than having to build a Docker Compose file).
```bash
git clone git@github.com:huggingface/text-generation-inference.git
cd text-generation-inference
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

sudo apt-get install libssl-dev gcc -y

conda create -n text-generation-inference python=3.11
conda activate text-generation-inference
BUILD_EXTENSIONS=True make install

```
```bash
model=bigscience/bloom-560m
num_shard=1
volume=$PWD/data
docker run \
    --gpus all \
    --shm-size 1g \
    -p 8081:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
        --model-id $model \
        --num-shard $num_shard
```

Now run the benchmarking tool:
```bash
model=bigscience/bloom-560m
num_shard=1
volume=$PWD/data
docker run \
    -v $volume:/data \
    --entrypoint text-generation-benchmark \
    ghcr.io/huggingface/text-generation-inference:latest \
        --tokenizer-name bigscience/bloom-560m
```
--gpus all \
    --shm-size 1g \
    -p 8081:80 \
    -v $volume:/data \