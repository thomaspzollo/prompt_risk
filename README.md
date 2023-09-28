# Prompt Risk Control
Prompt Risk Control (PRC) is a framework for selecting prompts that minimize a risk criterion (e.g., error rate, toxicity, etc.). This framework accounts for more than just empirical average performance on a validation set when selecting a prompt. It takes into account the worst-case performance of a prompt through the use of metrics like conditional value at risk (CVaR). While problematic generations are rare for many language models, they can be catastrophic in real-world applications. This framework allows users to select prompts that minimize the risk of such generations.

## Environment Setup
To install the dependencies, run the following command:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-deps
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

## Make predictions for experiments
A useful starting point is to run `smoke_test.sh` to ensure everything is working properly. This will check that all model and dataset combinations are working as expected with a small amount of data.
```bash
./smoke_test.sh
```

Next we ramp up our pipeline output in waves. Note that the pipeline won't repeat work so we can build on our output incrementally. We ran `run.sh` to generate a moderate amount of output for all experiments of interest.
```bash
nohup ./run.sh \
> run.log 2>&1 &
```

Finally, we scale up in a few key places such as on the Anthropic datasets.
```bash
nohup ./run_chosen_hyps.sh \
> run_chosen_hyps.log 2>&1 &
```

### Debugging
You may find that that you need to debug the Docker component of the pipeline. To manually run a container, run:
```bash
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

To run a specific experiment, take a look at the `run.sh` script for examples of how to run the pipeline for a specific model and dataset. Here's a sample command:
```bash
nohup python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path google/flan-t5-xxl \
    --num-gpus 4 \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 50 \
    --seed 42
> generate_outputs.log 2>&1 &
```

## Analyze results
For section 5.1 of the paper we run the following notebooks in the notebooks directory:
```bash
code_score_pass_10.ipynb
mean_experiments.py with params "--single_prompt prompt_ind 0" and "--single_prompt prompt_ind 2"
code_bnd_cmps_exp.ipynb
```
For section 5.2 we run the following notebooks in the notebooks directory:
```bash
chat_score_tox.ipynb
chat_var_exp.ipynb
chat_dist_shift_exp.ipynb
```

For section 5.3 we run the following notebooks in the notebooks directory:
```bash
meqsum_gini_exp.ipynb
```

## Package up results
To package up the results for submission, run the following command:
```bash
./package_results.sh
```