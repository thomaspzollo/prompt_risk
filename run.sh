# main script for running all experiments in succession
NUM_GPUS=8

# full_chat tests
# embed
echo "Embedding full_chat"
python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total 5000 \
    --batch-size 1000 \
    --seed 42 \
    --embed

# generate with flan-t5
echo "Generating full_chat with flan-t5-xxl"
python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path google/flan-t5-xxl \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 50 \
    --seed 42

# eval full_chat
echo "Evaluating full_chat"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets full_chat \
    --loss-fn weqweasdas/hh_rlhf_rm_open_llama_3b \
    --batch-size 5 \
    --eval-models google/flan-t5-xxl

# red_team_chat tests
# embed
echo "Embedding red_team_chat"
python -u -m scripts.generate_outputs \
    --datasets red_team_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total 5000 \
    --batch-size 1000 \
    --seed 42 \
    --embed

# generate with flan-t5
echo "Generating red_team_chat with flan-t5-xxl"
python -u -m scripts.generate_outputs \
    --datasets red_team_chat \
    --model-name-or-path google/flan-t5-xxl \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 50 \
    --seed 42

# eval red_team_chat
echo "Evaluating red_team_chat"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets red_team_chat \
    --loss-fn weqweasdas/hh_rlhf_rm_open_llama_3b \
    --batch-size 5 \
    --eval-models google/flan-t5-xxl

# cnn_dailymail tests
# embed
echo "Embedding cnn_dailymail"
python -u -m scripts.generate_outputs \
    --datasets cnn_dailymail \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total 5000 \
    --batch-size 200 \
    --seed 42 \
    --embed

# generate with llama 2
echo "Generating cnn_dailymail with meta-llama/Llama-2-7b-chat-hf"
python -u -m scripts.generate_outputs \
    --datasets cnn_dailymail \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 20 \
    --seed 42

# eval cnn_dailymail
echo "Evaluating cnn_dailymail"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets cnn_dailymail \
    --loss-fn rouge

# xsum tests
# embed
echo "Embedding xsum"
python -u -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total 5000 \
    --batch-size 200 \
    --seed 42 \
    --embed

# generate with llama 2
echo "Generating xsum with meta-llama/Llama-2-7b-chat-hf"
python -u -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 20 \
    --seed 42

# eval xsum
echo "Evaluating xsum"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets xsum \
    --loss-fn rouge

echo "Generating meqsum with tiiuae/falcon-40b-instruct"
python -u -m scripts.generate_outputs \
    --datasets bigbio/meqsum \
    --model-name-or-path tiiuae/falcon-40b-instruct \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 20 \
    --seed 42

# eval meqsum
echo "Evaluating meqsum"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets bigbio/meqsum \
    --loss-fn rouge

# meqsum 7b (this only runs on a single GPU, run on a smaller server or do a better job of parallelizing)
echo "Generating meqsum with tiiuae/falcon-7b-instruct"
# have to use 1 gpu here: ValueError: `num_heads` must be divisible by `num_shards` (got `num_heads`: 71 and `num_shards`: 4) <-- who thought 71 was a good idea?
python -u -m scripts.generate_outputs \
    --datasets bigbio/meqsum \
    --model-name-or-path tiiuae/falcon-7b-instruct \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 2000 \
    --num-hypotheses 50 \
    --seed 42

# mbpp tests <-- this has bugs and I don't know why yet
echo "Generating mbpp with codellama/CodeLlama-7b-Instruct-hf"
python -u -m scripts.generate_outputs \
    --datasets mbpp \
    --model-name-or-path codellama/CodeLlama-7b-Instruct-hf \
    --num-gpus $NUM_GPUS \
    --print-container-logs \
    --n-total 1000 \
    --num-hypotheses 30 \
    --num-return-sequences 5 \
    --seed 42 \
    --do-sample

# eval mbpp
echo "Evaluating mbpp"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets mbpp \
    --loss-fn pass@k