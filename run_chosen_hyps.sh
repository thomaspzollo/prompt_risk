# main script for running all experiments in succession
NUM_GPUS=4
N_TOTAL=8000
NUM_HYPOTHESES=20

# full_chat tests
# embed
echo "Embedding full_chat"
python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total 40000 \
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
    --n-total 40000 \
    --num-hypotheses 1 \
    --seed 42 \
    --use-chosen-hypotheses

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
    --n-total $N_TOTAL \
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
    --n-total $N_TOTAL \
    --num-hypotheses $NUM_HYPOTHESES \
    --seed 42 \
    --use-chosen-hypotheses

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
    --n-total $N_TOTAL \
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
    --n-total $N_TOTAL \
    --num-hypotheses $NUM_HYPOTHESES \
    --seed 42 \
    --use-chosen-hypotheses

# eval cnn_dailymail
echo "Evaluating cnn_dailymail"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets cnn_dailymail \
    --loss-fn rouge

# eval with bertscore too
echo "Evaluating cnn_dailymail with bertscore"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets cnn_dailymail \
    --loss-fn bertscore \
    --batch-size 400

# xsum tests
# embed
echo "Embedding xsum"
python -u -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus $NUM_GPUS \
    --n-total $N_TOTAL \
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
    --n-total $N_TOTAL \
    --num-hypotheses $NUM_HYPOTHESES \
    --seed 42 \
    --use-chosen-hypotheses

# eval xsum
echo "Evaluating xsum"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets xsum \
    --loss-fn rouge

# eval with bertscore too
echo "Evaluating xsum with bertscore"
python -u -m scripts.compute_loss \
    --output-dir output \
    --datasets xsum \
    --loss-fn bertscore \
    --batch-size 300