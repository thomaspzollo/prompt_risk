# main script for running all experiments in succession

# full_chat tests
# embed
python -u -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 4 \
    --n-total 20 \
    --batch-size 20 \
    --seed 42 \
    --embed

# generate with flan-t5
python -m scripts.generate_outputs \
    --datasets full_chat \
    --model-name-or-path google/flan-t5-xl \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --seed 42

# eval full_chat
python -m scripts.compute_loss \
    --output-dir output \
    --datasets full_chat \
    --loss-fn weqweasdas/hh_rlhf_rm_open_llama_3b \
    --batch-size 30 \
    --eval-models google/flan-t5-xl

# red_team_chat tests
# embed
python -u -m scripts.generate_outputs \
    --datasets red_team_chat \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 20 \
    --batch-size 20 \
    --seed 42 \
    --embed

# generate with flan-t5
python -m scripts.generate_outputs \
    --datasets red_team_chat \
    --model-name-or-path google/flan-t5-xl \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --seed 42

# eval red_team_chat
python -m scripts.compute_loss \
    --output-dir output \
    --datasets red_team_chat \
    --loss-fn weqweasdas/hh_rlhf_rm_open_llama_3b \
    --batch-size 30 \
    --eval-models google/flan-t5-xl
 
# mbpp tests
python -m scripts.generate_outputs \
    --datasets mbpp \
    --model-name-or-path codellama/CodeLlama-7b-Instruct-hf \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --num-return-sequences 2 \
    --seed 42 \
    --do-sample

# eval mbpp
python -m scripts.compute_loss \
    --output-dir output \
    --datasets mbpp \
    --loss-fn pass@k

# meqsum tests
python -m scripts.generate_outputs \
    --datasets bigbio/meqsum \
    --model-name-or-path tiiuae/falcon-7b-instruct \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --seed 42

# eval meqsum
python -m scripts.compute_loss \
    --output-dir output \
    --datasets bigbio/meqsum \
    --loss-fn rouge

# cnn_dailymail tests
# embed
python -u -m scripts.generate_outputs \
    --datasets cnn_dailymail \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 20 \
    --batch-size 20 \
    --seed 42 \
    --embed

# generate with llama 2
python -m scripts.generate_outputs \
    --datasets cnn_dailymail \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --seed 42

# eval cnn_dailymail
python -m scripts.compute_loss \
    --output-dir output \
    --datasets cnn_dailymail \
    --loss-fn rouge

# xsum tests
# embed
python -u -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --num-gpus 2 \
    --n-total 20 \
    --batch-size 20 \
    --seed 42 \
    --embed

# generate with llama 2
python -m scripts.generate_outputs \
    --datasets xsum \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf \
    --num-gpus 1 \
    --print-container-logs \
    --n-total 20 \
    --num-hypotheses 2 \
    --seed 42

# eval xsum
python -m scripts.compute_loss \
    --output-dir output \
    --datasets xsum \
    --loss-fn rouge