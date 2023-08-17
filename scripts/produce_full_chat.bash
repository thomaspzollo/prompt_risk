# python produce_loss_dist.py --dataset="full_chat" --loss_fn="toxicity" --model_size="small" --num_hypotheses=50 --batch_size=256 --device="cuda:2";
# python produce_loss_dist.py --dataset="full_chat" --loss_fn="toxicity" --model_size="base" --num_hypotheses=50 --batch_size=256 --device="cuda:2";
python produce_loss_dist.py --dataset="full_chat" --loss_fn="toxicity" --model_size="large" --num_hypotheses=50 --batch_size=64 --device="cuda:2";
python produce_loss_dist.py --dataset="full_chat" --loss_fn="toxicity" --model_size="XL" --num_hypotheses=50 --batch_size=16 --device="cuda:2";