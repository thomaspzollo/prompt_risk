# python produce_loss_dist.py --dataset="xsum" --loss_fn="bertscore" --model_size="small" --num_hypotheses=50 --batch_size=256 --device="cuda:0";
# python produce_loss_dist.py --dataset="xsum" --loss_fn="bertscore" --model_size="base" --num_hypotheses=50 --batch_size=256 --device="cuda:0";
python produce_loss_dist.py --dataset="xsum" --loss_fn="bertscore" --model_size="large" --num_hypotheses=50 --batch_size=64 --device="cuda:0";
python produce_loss_dist.py --dataset="xsum" --loss_fn="bertscore" --model_size="XL" --num_hypotheses=50 --batch_size=16 --device="cuda:0";