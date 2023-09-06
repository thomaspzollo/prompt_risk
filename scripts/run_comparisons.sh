python mean_experiments.py --dataset=xsum --loss_fn=bertscore --save_csv;
python mean_experiments.py --dataset=red_team_chat --loss_fn=toxicity --save_csv;
python mean_experiments.py --dataset=full_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=1.0 --dataset=xsum --loss_fn=bertscore --save_csv;
python var_interval_experiments.py --beta_lo=0.9 --beta_hi=1.0 --dataset=xsum --loss_fn=bertscore --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=0.95 --dataset=xsum --loss_fn=bertscore --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=1.0 --dataset=red_team_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.9 --beta_hi=1.0 --dataset=red_team_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=0.95 --dataset=red_team_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=1.0 --dataset=full_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.9 --beta_hi=1.0 --dataset=full_chat --loss_fn=toxicity --save_csv;
python var_interval_experiments.py --beta_lo=0.75 --beta_hi=0.95 --dataset=full_chat --loss_fn=toxicity --save_csv;

