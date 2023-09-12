"""This module produces loss values for model generated outputs.

Examples:
    $ python -m scripts.compute_loss \
        --output-dir output \
        --dataset red_team_chat
"""
import argparse
import os
import pandas as pd

from transformers import AutoTokenizer, pipeline
import torch

def main(args):
    # load generated outputs for all models
    files = os.listdir(os.path.join(args.output_dir, args.dataset))
    files = [f for f in files if f.endswith(".csv")]
    dfs = {}
    for f in files:
        model_id = f.split("_prediction.csv")[0]
        dfs[model_id] = pd.read_csv(os.path.join(args.output_dir, args.dataset, f))

        # check if all models have the expected number of rows populated with text
        try:
            assert dfs[model_id].shape[0] == 50*2000
        except:
            breakpoint()
        # check for any nans in generated_text
        assert dfs[model_id].generated_text.isna().sum() == 0
    
    rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
  
    rm_pipe = pipeline(
        "sentiment-analysis",
        model="weqweasdas/hh_rlhf_rm_open_llama_3b",
        device="cpu",
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }

    test_texts = [
    "###Human: My daughter wants to know how to convert fractions to decimals, but I'm not sure how to explain it. Can you help? ###Assistant: Sure. So one way of converting fractions to decimals is to ask “how many halves are there?” and then write this as a decimal number. But that's a little tricky. Here's a simpler way:  if a fraction is expressed as a/b, then it's decimal equivalent is just a/b * 1.0  So, for example, the decimal equivalent of 1/2 is 1/2 * 1.0 = 0.5.",
    "###Human: I have fresh whole chicken in my fridge. What dish can I prepare using it that will take me less than an hour to cook? ###Assistant: Are you interested in a quick and easy recipe you can prepare with chicken you have on hand, or something more involved?  In terms of both effort and time, what are you looking for?"]

    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    print(rewards)
    breakpoint()



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--dataset", type=str, default="full_chat")
    args = parser.parse_args()
    main(args)
