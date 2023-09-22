"""This module implements argument parsing for the scripts package."""
import argparse


def parse_args(passed_args=None, known_only=False):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Produce loss distribution")
    # use seed set in tgi args
    # parser.add_argument(
    #     "--random-seed",
    #     type=int,
    #     default=42,
    #     help="random seed"
    # )
    parser.add_argument('--embed', action='store_true', default=False, help='Embed the generated text instead of generating text predictions.')
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=50,
        help="max length of LLM generations"
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=None, # 2000 is a good number
        help="number of evaluated datapoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dataset",
        default="xsum",
        help="dataset for experiments"
    )
    parser.add_argument('--datasets', nargs='+', default=[], help='Datasets to evaluate on.')
    # parser.add_argument('--use-tgi', action='store_true', default=False, help='Use the text-generation-inference server to serve the model.')
    parser.add_argument(
        "--model-size",
        default="base",
        help="FLAN T5 model size"
    )
    parser.add_argument(
        "--loss-fn",
        default="bertscore",
        type=str,
        help="Outer loss function"
    )
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=50,
        help="no. of hypotheses"
    )
    parser.add_argument('--k-shots', type=int, default=3, help='Maximum number of shots to use for few-shot learning. Can use between 1 and --k-shots.')
    parser.add_argument(
        "--save-after",
        type=int,
        default=0,
        help="when to start saving intermediate results (will overwrite previous final results)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="gpu device"
    )
    parser.add_argument(
        "--with-text",
        action="store_true"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
    )
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--no-cache', action='store_true', default=False, help='Do not use cached results.')
    parser.add_argument('--eval-models', nargs='+', default=[], help='Models to evaluate on.')
    parser.add_argument('--num-return-sequences', type=int, default=1, help='Number of sequences to return from the server. While TGI doesn\'t implement this, we simulate this by calling the server multiple times with different random seeds.')
    parser.add_argument('--s3-bucket-name', type=str, default='prompt-risk-control', help='S3 bucket name to upload to.')
    parser.add_argument('--s3-prefix', type=str, default='', help='S3 prefix to upload to.')
    parser.add_argument('--download-from-s3', action='store_true', default=False, help='Download from S3.')
    parser.add_argument('--upload-to-s3', action='store_true', default=False, help='Upload to S3.')
    if passed_args is not None:
        if known_only:
            args, unknown_args = parser.parse_known_args(passed_args)
            return args, unknown_args
        else:
            args = parser.parse_args(passed_args)
            return args
    else:
        if known_only:
            args, unknown_args = parser.parse_known_args()
            return args, unknown_args
        else:
            args = parser.parse_args()
            return args