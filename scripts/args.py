"""This module implements argument parsing for the scripts package."""
import argparse


def parse_args(passed_args=None, known_only=False):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Produce loss distribution")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="random seed"
    )
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
    parser.add_argument('--use-tgi', action='store_true', default=False, help='Use the text-generation-inference server to serve the model.')
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