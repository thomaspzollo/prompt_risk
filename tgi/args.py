"""This module implements argument parsing for the text-generation-inference package."""
import argparse


def parse_args(passed_args=None, known_only=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host IP or domain name')
    parser.add_argument('--hf-cache-dir', type=str, default='data')
    parser.add_argument('--model-name-or-path',
                        type=str,
                        default='meta-llama/Llama-2-7b-hf')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        help=
        'float32, float16, or bfloat16. Note that bfloat16 won\'t be available on older GPUs (e.g. V100s)'
    )
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--server-port', type=int, default=8081)
    parser.add_argument('--container-quantize',
                        action='store_true',
                        default=False)
    parser.add_argument('--max-input-length', type=int, default=2048)
    parser.add_argument('--max-total-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--do-sample', action='store_true', default=False)
    parser.add_argument('--print-container-logs',
                        action='store_true',
                        default=False)
    if passed_args is not None:
        if known_only:
            args, unknown_args = parser.parse_known_args(passed_args)
            return args, unknown_args
        else:
            args = parser.parse_args(passed_args)
            return args
    args = parser.parse_args()
    return args