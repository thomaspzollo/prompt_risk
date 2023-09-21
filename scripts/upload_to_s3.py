"""This script uploads a specified directory to S3.

Examples:
    $ python -m scripts.upload_to_s3 \
        --output-dir ./output \
        --s3-bucket-name prompt-risk-control
    
    $ python -m scripts.upload_to_s3 \
        --output-dir ./output/bigbio_meqsum \
        --s3-bucket-name prompt-risk-control \
        --s3-prefix bigbio_meqsum
"""
import os

from .args import parse_args as scripts_parse_args

import boto3

def upload_dir_to_s3(bucket_name, dir_to_upload, s3_prefix):
    s3_client = boto3.client('s3')
    for root, _, filenames in os.walk(dir_to_upload):
        for filename in filenames:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, dir_to_upload)
            s3_path = os.path.join(s3_prefix, relative_path)
            s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to {s3_path}")

def main(args):
    bucket_name = args.s3_bucket_name
    dir_to_upload = args.output_dir
    s3_prefix = args.s3_prefix
    upload_dir_to_s3(bucket_name, dir_to_upload, s3_prefix)

if __name__ == '__main__':
    args = scripts_parse_args()
    main(args)