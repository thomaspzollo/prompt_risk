"""This script uploads a specified directory to S3 or downloads from S3.

Examples:
    $ python -m scripts.s3 \
        --output-dir ./output \
        --s3-bucket-name prompt-risk-control \
        --upload-to-s3
    
    $ python -m scripts.s3 \
        --output-dir ./output/bigbio_meqsum \
        --s3-bucket-name prompt-risk-control \
        --s3-prefix bigbio_meqsum \
        --upload-to-s3
    
    $ python -m scripts.s3 \
        --output-dir ./output \
        --s3-bucket-name prompt-risk-control \
        --download-from-s3
        
    $ python -m scripts.s3 \
        --output-dir ./output \
        --s3-bucket-name prompt-risk-control \
        --s3-prefix bigbio_meqsum \
        --download-from-s3
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

def download_dir_from_s3(bucket_name, s3_prefix, dir_to_download):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for key in result['Contents']:
            if not key['Key'].endswith('/'):
                s3_path = key['Key']
                local_path = os.path.join(dir_to_download, s3_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_client.download_file(bucket_name, s3_path, local_path)
                print(f"Downloaded {s3_path} to {local_path}")

def main(args):
    if args.upload_to_s3 and args.download_from_s3:
        raise ValueError("Cannot upload and download at the same time.")
    bucket_name = args.s3_bucket_name
    local_dir = args.output_dir
    s3_prefix = args.s3_prefix
    if args.upload_to_s3:
        upload_dir_to_s3(bucket_name, local_dir, s3_prefix)
    elif args.download_from_s3:
        download_dir_from_s3(bucket_name, s3_prefix, local_dir)

if __name__ == '__main__':
    args = scripts_parse_args()
    main(args)