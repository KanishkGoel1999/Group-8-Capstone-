#!/usr/bin/env python3
"""
s3_downloader.py

Download .csv and model artifact files from an S3 bucket into
local directories. Requires:
  - AWS credentials configured (aws configure, environment, or role)
  - boto3 installed (pip install boto3)
"""

import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class S3AssetDownloader:
    """
    Downloads CSVs and PyTorch model files from a specified S3 bucket
    into local 'data/' and 'model_artifacts/' directories.
    """

    def __init__(self,
                 bucket_name: str,
                 aws_profile: str = None,
                 region_name: str = None):
        """
        :param bucket_name: Name of the S3 bucket (e.g., 'influentialdata')
        :param aws_profile:  (Optional) AWS CLI profile to use
        :param region_name:  (Optional) AWS region (e.g., 'us-east-2')
        """
        session = boto3.Session(profile_name=aws_profile,
                                region_name=region_name) \
                  if aws_profile or region_name \
                  else boto3.Session()
        self.s3 = session.client('s3')
        self.bucket = bucket_name

        # Set up local directories relative to this script
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / 'data'
        self.model_dir = base_dir / 'model_artifacts'

        for d in (self.data_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)

    def list_all_keys(self):
        """
        Generator over all object keys in the S3 bucket.
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket):
                for obj in page.get('Contents', []):
                    yield obj['Key']
        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error listing objects: {e}")
            return

    def download_with_extension(self, ext: str, target_dir: Path):
        """
        Download all objects ending with `ext` into `target_dir`.
        :param ext: e.g. '.csv', '.pt', or '.pth'
        :param target_dir: Path object of local directory
        """
        for key in self.list_all_keys():
            if key.lower().endswith(ext):
                dest = target_dir / Path(key).name
                logging.info(f"→ Downloading {key} → {dest}")
                try:
                    self.s3.download_file(self.bucket, key, str(dest))
                except (BotoCoreError, ClientError) as e:
                    logging.error(f"Failed to download {key}: {e}")

    def run(self):
        """
        Download CSVs into data/ and model artifacts into model_artifacts/.
        """
        logging.info("Starting CSV downloads...")
        self.download_with_extension('.csv', self.data_dir)

        logging.info("Starting model artifact downloads (.pt, .pth)...")
        for ext in ('.pt', '.pth'):
            self.download_with_extension(ext, self.model_dir)

        logging.info("All downloads complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    downloader = S3AssetDownloader(bucket_name='influentialdata')
    downloader.run()
