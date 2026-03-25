"""
Script to convert IMPaCTS.zip to Parquet and push to HuggingFace Hub.

Usage:
    pip install datasets huggingface_hub pandas pyarrow
    huggingface-cli login
    python push_to_hf.py --repo-id mpapucci/impacts
"""

import argparse
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


def build_dataset_dict_chunked(zip_path: str, chunk_size: int = 5000) -> DatasetDict:
    """Stream dataset using generators to minimize memory usage."""
    print(f"Streaming dataset from {zip_path} (chunk_size={chunk_size})...")
    
    # First pass: scan to find domains and count rows
    print("Scanning domains...")
    domains = set()
    row_count = 0
    for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
        domains.update(chunk_df["domain"].unique())
        row_count += len(chunk_df)
    
    domains = sorted(list(domains))
    print(f"Found {len(domains)} domains with {row_count:,} total rows\n")
    
    # Create generators for each domain + all
    def make_generator(filter_domain=None):
        """Generator that yields records from CSV, optionally filtered by domain."""
        processed = 0
        for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
            if filter_domain is None:
                records = chunk_df.to_dict(orient="records")
            else:
                records = chunk_df[chunk_df["domain"] == filter_domain].to_dict(orient="records")
            
            for record in records:
                yield record
                processed += 1
                if processed % 50000 == 0:
                    print(f"    Uploaded {processed:,} rows...")
    
    # Build dataset splits using generators
    print("Building domain splits...")
    dataset_splits = {}
    for domain in domains:
        print(f"  {domain}...", end="", flush=True)
        dataset_splits[domain] = Dataset.from_generator(
            lambda d=domain: make_generator(filter_domain=d)
        )
        print(f" ✓")
    
    # Also include combined "all" split
    print(f"  all...", end="", flush=True)
    dataset_splits["all"] = Dataset.from_generator(
        lambda: make_generator(filter_domain=None)
    )
    print(f" ✓")
    
    return DatasetDict(dataset_splits)


def push_to_hub(dataset_dict: DatasetDict, repo_id: str):
    print(f"\nPushing to HuggingFace Hub: {repo_id}")
    dataset_dict.push_to_hub(
        repo_id,
        private=False,  # set True if you want it private first
    )
    print(f"Done! View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-path",
        default="IMPaCTS.zip",
        help="Path to IMPaCTS.zip (default: IMPaCTS.zip)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo id, e.g. ItalianNLPLab/IMPaCTS",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for streaming (default: 10000 rows)",
    )
    args = parser.parse_args()

    dataset_dict = build_dataset_dict_chunked(args.zip_path, chunk_size=args.chunk_size)
    push_to_hub(dataset_dict, args.repo_id)
