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


def load_dataset(zip_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {zip_path}...")
    df = pd.read_csv(zip_path, compression="zip")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def build_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    """Split by domain into separate configs — here we use them as splits."""
    domains = df["domain"].unique()
    print(f"Domains found: {list(domains)}")

    splits = {}
    for domain in sorted(domains):
        subset = df[df["domain"] == domain].reset_index(drop=True)
        print(f"  {domain}: {len(subset):,} rows")
        splits[domain] = Dataset.from_pandas(subset, preserve_index=False)

    # Also include a combined "all" split
    splits["all"] = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
    print(f"  all: {len(df):,} rows")

    return DatasetDict(splits)


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
    args = parser.parse_args()

    df = load_dataset(args.zip_path)
    dataset_dict = build_dataset_dict(df)
    push_to_hub(dataset_dict, args.repo_id)
