import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

# A dictionary mapping the task names from lm-eval to their Hugging Face Hub IDs
DATASET_MAPPING = {
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "gsm8k": ("gsm8k", "main"),
    "hellaswag": "hellaswag",
    # vvvvvvvvvvvv THIS LINE IS CORRECTED vvvvvvvvvvvv
    "math":  "open-r1/OpenR1-Math-220k"
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}

def main():
    """
    Downloads datasets required for evaluation and saves them to a local cache directory.
    This allows for offline evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets for offline LLM evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./huggingface_cache",
        help="The local directory to download and cache the datasets."
    )
    args = parser.parse_args()

    # Ensure the cache directory exists
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Datasets will be saved to: {os.path.abspath(args.cache_dir)}\n")

    # Iterate over the datasets and download each one
    for task_name, hub_id in tqdm(DATASET_MAPPING.items(), desc="Downloading Datasets"):
        try:
            print(f"Downloading '{task_name}' from Hub ID: '{hub_id}'...")
            if isinstance(hub_id, tuple):
                # Handle datasets with specific subsets, like gsm8k
                path, name = hub_id
                load_dataset(path=path, name=name, cache_dir=args.cache_dir)
            else:
                load_dataset(path=hub_id, cache_dir=args.cache_dir)
            print(f"Successfully downloaded '{task_name}'.\n")
        except Exception as e:
            print(f"\nERROR: Failed to download '{task_name}'.")
            print(f"Hub ID: '{hub_id}'")
            print(f"Error details: {e}\n")
            print("Please check your network connection or the dataset ID and try again.")

    print("All specified datasets have been processed.")
    print(f"You can now run the evaluation script in offline mode using the cache at: {os.path.abspath(args.cache_dir)}")

if __name__ == "__main__":
    main()