# common/config_utils.py
import json

def _load_json_with_comments(path: str) -> dict:
    """A helper to load JSON and provide a better error message for common mistakes."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"--- CONFIGURATION FILE ERROR ---")
        print(f"Failed to parse JSON file at: {path}")
        print(f"Error message: {e}")
        print("\nCommon reasons for this error include:")
        print("1. A missing comma ',' between key-value pairs.")
        print("2. A trailing comma ',' after the last element in a list or object.")
        print("3. Using comments (like // or #), which are NOT supported in standard JSON.")
        print("\nPlease check the file content and try again.")
        # Re-raise the exception to stop execution
        raise e

def load_model_config(path: str) -> dict:
    return _load_json_with_comments(path)

def load_dataset_config(path: str) -> dict:
    return _load_json_with_comments(path)