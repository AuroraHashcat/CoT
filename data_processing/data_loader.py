# data_processing/data_loader.py
from datasets import load_dataset, get_dataset_split_names
from typing import List, Dict, Callable

# --- Individual Formatting Functions for each Dataset ---

def _format_commonsense_qa(item: Dict) -> Dict:
    """Formats a CommonsenseQA item."""
    question_text = item['question']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(item['choices']['label'], item['choices']['text'])])
    full_prompt = f"Question: {question_text}\n\nChoices:\n{choices_text}"
    return {"id": item['id'], "question": full_prompt, "answerKey": item['answerKey']}

def _format_arc_challenge(item: Dict) -> Dict:
    """Formats an ARC-Challenge item."""
    return _format_commonsense_qa(item) # ARC-C shares the same structure as CSQA

def _format_openbookqa(item: Dict) -> Dict:
    """Formats an OpenBookQA item."""
    question_text = item['question_stem']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(item['choices']['label'], item['choices']['text'])])
    full_prompt = f"Question: {question_text}\n\nChoices:\n{choices_text}"
    return {"id": item['id'], "question": full_prompt, "answerKey": item['answerKey']}

def _format_piqa(item: Dict) -> Dict:
    """Formats a PIQA item."""
    question_text = item['goal']
    choices_text = f"A. {item['sol1']}\nB. {item['sol2']}"
    full_prompt = f"Question: {question_text}\n\nWhich solution is correct?\nChoices:\n{choices_text}"
    answer_key = "A" if item['label'] == 0 else "B"
    return {"id": f"piqa_{item['goal'][:20]}", "question": full_prompt, "answerKey": answer_key}

def _format_social_i_qa(item: Dict) -> Dict:
    """Formats a SocialIQA item."""
    question_text = f"{item['context']}\nQuestion: {item['question']}"
    choices_text = f"A. {item['answerA']}\nB. {item['answerB']}\nC. {item['answerC']}"
    full_prompt = f"Context and Question:\n{question_text}\n\nChoices:\n{choices_text}"
    answer_key = chr(ord('A') + int(item['label']) - 1)
    return {"id": f"siqa_{item['context'][:20]}", "question": full_prompt, "answerKey": answer_key}

def _format_boolq(item: Dict) -> Dict:
    """Formats a BoolQ item."""
    question_text = f"Context: {item['passage']}\nQuestion: {item['question']}?"
    choices_text = "A. yes\nB. no"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    answer_key = "A" if item['answer'] else "B"
    # BoolQ doesn't have a unique ID, so we create one from the question
    return {"id": f"boolq_{item['question'][:30]}", "question": full_prompt, "answerKey": answer_key}

def _format_gsm8k(item: Dict) -> Dict:
    """Formats a GSM8K item."""
    import re
    question_text = item['question']
    answer_match = re.search(r'#### ([\d,.]+)', item['answer'])
    answer = answer_match.group(1).replace(',', '') if answer_match else "N/A"
    return {"id": f"gsm8k_{item['question'][:20]}", "question": question_text, "answerKey": answer}

# --- Data Loader Factory ---
DATASET_FORMATTERS: Dict[str, Callable[[Dict], Dict]] = {
    "CommonsenseQA": _format_commonsense_qa,
    "ARC-Challenge": _format_arc_challenge,
    "OpenBookQA": _format_openbookqa,
    "PIQA": _format_piqa,
    "SocialIQA": _format_social_i_qa,
    "BoolQ": _format_boolq,
    "GSM8K": _format_gsm8k,
}

def load_data(config: Dict) -> List[Dict]:
    """
    Main data loading function. It intelligently handles datasets with and without
    multiple configurations based on the provided JSON config.
    """
    dataset_name = config['dataset_name']
    
    if dataset_name not in DATASET_FORMATTERS:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Please add a formatter for it in data_loader.py.")

    formatter = DATASET_FORMATTERS[dataset_name]
    
    hf_id = config['hf_id']
    hf_config = config.get('hf_config', None)
    split = config['split']
    num_samples = config['num_samples']
    
    print(f"Loading data from Hugging Face Hub...")
    print(f"  ID:      '{hf_id}'")
    if hf_config:
        print(f"  Config:  '{hf_config}'")
    print(f"  Split:   '{split}'")
    
    try:
        load_args = {'path': hf_id, 'name': hf_config} if hf_config else {'path': hf_id}
        
        available_splits = get_dataset_split_names(**load_args)
        if split not in available_splits:
            raise ValueError(f"Split '{split}' not found for dataset '{hf_id}'. Available splits: {available_splits}")

        dataset = load_dataset(**load_args, split=split, streaming=True).take(num_samples)

    except Exception as e:
        print(f"\n--- DATASET LOADING FAILED ---")
        print(f"An error occurred while trying to load '{hf_id}'.")
        print(f"Error details: {e}")
        print("Please check:\n1. Your internet connection.\n2. The dataset identifier and configuration in your JSON file are correct on Hugging Face Hub.\n3. The specified split exists for this dataset.")
        raise e

    formatted_data = [formatter(item) for item in dataset]
    print(f"Successfully loaded and formatted {len(formatted_data)} samples.")
    return formatted_data