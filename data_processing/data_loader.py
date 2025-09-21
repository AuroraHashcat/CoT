# data_processing/data_loader.py
from datasets import load_dataset, get_dataset_split_names
from typing import List, Dict, Callable
import json
import os

# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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

def _format_strategyqa(item: Dict) -> Dict:
    """Formats a StrategyQA item."""
    # StrategyQA 是 yes/no 问题，类似于 BoolQ
    question_text = item['question']
    # 如果有 facts 字段，可以作为上下文
    if 'facts' in item and item['facts']:
        question_text = f"Context: {item['facts']}\nQuestion: {question_text}"
    
    choices_text = "A. yes\nB. no"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # answer 字段是 True/False，转换为 A/B
    answer_key = "A" if item['answer'] else "B"
    
    return {"id": item.get('qid', f"strategyqa_{item['question'][:30]}"), "question": full_prompt, "answerKey": answer_key}

def _format_gsm8k(item: Dict) -> Dict:
    """Formats a GSM8K item."""
    import re
    question_text = item['question']
    answer_match = re.search(r'#### ([\d,.]+)', item['answer'])
    answer = answer_match.group(1).replace(',', '') if answer_match else "N/A"
    return {"id": f"gsm8k_{item['question'][:20]}", "question": question_text, "answerKey": answer}

def _format_drop(item: Dict) -> Dict:
    """Formats a DROP dataset item (reading comprehension)."""
    question_text = item['Question']
    answer = item['Answer']
    # For DROP, we just return the question and answer directly since it's open-ended
    return {"id": f"drop_{question_text[:30]}", "question": question_text, "answerKey": answer}

def _format_local_json(item: Dict) -> Dict:
    """Formats items from local JSON files - auto-detects format."""
    # Try to detect the format based on available keys
    if 'context' in item and 'asks-for' in item and 'choice_0' in item:
        # COPA format
        return _format_copa(item)
    elif 'context' in item and 'ask-for' in item and 'choice_id0' in item:
        # CausalNet format
        return _format_causalnet(item)
    elif 'Question' in item and 'Answer' in item:
        # Check if it's HellaSwag format (has numbered choices in Question)
        if any(f"{i}." in item['Question'] for i in range(4)):
            return _format_hellaswag(item)
        # Check if it's Math format (mathematical content)
        elif any(char in item['Question'] for char in ['$', '\\', '=', '+', '-', '*', '/']):
            return _format_math(item)
        else:
            # Default DROP format
            return _format_drop(item)
    elif 'question' in item and 'choices' in item:
        # CommonsenseQA-like format
        return _format_commonsense_qa(item)
    elif 'question' in item and 'answer' in item:
        # Generic Q&A format
        return {"id": f"local_{item.get('id', item['question'][:30])}", "question": item['question'], "answerKey": item['answer']}
    else:
        raise ValueError(f"Unknown local JSON format for item: {item}")

def _format_hellaswag(item: Dict) -> Dict:
    """Formats a HellaSwag item."""
    if 'Question' in item and 'Answer' in item:
        # Local JSON format
        question_text = item['Question']
        answer_key = item['Answer']
        return {"id": f"hellaswag_{question_text[:30]}", "question": question_text, "answerKey": answer_key}
    else:
        # Standard HuggingFace format
        context = item['ctx']
        question = f"{context}\n\nWhich ending makes the most sense?"
        
        choices = item['endings']
        choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        full_prompt = f"{question}\n\nChoices:\n{choices_text}"
        
        answer_key = str(item['label'])
        return {"id": f"hellaswag_{item.get('ind', item['ctx'][:20])}", "question": full_prompt, "answerKey": answer_key}

def _format_math(item: Dict) -> Dict:
    """Formats a Math dataset item."""
    if 'Question' in item and 'Answer' in item:
        # Local JSON format
        question_text = item['Question']
        answer = item['Answer']
        return {"id": f"math_{question_text[:30]}", "question": question_text, "answerKey": answer}
    else:
        # Standard HuggingFace format (MATH dataset)
        question_text = item['problem']
        answer = item['solution']
        return {"id": f"math_{question_text[:30]}", "question": question_text, "answerKey": answer}

def _format_winogrande(item: Dict) -> Dict:
    """Formats a WinoGrande item."""
    sentence = item['sentence']
    option1 = item['option1']
    option2 = item['option2']
    
    # 构建问题文本，包含句子和选择项
    question_text = f"Fill in the blank: {sentence}"
    choices_text = f"A. {option1}\nB. {option2}"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # 处理答案：'1' -> 'A', '2' -> 'B'，空字符串表示测试集
    if item['answer'] == '1':
        answer_key = 'A'
    elif item['answer'] == '2':
        answer_key = 'B'
    else:
        answer_key = ''  # 测试集没有答案
    
    # 生成ID，使用句子的前30个字符
    item_id = f"winogrande_{sentence[:30].replace(' ', '_')}"
    
    return {"id": item_id, "question": full_prompt, "answerKey": answer_key}

def _format_codah(item: Dict) -> Dict:
    """Formats a CODAH item."""
    question_text = item['question_propmt']  # Note: there's a typo in the dataset - it's 'propmt' not 'prompt'
    candidate_answers = item['candidate_answers']
    
    # Create choice labels A, B, C, D
    choices_text = "\n".join([f"{chr(ord('A') + i)}. {answer}" for i, answer in enumerate(candidate_answers)])
    full_prompt = f"Question: {question_text}\n\nChoices:\n{choices_text}"
    
    # Convert answer index to letter
    correct_idx = item['correct_answer_idx']
    answer_key = chr(ord('A') + correct_idx)
    
    return {"id": f"codah_{item['id']}", "question": full_prompt, "answerKey": answer_key}

def _format_com2sense(item: Dict) -> Dict:
    """Formats a COM2SENSE item."""
    statement = item['statement']
    question_text = f"Statement: {statement}\nIs this statement true or false?"
    choices_text = "A. True\nB. False"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    answer_key = "A" if item['label'] else "B"
    return {"id": f"com2sense_{statement[:30]}", "question": full_prompt, "answerKey": answer_key}

def _format_creak(item: Dict) -> Dict:
    """Formats a CREAK item."""
    sentence = item['sentence']
    question_text = f"Statement: {sentence}\nIs this statement true or false?"
    choices_text = "A. True\nB. False"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # 处理标签：'true' -> 'A', 'false' -> 'B'
    if item['label'].lower() == 'true':
        answer_key = 'A'
    else:
        answer_key = 'B'
    
    return {"id": item.get('ex_id', f"creak_{sentence[:30]}"), "question": full_prompt, "answerKey": answer_key}

def _format_proofwriter(item: Dict) -> Dict:
    """Formats a ProofWriter item."""
    # 解析翻译字段中的英文内容
    en_content = item['translation']['en']
    
    # 解析内容：格式为 $answer$ ; $proof$ ; $question$ = ... ; $context$ = ...
    parts = en_content.split(' ; $context$ = ')
    if len(parts) != 2:
        # 如果格式不符合预期，返回原始内容
        return {"id": f"proofwriter_{en_content[:30]}", "question": en_content, "answerKey": ""}
    
    question_part = parts[0]
    context = parts[1]
    
    # 进一步解析问题部分：$answer$ ; $proof$ ; $question$ = ...
    question_split = question_part.split(' ; $question$ = ')
    if len(question_split) != 2:
        return {"id": f"proofwriter_{en_content[:30]}", "question": en_content, "answerKey": ""}
    
    answer_proof_part = question_split[0]
    question_text = question_split[1]
    
    # 构建完整的问题文本，包含上下文和问题
    full_prompt = f"Context: {context}\n\nQuestion: {question_text}\n\nChoices:\nA. True\nB. False"
    
    # 从ro字段获取答案
    ro_content = item['translation']['ro']
    if 'True' in ro_content:
        answer_key = 'A'
    elif 'False' in ro_content:
        answer_key = 'B'
    else:
        answer_key = ''  # 如果无法确定答案
    
    return {"id": f"proofwriter_{question_text[:30]}", "question": full_prompt, "answerKey": answer_key}

def _format_cladder(item: Dict) -> Dict:
    """Formats a CLadder item."""
    prompt = item['prompt']
    question_text = f"Causal Reasoning Question:\n{prompt}"
    choices_text = "A. yes\nB. no"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # 处理标签：'yes' -> 'A', 'no' -> 'B'
    answer_key = "A" if item['label'].lower() == 'yes' else "B"
    
    return {"id": f"cladder_{item.get('id', prompt[:30])}", "question": full_prompt, "answerKey": answer_key}

def _format_causalnet(item: Dict) -> Dict:
    """Formats a CausalNet item."""
    context = item['context']
    ask_for = item['ask-for']
    
    # 构建问题文本
    question_text = f"Context: {context}\n\nQuestion: What is the most plausible {ask_for} in this scenario?"
    
    # 构建选择项
    choices = []
    choice_labels = ['A', 'B', 'C']
    for i in range(3):
        choice_key = f'choice_id{i}'
        if choice_key in item:
            choices.append(f"{choice_labels[i]}. {item[choice_key]}")
    
    choices_text = "\n".join(choices)
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # 转换标签为选择项字母
    label = item['label']
    answer_key = choice_labels[label] if 0 <= label < len(choice_labels) else 'A'
    
    return {"id": item.get('index', f"causalnet_{context[:30]}"), "question": full_prompt, "answerKey": answer_key}

def _format_copa(item: Dict) -> Dict:
    """Formats a COPA item."""
    context = item['context']
    asks_for = item['asks-for']
    
    # 构建问题文本，根据asks-for类型调整问题表述
    if asks_for == 'cause':
        question_text = f"Context: {context}\n\nWhat was the CAUSE of this?"
    else:  # effect
        question_text = f"Context: {context}\n\nWhat was the EFFECT of this?"
    
    # 构建选择项
    choice_0 = item['choice_0']
    choice_1 = item['choice_1']
    choices_text = f"A. {choice_0}\nB. {choice_1}"
    full_prompt = f"{question_text}\n\nChoices:\n{choices_text}"
    
    # 转换答案：0 -> 'A', 1 -> 'B'
    answer_key = 'A' if item['answer'] == 0 else 'B'
    
    return {"id": item['id'], "question": full_prompt, "answerKey": answer_key}

def _format_gpqa(item: Dict) -> Dict:
    """Formats a GPQA item (local multiple choice)."""
    # 题干
    question_text = item['question']
    # 选项（如 ["a. 14", "b. 11", ...]）
    choices = item.get('choices', [])
    choices_text = "\n".join(choices)
    # 构建完整问题
    full_prompt = f"Question: {question_text}\n\nChoices:\n{choices_text}"
    # 答案（如 "b"）
    answer_key = item.get('answerKey', '')
    # id 字段（如无则用前30字）
    item_id = item.get('id', f"gpqa_{question_text[:30]}")
    return {"id": item_id, "question": full_prompt, "answerKey": answer_key}

# --- Data Loader Factory ---
DATASET_FORMATTERS: Dict[str, Callable[[Dict], Dict]] = {
    "CommonsenseQA": _format_commonsense_qa,
    "ARC-Challenge": _format_arc_challenge,
    "OpenBookQA": _format_openbookqa,
    "PIQA": _format_piqa,
    "SocialIQA": _format_social_i_qa,
    "BoolQ": _format_boolq,
    "StrategyQA": _format_strategyqa,
    "GSM8K": _format_gsm8k,
    "DROP": _format_drop,
    "HellaSwag": _format_hellaswag,
    "MATH": _format_math,
    "LOCAL_JSON": _format_local_json,
    "WinoGrande": _format_winogrande,
    "CODAH": _format_codah,
    "COM2SENSE": _format_com2sense,
    "CREAK": _format_creak,
    "ProofWriter": _format_proofwriter,
    "CLadder": _format_cladder,
    "CausalNet": _format_causalnet,
    "COPA": _format_copa,
    "GPQA": _format_gpqa,
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
    
    return _load_local_json(config, formatter)


def _load_local_json(config: Dict, formatter: Callable) -> List[Dict]:
    """Load data from local JSON files."""
    json_path = config['hf_id']  # For local files, hf_id contains the file path
    num_samples = config['num_samples']
    
    print(f"Loading data from local JSON file...")
    print(f"  Path:    '{json_path}'")
    print(f"  Samples: {num_samples}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Take only the requested number of samples
        if num_samples and num_samples < len(data):
            data = data[:num_samples]
        
        formatted_data = [formatter(item) for item in data]
        print(f"Successfully loaded and formatted {len(formatted_data)} samples from local file.")
        return formatted_data
        
    except Exception as e:
        print(f"\n--- LOCAL JSON LOADING FAILED ---")
        print(f"An error occurred while trying to load '{json_path}'.")
        print(f"Error details: {e}")
        print("Please check:\n1. The file path is correct.\n2. The file exists and is readable.\n3. The JSON format is valid.")
        raise e