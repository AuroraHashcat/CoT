import os
import argparse
import json
import re
from tqdm import tqdm
import time
from datetime import datetime
from common.config_utils import load_dataset_config
from common.answer_matcher import check_answer_correctness
from data_processing.data_loader import load_data
from causal_cot.llm_handler import create_llm_handler, load_llm_config

# 数据集类型映射
DATASET_TYPE_MAP = {
    "hellaswag": "multiple_choice",
    "commonsense_qa": "multiple_choice", 
    "arc_challenge": "multiple_choice",
    "boolq": "multiple_choice",
    "copa": "multiple_choice",
    "openbookqa": "multiple_choice",
    "piqa": "multiple_choice",
    "siqa": "multiple_choice",
    "winogrande": "multiple_choice",
    "strategyqa": "multiple_choice",
    "creak": "multiple_choice",
    "codah": "multiple_choice",
    "gsm8k": "fill_in_blank",
    "drop": "fill_in_blank", 
    "math": "fill_in_blank",
    "causalnet": "fill_in_blank",
    "cladder": "fill_in_blank",
    "com2sense": "fill_in_blank",
    "proofwriter": "fill_in_blank"
}

def get_dataset_type(dataset_name):
    """根据数据集名称获取题型"""
    for key in DATASET_TYPE_MAP:
        if key in dataset_name.lower():
            return DATASET_TYPE_MAP[key]
    return "fill_in_blank"  # 默认为填空题

def get_cot_prompt(item, dataset_type=None):
    q = item['question']
    
    # 根据数据集类型设置prompt context和指令
    if dataset_type == "multiple_choice":
        task_context = "This is a multiple choice question. You need to select the best option from the given choices."
        
        # 检查是否有choices字段，如果没有则从question中提取选项
        if 'choices' in item and item['choices']:
            q += '\nChoices:\n' + '\n'.join(item['choices'])
        
        conclusion_instruction = "Conclusion: [Your final answer - MUST be the option number only (0, 1, 2, or 3)]"
        additional_instruction = "\nIMPORTANT: For multiple choice questions, your final answer MUST be only the option number (0, 1, 2, or 3). Do not include any additional text in your conclusion."
        
    else:  # fill_in_blank or other types
        task_context = "This is a fill-in-the-blank or open-ended question. Provide a specific and accurate answer."
        conclusion_instruction = "Conclusion: [Your final answer, based strictly on the above causal reasoning]"
        additional_instruction = ""
    
    prompt = (
        "You are a meticulous logical reasoner.\n"
        f"{task_context}\n"
        "Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
        f"{q}\n"
        "\nOutput Format:\n"
        "Step 1: [First causal deduction or analysis]\n"
        "Step 2: [Second deduction, building upon Step 1]\n"
        "...\n"
        "Step N: [Final deduction that directly leads to the answer]\n"
        "===FINAL_ANSWER_START===\n"
        f"{conclusion_instruction}\n"
        "===FINAL_ANSWER_END===\n"
    )
    
    prompt += additional_instruction
    
    return prompt

def extract_cot_and_answer(text):
    import re
    print("=== LLM OUTPUT ===")
    print(text)
    steps = re.findall(r"Step \d+: (.*)", text)
    # 优先提取 \box{...}
    box_match = re.search(r"\\box\{(.*?)\}", text)
    if box_match:
        conclusion = box_match.group(1).strip()
    else:
        # 提取 Conclusion: 后的全部内容
        conclusion_match = re.search(r"===FINAL_ANSWER_START===.*?Conclusion:\s*(.*?)\s*===FINAL_ANSWER_END===", text, re.DOTALL)
        if not conclusion_match:
            conclusion_match = re.search(r"Conclusion:\s*(.*)", text)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else None
    print(f"Extracted steps: {steps}")
    print(f"Extracted conclusion: {conclusion}")
    return steps, conclusion

def get_answer_key(item):
    return item.get('answerKey') or item.get('Answer') or item.get('answer')

def convert_config_format(old_config):
    """Convert old config format to new llm_handler format."""
    model_info = old_config.get("model_info", {})
    api_key_info = old_config.get("api_key_info", {})
    params = old_config.get("params", {})
    
    # Get API key from environment
    api_key_env = api_key_info.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None
    
    new_config = {
        "type": "api",
        "provider": model_info.get("provider", "openai"),
        "model": model_info.get("name", ""),
        "api_key": api_key,
        "base_url": api_key_info.get("api_url"),
        "temperature": params.get("temperature", 0.7),
        "max_tokens": params.get("max_output_tokens", 2000),
        "max_retries": 3,
        "retry_delay": 1.0
    }
    
    return new_config

def main():
    parser = argparse.ArgumentParser(description="Baseline CoT inference (no causal validation)")
    parser.add_argument('--model_config', type=str, required=True, help="Path to model config JSON.")
    parser.add_argument('--dataset_config', type=str, required=True, help="Path to dataset config JSON.")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    print(f"[INFO] Loading model config from: {args.model_config}")
    print(f"[INFO] Loading dataset config from: {args.dataset_config}")
    try:
        dataset_config = load_dataset_config(args.dataset_config)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset config: {e}")
        return

    print("[INFO] Initializing model handler...")
    try:
        with open(args.model_config, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        model_config = convert_config_format(old_config)
        llm = create_llm_handler(model_config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model handler: {e}")
        return

    print(f"[INFO] Loading data...")
    try:
        data = load_data(dataset_config)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    model_name = os.path.basename(args.model_config).replace('.json', '')
    dataset_name = os.path.basename(args.dataset_config).replace('.json', '')
    dataset_type = get_dataset_type(dataset_name)  # 获取数据集类型
    print(f"[INFO] Dataset type detected: {dataset_type}")
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", dataset_name, "baseline")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
    if args.output:
        output_path = args.output

    results = []
    correct = 0

    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        start_time = time.time()
        try:
            prompt = get_cot_prompt(item, dataset_type)  # 传递数据集类型
            output = llm.query(prompt)
            steps, pred = extract_cot_and_answer(output)
        except Exception as e:
            print(f"[ERROR] LLM failed for sample id {item.get('id', '?')}: {e}")
            steps, pred = [], None
            output = str(e)
        gold = get_answer_key(item)
        # 根据数据集类型使用不同的匹配策略
        is_correct = check_answer_correctness(gold, pred, dataset_type)
        if is_correct:
            correct += 1
        elapsed = time.time() - start_time
        results.append({
            'id': item.get('id', ''),
            'question': item['question'],
            'answerKey': gold,
            'cot_output': output,
            'cot_steps': steps,
            'pred': pred,
            'is_correct': is_correct,
            'elapsed_time': elapsed
        })
        # 每做完一个题立即写入同一个文件（覆盖写入）
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save results after sample {idx+1}: {e}")

    acc = correct / len(data) if data else 0.0
    print(f'Accuracy: {acc:.2%}')

    metrics = {
        "overall_summary": {
            "total_questions_evaluated": len(results),
            "final_accuracy": f"{acc:.2%}",
            "total_successful_pipelines": correct
        },
        "detailed_metrics": {
            "intervention_summary": {
                "questions_with_intervention": "0/{}".format(len(results)),
                "accuracy_with_intervention": 0.0,
                "accuracy_without_intervention": acc
            },
            "causal_score_analysis": {},
            "reflection_effectiveness_analysis": {}
        }
    }
    correction_rate = 0.0

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "correction_rate": correction_rate,
                "results": results
            }, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Full results saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

if __name__ == '__main__':
    main()