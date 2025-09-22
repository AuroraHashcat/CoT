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
from causal_cot.llm_handler import create_llm_handler

# 数据集类型映射
DATASET_TYPE_MAP = {
    "math": "fill_in_blank",
    "causalnet": "multiple_choice",
    "cladder": "true_or_false",
    "commonsenseqa": "multiple_choice",
    "corr2cause": "true_or_false",
    "gpqa":"multiple_choice",
    "aqua":"multiple_choice",
    "strategyqa":"true_or_false",
    "codah":"multiple_choice",
    "copa":"multiple_choice"
}

def get_dataset_type(dataset_name):
    """根据数据集名称获取题型"""
    for key in DATASET_TYPE_MAP:
        if key in dataset_name.lower():
            return DATASET_TYPE_MAP[key]
    return "fill_in_blank"  # 默认为填空题

def get_cot_prompt(item, dataset_type=None):
    q = item['question']
    if dataset_type == "multiple_choice":
        if 'choices' in item and item['choices']:
            q += '\nChoices:\n' + '\n'.join(item['choices'])
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following MULTIPLE CHOICE question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "IMPORTANT: This is a multiple choice question. Your final answer MUST be exactly one of the choice letters (A, B, C, or D) or numbers (0, 1, 2, or 3), nothing else.\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [Single letter/number representing your choice: A, B, C, D or 0, 1, 2, 3]\n"
            "===FINAL_ANSWER_END===\n"
        )
    elif dataset_type == "fill_in_blank":
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [Your final answer, only a single number or expression, nothing else. based strictly on the above causal reasoning]\n"
            "===FINAL_ANSWER_END===\n"
        )
    elif dataset_type == "true_or_false":
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following TRUE or FALSE question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "IMPORTANT: This is a TRUE or FALSE question. Your final answer MUST be exactly true' or'false', nothing else.\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [true or false]\n"
            "===FINAL_ANSWER_END===\n"
        )
    return prompt


def extract_cot_and_answer(text: str) -> tuple[list[str], any]:
    import re
    steps = re.findall(r"Step \d+: (.*)", text)
    # 先尝试 Conclusion: 提取
    conclusion = None
    match = re.search(r"Conclusion:\s*(.*?)(?:\n|===FINAL_ANSWER_END===|$)", text)
    if match:
        conclusion = match.group(1).strip()
        conclusion = re.sub(r'[\*#]+', '', conclusion).strip()
    else:
        # 支持没有 Conclusion: 直接答案的情况
        match2 = re.search(r"===FINAL_ANSWER_START===\s*([A-Ea-e0-9]+)\s*===FINAL_ANSWER_END===", text, re.DOTALL)
        if match2:
            conclusion = match2.group(1).strip()
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
    total_elapsed = 0.0
    total_tokens = 0

    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        start_time = time.time()
        try:
            prompt = get_cot_prompt(item, dataset_type)  # 传递数据集类型
            output = llm.query(prompt)
            steps, pred = extract_cot_and_answer(output)
            # 获取token用量，假设output为字符串或dict
            if isinstance(output, dict) and "tokens_used" in output:
                tokens_used = output["tokens_used"]
                cot_output = output.get("text", str(output))
            else:
                tokens_used = None
                cot_output = output if isinstance(output, str) else str(output)
        except Exception as e:
            print(f"[ERROR] LLM failed for sample id {item.get('id', '?')}: {e}")
            steps, pred = [], None
            cot_output = str(e)
            tokens_used = None
        gold = get_answer_key(item)
        is_correct = check_answer_correctness(gold, pred, dataset_type)
        if is_correct:
            correct += 1
        elapsed = time.time() - start_time
        total_elapsed += elapsed
        if tokens_used is not None:
            total_tokens += tokens_used
        results.append({
            'id': item.get('id', ''),
            'question': item['question'],
            'answerKey': gold,
            'cot_output': cot_output,
            'cot_steps': steps,
            'pred': pred,
            'is_correct': is_correct,
            'elapsed_time': elapsed,
            'tokens_used': tokens_used
        })
        # 每做完一个题立即写入同一个文件（覆盖写入）
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save results after sample {idx+1}: {e}")

    acc = correct / len(data) if data else 0.0
    avg_elapsed_time = total_elapsed / len(results) if results else 0.0
    avg_tokens_used = total_tokens / len(results) if results and total_tokens > 0 else None
    print(f'Accuracy: {acc:.2%}')
    print(f'Average elapsed time: {avg_elapsed_time:.2f}s')
    if avg_tokens_used is not None:
        print(f'Average tokens used: {avg_tokens_used:.2f}')

    metrics = {
        "overall_summary": {
            "total_questions_evaluated": len(results),
            "final_accuracy": f"{acc:.2%}",
            "total_successful_pipelines": correct,
            "avg_elapsed_time": avg_elapsed_time,
            "avg_tokens_used": avg_tokens_used
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