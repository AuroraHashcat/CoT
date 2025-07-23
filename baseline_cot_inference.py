import os
import argparse
import json
import re
from tqdm import tqdm
import time
from common.config_utils import load_dataset_config
from data_processing.data_loader import load_data
from causal_cot.llm_handler import get_llm_handler

def get_cot_prompt(item):
    # 自动判断是否有choices字段
    q = item['question']
    if 'choices' in item and item['choices']:
        q += '\nChoices:\n' + '\n'.join(item['choices'])
    prompt = (
        "You are a meticulous logical reasoner.\n"
        "Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
        f"{q}\n"
        "\nOutput Format:\n"
        "Step 1: [First causal deduction or analysis]\n"
        "Step 2: [Second deduction, building upon Step 1]\n"
        "...\n"
        "Step N: [Final deduction that directly leads to the answer]\n"
        "===FINAL_ANSWER_START===\n"
        "Conclusion: [Your final answer, based strictly on the above causal reasoning]\n"
        "===FINAL_ANSWER_END===\n"
    )
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
        llm = get_llm_handler(args.model_config)
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
    output_dir = os.path.join("results", dataset_name, "baseline")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.json")
    if args.output:
        output_path = args.output

    results = []
    correct = 0

    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        start_time = time.time()
        try:
            prompt = get_cot_prompt(item)
            output = llm.query(prompt)
            steps, pred = extract_cot_and_answer(output)
        except Exception as e:
            print(f"[ERROR] LLM failed for sample id {item.get('id', '?')}: {e}")
            steps, pred = [], None
            output = str(e)
        gold = get_answer_key(item)
        # 宽松匹配：只要gold在pred中即可判为正确
        is_correct = (gold is not None and pred is not None and str(gold) in str(pred))
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