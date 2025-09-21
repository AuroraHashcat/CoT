import os
import logging
import json
import csv
import re
import ast
import random
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def load_corr2cause(split="train"):
    dataset_name = "causal-nlp/corr2cause"

    # 设置 huggingface 镜像（国内环境可能需要）
    # 方法1: 官方镜像
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 方法2: 清华镜像（有时可用）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 方法3: 也可以直接用 Gitee 上的 huggingface 镜像仓库（需要提前 git clone）
    # git clone https://gitee.com/mirrors/huggingface-datasets.git

    try:
        logging.info(f"Loading dataset {dataset_name}, split={split}")
        dataset = load_dataset(dataset_name, split=split)
        logging.info(f"Loaded dataset with {len(dataset)} rows.")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load from HuggingFace Hub: {e}")
        logging.info("尝试使用手动下载的本地数据集...")


def convert(input_csv, output_json):
    results = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("input", "")
            label = int(row.get("label", 0))
            answer = "true" if label == 1 else "false"
            results.append({
                "Question": question,
                "Answer": answer
            })
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def convert_gpqa(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        q = item['question']
        # 支持 Options 或 nOptions
        options_match = re.search(r'(Options|nOptions):\n((?:- .+\n)+)', q)
        if options_match:
            opts = [line.strip('- ').strip() for line in options_match.group(2).split('\n') if line.strip()]
        else:
            opts = []
        # 编号为 a/b/c/d...
        choice_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        choices = [f"{choice_labels[i]}. {opt}" for i, opt in enumerate(opts)]
        # answerkey 转为选项字母（类型和空格都做标准化）
        answer = str(item.get('answer', '')).strip()
        answer_label = None
        for i, opt in enumerate(opts):
            if str(opt).strip() == answer:
                answer_label = choice_labels[i]
                break
        # 去掉选项部分，保留主问题文本，并去除最后一句提示
        question_text = re.sub(r'(Options|Options):\n((?:- .+\n)+)', '', q).strip()
        question_text = re.sub(r'Please solve the question above, then store the final answer in \\boxed\{answer\}\.?', '', question_text, flags=re.IGNORECASE).strip()
        results.append({
            "question": question_text,
            "choices": choices,
            "answerKey": answer_label
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def convert_csv_to_gpqa(csv_path, json_path):
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question'].strip()
            options_raw = row['options']
            # 用正则提取每个选项
            # 支持格式如 'A ) ...' 'B ) ...' ...
            option_pattern = r"([A-E])\s*\)\s*([^'\"]+)"
            options_list = re.findall(option_pattern, options_raw)
            choices = [f"{label.lower()}. {text.strip()}" for label, text in options_list]
            answerKey = row['correct'].strip().lower()  # 转小写，和 choices 编号一致
            results.append({
                "question": question,
                "choices": choices,
                "answerKey": answerKey
            })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


import json
import random

def convert_causalnet_to_true_false(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        # 随机选一个选项编号
        choice_idx = random.choice([0, 1, 2])
        context = item.get('context', '')
        choice_text = item.get(f'choice_id{choice_idx}', '')
        # 拼接判断题内容
        question = f"{context}\n\nStatement: {choice_text}"
        label = item.get('label', None)
        answer = 'true' if label == choice_idx else 'false'
        results.append({
            "question": question,
            "answer": answer
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def download_and_save(dataset_name, split, output_path):
    print(f"正在下载 {dataset_name} ({split}) ...")
    ds = load_dataset(dataset_name, split=split)
    print(f"下载完成，共 {len(ds)} 条数据，正在保存到 {output_path} ...")
    # 转为 list[dict] 并保存为 json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([dict(x) for x in ds], f, indent=2, ensure_ascii=False)
    print(f"已保存到 {output_path}")



# import json

# input_file = 'cladder_test.json'
# output_file = 'cladder.json'

# with open(input_file, 'r', encoding='utf-8') as fin:
#     data_list = json.load(fin)

# new_list = []
# for data in data_list:
#     new_list.append({
#         "Question": data["prompt"],
#         "Answer": data["label"]
#     })

# with open(output_file, 'w', encoding='utf-8') as fout:
#     json.dump(new_list, fout, ensure_ascii=False, indent=2)

# 用法示例
#if __name__ == "__main__":
    # convert_gpqa("gpqa.json", "gpqa_mc.json")
    # convert_csv_to_gpqa("aqua.csv", "aqua.json")
    # convert_causalnet_to_true_false(
    #     "CausalProbe-2024/benchmarks/causalnet.json",
    #     "CausalProbe-2024/benchmarks/causalnet_true_false.json"
    # # )
    # download_and_save("causal-nlp/CLadder", "full_v1.5_default", "cladder_test.json")
    # download_and_save("commonsense_qa", "validation", "commonsenseqa_val.json")


def convert_commonsenseqa_json(input_path, output_path):
    label_map = {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e'}
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    results = []
    for item in data:
        # 拼接 choices
        choices = [
            f"{label_map[label]}. {text}"
            for label, text in zip(item['choices']['label'], item['choices']['text'])
        ]
        answer_key = label_map.get(item['answerKey'], item['answerKey'].lower())
        results.append({
            "question": item["question"],
            "choices": choices,
            "answerKey": answer_key
        })
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)


import json
import csv
import re

def convert_json_to_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results'] if 'results' in data else data

    with open(csv_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'input', 'label', 'num_variables', 'template', 'expected_answer'])
        for idx, item in enumerate(results):
            q = item['question']
            # premise: question到Statement:之前
            premise = q.split('Statement:')[0].strip()
            # hypothesis: Statement:之后的句子
            hypothesis = q.split('Statement:')[1].strip() if 'Statement:' in q else ''
            # 拼接hypothesis和模型输出
            pred = item.get('pred', None)
            answerkey = item.get('answerKey', '')
            # hypothesis后加 is {pred or answerkey}
            if pred is not None:
                input_str = f"{premise}\nHypothesis: {hypothesis} is {pred}"
                label = item.get('is_correct', True)
            else:
                input_str = f"{premise}\nHypothesis: {hypothesis} is {answerkey}"
                label = True
            # 其它字段固定
            writer.writerow([
                item.get('id', idx),
                input_str,
                label,
                2,
                'unknown',
                'None'
            ])


def convert_mc_json_to_csv(json_path, csv_path):
    import json
    import csv
    import re

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results'] if 'results' in data else data

    with open(csv_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'input', 'label', 'num_variables', 'template', 'expected_answer'])
        for idx, item in enumerate(results):
            q = item['question']
            # premise: question到\n\nChoices:之前，并把"Question:"替换为"Premise:"
            premise = q.split('\n\nChoices:')[0].strip().replace("Question:", "Premise:")
            # 选项列表
            choices = []
            choices_match = re.findall(r'([a-e])\. (.+)', q)
            if choices_match:
                choices = [text.strip() for _, text in choices_match]
            # pred选项或answerkey选项
            pred = item.get('pred', None)
            answerkey = item.get('answerKey', '')
            # 选项编号
            option_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            if pred is not None and isinstance(pred, str) and pred.lower() in option_map:
                hyp_idx = option_map[pred.lower()]
            elif answerkey and answerkey.lower() in option_map:
                hyp_idx = option_map[answerkey.lower()]
            else:
                hyp_idx = None
            hypothesis = choices[hyp_idx] if hyp_idx is not None and hyp_idx < len(choices) else ''
            # 拼接 input
            input_str = f"{premise}\nHypothesis: {hypothesis}"
            # label
            label = item.get('is_correct', True) if pred is not None else True
            # 其它字段固定
            writer.writerow([
                item.get('id', idx),
                input_str,
                label,
                2,
                'unknown',
                'None'
            ])

import pandas as pd
def download_codah(save_path="codah.csv", split="test"):
    # 下载 codah 数据集
    ds = load_dataset("codah", split=split)
    df = pd.DataFrame(ds)
    df.to_csv(save_path, index=False)
    print(f"已保存到 {save_path}")


import pandas as pd
import json

def test(csv_path, json_path):
    df = pd.read_csv(csv_path)
    questions = []
    for idx, row in df.iterrows():
        # 选项前加字母
        choices = []
        for i, choice in enumerate(eval(row['candidate_answers'])):
            choices.append(f"{chr(97+i)}. {choice}")
        # 正确答案字母
        answerKey = chr(97 + int(row['correct_answer_idx']))
        questions.append({
            "question": row['question_propmt'],
            "choices": choices,
            "answerKey": answerKey
        })
    # 保存为 json 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

# 用法示例
# test('your_file.csv', 'output.json')


# 用法示例
if __name__ == "__main__":
    # download_codah("codah.csv", "train")
    test('codah.csv', 'codah.json')
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/qwen2.5-7B_20250921_002544.json",
    #     "gpqa_qwen-7b.csv"
    # )
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/gpt-3.5_20250921_103220.json",
    #     "gpqa_gpt-3.5.csv"
    # )
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/gpt-5_20250921_103220.json",
    #     "gpqa_gpt-5.csv"
    # )
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/llama-3.1-8B_20250921_002545.json",
    #     "gpqa_llama-8b.csv"
    # )
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/o3-mini_20250921_103220.json",
    #     "gpqa_o3-mini.csv"
    # )
    # convert_mc_json_to_csv(
    #     "results/dataset_gpqa/baseline/qwen2.5-72B_20250921_002544.json",
    #     "gpqa_qwen-72b.csv"
    # )
