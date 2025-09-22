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

# import json
# import re

# def add_pred_and_is_correct(json_path, output_path):
#     import json
#     import re

#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     results = data['results'] if 'results' in data else data
#     count = 0

#     for item in results:
#         cot_output = item.get('cot_output', '')
#         match = re.search(r'===FINAL_ANSWER_START===\s*([A-Ea-e0-9]+)\s*===FINAL_ANSWER_END===', cot_output, re.DOTALL)
#         pred = match.group(1).strip() if match else None
#         item['pred'] = pred
#         answer_key = item.get('answerKey', '').strip()
#         is_correct = (pred is not None) and (pred.lower() == answer_key.lower())
#         if is_correct:
#             count += 1
#         item['is_correct'] = is_correct

#     total = len(results)
#     acc = count / total if total > 0 else 0.0
#     acc_str = f"{acc*100:.2f}%"

#     # 写入 metrics
#     if "metrics" in data and "overall_summary" in data["metrics"]:
#         data["metrics"]["overall_summary"]["final_accuracy"] = acc_str
#         data["metrics"]["overall_summary"]["total_questions_evaluated"] = total

#     # print(f"总共 {total} 条数据，正确 {count} 条，准确率 {acc_str}")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# # 用法示例
# # add_pred_and_is_correct('claude-3.5-sonnet_20250922_021218.json', 'claude-3.5-sonnet_20250922_021218_with_pred.json')、

# # 用法示例
# if __name__ == "__main__":
#     add_pred_and_is_correct('results/dataset_causalnet/baseline/claude-3.7-sonnet_20250922_021217.json', 'results/dataset_causalnet/baseline/claude-3.7-sonnet_20250922_021217_new.json')
#     add_pred_and_is_correct('results/dataset_codah/baseline/claude-3.5-sonnet_20250922_020911.json', 'results/dataset_codah/baseline/claude-3.5-sonnet_20250922_020911_new.json')
#     add_pred_and_is_correct('results/dataset_codah/baseline/claude-3.7-sonnet_20250922_020911.json', 'results/dataset_codah/baseline/claude-3.7-sonnet_20250922_020911_new.json')

import os
import json
import re

def add_pred_and_is_correct(json_path, output_path):
    import json
    import re

    def normalize_tf(val):
        # 支持多种 true/false 表达
        true_set = {'true', 't', 'yes', 'y', '1', 'correct'}
        false_set = {'false', 'f', 'no', 'n', '0', 'incorrect'}
        val = str(val).strip().lower()
        if val in true_set:
            return 'true'
        elif val in false_set:
            return 'false'
        return val

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results'] if 'results' in data else data
    count = 0

    for item in results:
        cot_output = item.get('cot_output', '')
        # 选择题答案
        match = re.search(r'===FINAL_ANSWER_START===\s*([A-Ea-e0-9]+)\s*===FINAL_ANSWER_END===', cot_output, re.DOTALL)
        pred = match.group(1).strip() if match else None
        # 判断题答案
        if not pred:
            match_tf = re.search(r'===FINAL_ANSWER_START===\s*(true|false|True|False|1|0|yes|no|correct|incorrect)\s*===FINAL_ANSWER_END===', cot_output, re.IGNORECASE)
            if match_tf:
                pred = normalize_tf(match_tf.group(1))
        item['pred'] = pred

        answer_key = str(item.get('answerKey', '')).strip()
        # 判断题宽松匹配
        if pred in ['true', 'false'] or normalize_tf(answer_key) in ['true', 'false']:
            is_correct = normalize_tf(pred) == normalize_tf(answer_key)
        else:
            is_correct = (pred is not None) and (pred.lower() == answer_key.lower())
        if is_correct:
            count += 1
        item['is_correct'] = is_correct

    total = len(results)
    acc = count / total if total > 0 else 0.0
    acc_str = f"{acc*100:.2f}%"

    # 写入 metrics
    if "metrics" in data and "overall_summary" in data["metrics"]:
        data["metrics"]["overall_summary"]["final_accuracy"] = acc_str
        data["metrics"]["overall_summary"]["total_questions_evaluated"] = total

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def batch_process_results():
    base_dir = "results"
    sub_dirs = ["dataset_strategyqa"]
    models = ["claude-3.7-sonnet", "claude-3.5-sonnet"]

    for sub in sub_dirs:
        sub_path = os.path.join(base_dir, sub, "baseline")
        if not os.path.exists(sub_path):
            continue
        for fname in os.listdir(sub_path):
            for model in models:
                if fname.startswith(model) and fname.endswith(".json"):
                    in_file = os.path.join(sub_path, fname)
                    out_file = os.path.join(sub_path, fname.replace(".json", "_new.json"))
                    print(f"处理 {in_file} -> {out_file}")
                    add_pred_and_is_correct(in_file, out_file)


import json

def convert_hellaswag(input_path, output_path):
    label_map = ['a', 'b', 'c', 'd', 'e']
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    results = []
    for item in data:
        question = item.get('ctx', '')
        choices = [f"{label_map[i]}. {ending}" for i, ending in enumerate(item.get('endings', []))]
        # label为字符串数字，转为字母
        label_idx = int(item.get('label', 0))
        answerKey = label_map[label_idx]
        results.append({
            "question": question,
            "choices": choices,
            "answerKey": answerKey
        })
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

# 用法示例
# convert_hellaswag('hellaswag.json', 'hellaswag_formatted.json')

if __name__ == "__main__":
    convert_hellaswag('hellaswag.json', 'hellaswag2.json')