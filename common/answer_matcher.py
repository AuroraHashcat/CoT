# common/answer_matcher.py
"""
通用答案匹配模块，支持不同数据集类型的答案格式
"""
import re

def check_answer_correctness(gold, pred, dataset_type):
    """根据数据集类型检查答案正确性"""
    if gold is None or pred is None:
        return False
    
    if dataset_type == "multiple_choice":
        # 选择题：支持字母和数字格式的匹配
        gold_str = str(gold).strip()
        pred_str = str(pred).strip()
        
        # 创建字母到数字的映射
        letter_to_number = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}
        number_to_letter = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        
        # 直接匹配
        if gold_str == pred_str:
            return True
        
        # 字母到数字转换匹配
        if gold_str in letter_to_number and letter_to_number[gold_str] == pred_str:
            return True
        
        # 数字到字母转换匹配
        if gold_str in number_to_letter and number_to_letter[gold_str] == pred_str:
            return True
        
        # 从预测中提取数字和字母进行匹配
        pred_numbers = re.findall(r'\b[0-3]\b', pred_str)
        pred_letters = re.findall(r'\b[A-D]\b', pred_str)
        
        # 检查数字匹配
        if gold_str in letter_to_number:
            return letter_to_number[gold_str] in pred_numbers
        elif gold_str in pred_numbers:
            return True
        
        # 检查字母匹配
        if gold_str in number_to_letter:
            return number_to_letter[gold_str] in pred_letters
        elif gold_str in pred_letters:
            return True
        
        return False
    else:
        # 填空题：宽松匹配
        return str(gold).lower() in str(pred).lower()

# 为了向后兼容，提供一个更简单的接口
def is_answer_correct(ground_truth, prediction, dataset_name):
    """
    简化的答案匹配接口
    
    Args:
        ground_truth: 标准答案
        prediction: 模型预测
        dataset_name: 数据集名称
    
    Returns:
        bool: 是否匹配
    """
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
    
    dataset_type = get_dataset_type(dataset_name)
    return check_answer_correctness(ground_truth, prediction, dataset_type)
