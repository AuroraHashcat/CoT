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
        # 选择题：支持多种表达形式（大小写字母 abcd/ABCD，数字 0123）
        gold_str = str(gold).strip().lower()
        pred_str = str(pred).strip().lower()
        # 多种选项表达
        a_variants = {'a', 'A', '0'}
        b_variants = {'b', 'B', '1'}
        c_variants = {'c', 'C', '2'}
        d_variants = {'d', 'D', '3'}
        option_map = {
            'a': a_variants,
            'b': b_variants,
            'c': c_variants,
            'd': d_variants,
            '0': a_variants,
            '1': b_variants,
            '2': c_variants,
            '3': d_variants
        }
        # gold_str 属于哪个选项
        for key, variants in option_map.items():
            if gold_str in variants:
                return any(v in pred_str for v in variants)
        # 直接匹配
        return gold_str == pred_str
    elif dataset_type == "true_or_false":
        # 判断题：支持多种表达形式
        true_variants = {'true', 't', 'yes', 'y', '1', 'correct','TRUE'}
        false_variants = {'false', 'f', 'no', 'n', '0', 'incorrect','FALSE'}
        
        gold_str = str(gold).strip().lower()
        pred_str = str(pred).strip().lower()
        
        if gold_str in true_variants:
            return pred_str in true_variants
        elif gold_str in false_variants:
            return pred_str in false_variants
        else:
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
        "math": "fill_in_blank",
        "causalnet": "multiple_choice",
        "cladder": "true_or_false"
    }
    
    def get_dataset_type(dataset_name):
        """根据数据集名称获取题型"""
        for key in DATASET_TYPE_MAP:
            if key in dataset_name.lower():
                return DATASET_TYPE_MAP[key]
        return "fill_in_blank"  # 默认为填空题
    
    dataset_type = get_dataset_type(dataset_name)
    return check_answer_correctness(ground_truth, prediction, dataset_type)
