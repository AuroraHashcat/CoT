import os
from datasets import load_dataset

# 1. 设置镜像站（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 测试CROW数据集
try:
    ds = load_dataset("crow")
    print("Dataset keys:", list(ds.keys()))
    if "test" in ds:
        print("Test sample:", ds["test"][0])
    elif "validation" in ds:
        print("Validation sample:", ds["validation"][0])
    else:
        print("Train sample:", ds["train"][0])
except Exception as e:
    print(f"Error loading CROW dataset: {e}")
    # 尝试其他可能的名称
    try:
        ds = load_dataset("allenai/crow")
        print("Found with allenai prefix")
        print("Dataset keys:", list(ds.keys()))
        print("Sample:", ds[list(ds.keys())[0]][0])
    except Exception as e2:
        print(f"Also failed with allenai prefix: {e2}")