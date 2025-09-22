#!/usr/bin/env bash

models=(
    "configs/model/qwen2.5-7B.json"
    "configs/model/llama-3.1-8B.json"
    "configs/model/claude-3.7-sonnet.json"
    "configs/model/claude-3.5-sonnet.json"
    "configs/model/qwen2.5-72B.json"
    "configs/model/llama-3.1-70B.json"
    "configs/model/deepseek-r1.json"
)
datasets=(
    "configs/dataset_hellaswag.json"
)


script="baseline_cot_inference.py"
work_dir="/home/wuchenxi/projects/Causal-Cot2"

run_session() {
    local session_name=$1
    local model_config=$2
    local dataset_config=$3

    cmds="cd $work_dir && conda activate causal;"
    cmds+="echo \"[$session_name] Running $model_config $dataset_config ...\";"
    cmds+="python $script --model_config $model_config --dataset_config $dataset_config;"
    cmds+="echo \"[$session_name] Finished.\";"
    screen -S "$session_name" -dm bash -c "$cmds"
}

for model_config in "${models[@]}"; do
    for dataset_config in "${datasets[@]}"; do
        session_name="$(basename "$model_config" .json)_$(basename "$dataset_config" .json)"
        echo "启动 $session_name ..."
        run_session "$session_name" "$model_config" "$dataset_config"
    done
done

echo "所有screen已启动，可用 screen -ls 查看，screen -r {name} 进入查看进度。"

    # "api_key_info": {
    #     "api_key_env": "DEEPSEEK_API_KEY",
    #     "api_url": "https://api.deepseek.com"
    # },