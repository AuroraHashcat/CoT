# run_experiment.py
import argparse
import json
import os
from tqdm import tqdm
from common.config_utils import load_dataset_config
from data_processing.data_loader import load_data
from causal_cot.llm_handler import get_llm_handler
from causal_cot.knowledge_prober import KnowledgeProber
from causal_cot.pipeline import CausalCoTPipeline
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Run Causal-CoT Framework with Flexible Configurations")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config JSON (e.g., configs/model_api.json).")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config JSON (e.g., configs/dataset_csqa.json).")
    args = parser.parse_args()

    # 1. Load Configurations
    print(f"Loading model config from: {args.model_config}")
    print(f"Loading dataset config from: {args.dataset_config}")
    dataset_config = load_dataset_config(args.dataset_config)
    
    # 2. Setup Components
    print("Initializing components...")
    llm_handler = get_llm_handler(args.model_config)
    prober = KnowledgeProber(llm_handler)
    pipeline = CausalCoTPipeline(llm_handler, prober)

    # 3. Load Data
    print(f"Loading {dataset_config['num_samples']} samples from {dataset_config['dataset_name']}...")
    dataset = load_data(dataset_config)

    # 4. Run Pipeline and Log Verbose Output
    print("\n" + "="*80)
    print("Running Causal-CoT pipeline on dataset...")
    all_results = []
    for item in tqdm(dataset, desc=f"Processing {dataset_config['dataset_name']}"):
        result = pipeline.run(item['question'])
        
        # Add metadata for evaluation
        result['id'] = item['id']
        result['question'] = item['question']
        result['ground_truth'] = item['answerKey']
        all_results.append(result)
        
        # Verbose logging to console
        tqdm.write("\n" + f"----- [ Sample ID: {result['id']} ] -----")
        tqdm.write(f"Question: {item['question']}")
        tqdm.write("\n--- Causal Verification Trace ---")
        for i, probe_entry in enumerate(result['trace']['probe_history']):
            tqdm.write(f"\nStep {i+1}: {probe_entry['step']}")
            tqdm.write("  [KG Evidence]:")
            for line in probe_entry.get('kg_evidence', []):
                tqdm.write(f"    {line}")
            probe_res = probe_entry.get('result', {})
            decision = "VALID" if probe_res.get('should_include') else "INVALID"
            tqdm.write(f"  [Causal Judgment]: {decision}")
            tqdm.write(f"    Structure: {probe_res.get('causal_structure', 'N/A')}")
            tqdm.write(f"    Reasoning: {probe_res.get('explanation', 'N/A')}")
        
        if result['trace']['interventions'] > 0:
            tqdm.write("\n[INFO] Self-correction was triggered.")
        tqdm.write(f"\nFinal Answer Key: {result['final_answer_key']} (Ground Truth: {result['ground_truth']})")
        tqdm.write("----- [ End of Sample ] -----")

    # 5. Evaluate and Save
    print("\n" + "="*80)
    print("Evaluating results...")
    evaluator = Evaluator(all_results)
    metrics = evaluator.run_evaluation()

    print("\n--- Final Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))
    
    model_name = os.path.basename(args.model_config).replace('.json', '')
    dataset_name = os.path.basename(args.dataset_config).replace('.json', '').replace('dataset_', '')
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "results": all_results}, f, indent=4, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_path}")

if __name__ == "__main__":
    main()