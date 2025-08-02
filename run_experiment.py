# run_experiment.py
import argparse
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from common.config_utils import load_dataset_config
from common.answer_matcher import is_answer_correct
from data_processing.data_loader import load_data
from causal_cot.llm_handler import create_llm_handler, load_llm_config
from causal_cot.knowledge_prober import Knowledge_Prober
from causal_cot.pipeline import EnhancedCausalCoTPipeline
from evaluation.evaluator import Evaluator

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

def format_question_with_choices(item, dataset_type):
    """根据数据集类型格式化问题"""
    question = item['question']
    
    if dataset_type == "multiple_choice":
        # 检查是否有choices字段
        if 'choices' in item and item['choices']:
            question += '\nChoices:\n' + '\n'.join(item['choices'])
    
    return question

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

def validate_setup(args):
    """Validate configuration files and setup before running experiments."""
    print("Validating setup...")
    
    # Check if model config exists
    if not os.path.exists(args.model_config):
        print(f"Model configuration file not found: {args.model_config}")
        return False
    
    # Check if dataset config exists
    if not os.path.exists(args.dataset_config):
        print(f"Dataset configuration file not found: {args.dataset_config}")
        return False
    
    print("Setup validation passed")
    return True

def display_experiment_info(args, dataset_config, dataset):
    """Display experiment information."""
    print("\n" + "="*80)
    print("CAUSAL CHAIN-OF-THOUGHT EXPERIMENT")
    print("="*80)
    print(f"Model Config: {args.model_config}")
    print(f"Dataset Config: {args.dataset_config}")
    print(f"Questions to Process: {len(dataset)}")
    print(f"Dataset Name: {dataset_config.get('name', 'Unknown')}")
    print(f"Using: Enhanced Integrated Causal Validation")
    print("="*80)

def process_single_question(idx, item, pipeline, total_questions, dataset_type, dataset_name):
    """Process a single question with detailed logging."""
    question_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Processing Question {idx + 1}/{total_questions} | ID: {item['id']}")
    print(f"{'='*80}")
    
    # 格式化问题（包含选择项）
    formatted_question = format_question_with_choices(item, dataset_type)
    print(f"Question: {formatted_question}")
    print(f"Ground Truth: {item['answerKey']}")
    
    # Run the enhanced pipeline
    result = pipeline.run(formatted_question)
    
    # Add metadata
    result['id'] = item['id']
    result['question'] = item['question']
    result['ground_truth'] = item['answerKey']
    result['processing_time'] = time.time() - question_start_time
    
    # Display detailed trace information
    display_detailed_trace(result, dataset_name)
    
    return result

def display_detailed_trace(result, dataset_name="unknown"):
    """Display comprehensive trace information."""
    trace = result.get('trace', {})
    
    print(f"\n{'-'*60} PROCESSING TRACE {'-'*60}")
    
    # Initial reasoning
    initial_cot = trace.get('initial_cot', [])
    print(f"Initial Reasoning ({len(initial_cot)} steps):")
    for i, step in enumerate(initial_cot, 1):
        print(f"  {i}. {step}")
    
    print(f"Initial Conclusion: {trace.get('initial_conclusion', 'N/A')}")
    
    # Integrated analyses
    integrated_analyses = trace.get('integrated_analyses', [])
    print(f"\nIntegrated Causal Analyses ({len(integrated_analyses)} performed):")
    
    for i, analysis in enumerate(integrated_analyses, 1):
        step = analysis.get('reasoning_step', 'Unknown step')
        is_valid = analysis.get('is_valid', False)
        confidence = analysis.get('confidence', 'unknown')
        
        status = "ACCEPTED" if is_valid else "REJECTED"
        print(f"\n  Step {i}: {step[:60]}{'...' if len(step) > 60 else ''}")
        print(f"  └─ {status} (confidence: {confidence})")
        
        # Show detailed reasoning for rejected steps
        if not is_valid:
            key_reasoning = analysis.get('key_reasoning', 'No reason provided')
            print(f"     Rejection reason: {key_reasoning}")
        
        # Show probe results summary
        probe_result = analysis.get('probe_result', {})
        if 'graph_stats' in probe_result:
            stats = probe_result['graph_stats']
            structures = probe_result.get('structures_found', {})
            total_structures = sum(structures.values()) if structures else 0
            print(f"     Graph: {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges, {total_structures} structures")
            print(f"     Stage: {stats.get('exploration_stage', 'unknown')}")
    
    # Final results
    interventions = trace.get('interventions', 0)
    validated_steps = trace.get('validated_steps', [])
    
    print(f"\nSelf-Corrections: {interventions}")
    print(f"Final Validated Steps ({len(validated_steps)}):")
    for i, step in enumerate(validated_steps, 1):
        print(f"  {i}. {step}")
    
    print(f"\nFinal Answer: {result.get('final_answer_key', 'N/A')} (Ground Truth: {result.get('ground_truth', 'N/A')})")
    
    # Performance metrics
    processing_time = result.get('processing_time', 0)
    print(f"Processing Time: {processing_time:.2f} seconds")
    
    # Success indicator
    success = result.get('success', False)
    gold = result.get('ground_truth')
    pred = result.get('final_answer_key')
    accuracy = is_answer_correct(gold, pred, dataset_name)
    print(f"Success: {success} | Accuracy: {'Correct' if accuracy else 'Incorrect'}")

def analyze_llm_performance(llm_handler, results):
    """Analyze and display LLM performance statistics."""
    stats = llm_handler.get_stats()
    
    print(f"\n{'='*80}")
    print("LLM PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total LLM Queries: {stats['total_queries']}")
    print(f"Failed Queries: {stats['failed_queries']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    
    # Calculate average queries per question
    if results:
        avg_queries = stats['total_queries'] / len(results)
        print(f"Average Queries per Question: {avg_queries:.1f}")
    
    # Intervention statistics
    total_interventions = sum(result.get('trace', {}).get('interventions', 0) for result in results)
    intervention_rate = (total_interventions / len(results)) if results else 0
    print(f"Total Self-Corrections: {total_interventions}")
    print(f"Average Interventions per Question: {intervention_rate:.2f}")

def generate_output_path(args):
    """Generate output path for incremental saving."""
    model_name = os.path.basename(args.model_config).replace('.json', '')
    dataset_name = os.path.basename(args.dataset_config).replace('.json', '').replace('dataset_', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, f"{dataset_name}_{model_name}_{timestamp}_results.json")

def save_incremental_results(output_path, all_results, args, dataset_name, partial_metrics=None):
    """Save results incrementally after each question."""
    try:
        # Prepare current comprehensive results
        comprehensive_results = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_config": args.model_config,
                "dataset_config": args.dataset_config,
                "total_questions_processed": len(all_results),
                "pipeline_version": "EnhancedCausalCoTPipeline_v2.1",
                "status": "completed" if partial_metrics is not None else "in_progress"
            },
            "metrics": partial_metrics or {
                "accuracy": sum(1 for r in all_results if is_answer_correct(r.get('ground_truth'), r.get('final_answer_key'), dataset_name)) / len(all_results) if all_results else 0.0,
                "partial_evaluation": True
            },
            "results": all_results,
            "summary_statistics": {
                "total_questions_processed": len(all_results),
                "successful_completions": sum(1 for r in all_results if r.get('success', False)),
                "correct_answers": sum(1 for r in all_results if is_answer_correct(r.get('ground_truth'), r.get('final_answer_key'), dataset_name)),
                "total_interventions": sum(r.get('trace', {}).get('interventions', 0) for r in all_results),
                "average_processing_time": sum(r.get('processing_time', 0) for r in all_results) / len(all_results) if all_results else 0
            }
        }
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save incremental results: {e}")
        return False

def save_detailed_results(all_results, metrics, args, dataset_name):
    """Save comprehensive results with enhanced metadata."""
    # Create results directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    model_name = os.path.basename(args.model_config).replace('.json', '')
    dataset_name = os.path.basename(args.dataset_config).replace('.json', '').replace('dataset_', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_{timestamp}_results.json")
    
    # Prepare comprehensive results
    comprehensive_results = {
        "experiment_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_config": args.model_config,
            "dataset_config": args.dataset_config,
            "total_questions": len(all_results),
            "pipeline_version": "EnhancedCausalCoTPipeline_v2.1"
        },
        "metrics": metrics,
        "results": all_results,
        "summary_statistics": {
            "total_questions": len(all_results),
            "successful_completions": sum(1 for r in all_results if r.get('success', False)),
            "correct_answers": sum(1 for r in all_results if is_answer_correct(r.get('ground_truth'), r.get('final_answer_key'), dataset_name)),
            "total_interventions": sum(r.get('trace', {}).get('interventions', 0) for r in all_results),
            "average_processing_time": sum(r.get('processing_time', 0) for r in all_results) / len(all_results) if all_results else 0
        }
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=4, ensure_ascii=False)
    
    print(f"Complete results saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced Causal-CoT Framework with Integrated Validation")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config JSON.")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit number of questions to process (for testing).")
    args = parser.parse_args()

    # Validate setup
    if not validate_setup(args):
        return 1

    try:
        # Load configurations and data
        print("Loading configurations and data...")
        dataset_config = load_dataset_config(args.dataset_config)
        dataset = load_data(dataset_config)
        
        # 检测数据集类型
        dataset_name = os.path.basename(args.dataset_config).replace('.json', '').replace('dataset_', '')
        dataset_type = get_dataset_type(dataset_name)
        print(f"[INFO] Dataset type detected: {dataset_type}")
        
        # Limit questions if specified (useful for testing)
        if args.max_questions:
            dataset = dataset[:args.max_questions]
            print(f"Limited to {len(dataset)} questions for testing")
        
        # Initialize components
        print("Initializing enhanced pipeline components...")
        with open(args.model_config, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        llm_config = convert_config_format(old_config)
        llm_handler = create_llm_handler(llm_config)
        prober = Knowledge_Prober(llm_handler)
        pipeline = EnhancedCausalCoTPipeline(llm_handler, prober, dataset_type=dataset_type)  # 传递数据集类型
        
        # Display experiment information
        display_experiment_info(args, dataset_config, dataset)
        
        # Process all questions with incremental saving
        print(f"\nStarting enhanced processing of {len(dataset)} questions...")
        experiment_start_time = time.time()
        
        all_results = []
        
        # Generate output path for incremental saving (only once)
        output_path = generate_output_path(args)
        print(f"Results will be saved incrementally to: {output_path}")
        
        # Use tqdm for progress tracking unless verbose mode
        if args.verbose:
            # Verbose mode: detailed logging for each question
            for idx, item in enumerate(dataset):
                result = process_single_question(idx, item, pipeline, len(dataset), dataset_type, dataset_name)
                all_results.append(result)
                
                # Save incrementally after each question
                save_incremental_results(output_path, all_results, args, dataset_name)
        else:
            # Normal mode: progress bar with minimal logging
            for idx, item in enumerate(tqdm(dataset, desc="Processing Questions")):
                question_start_time = time.time()
                
                # 格式化问题（包含选择项）
                formatted_question = format_question_with_choices(item, dataset_type)
                result = pipeline.run(formatted_question)
                
                # Add metadata
                result['id'] = item['id']
                result['question'] = item['question']
                result['ground_truth'] = item['answerKey']
                result['processing_time'] = time.time() - question_start_time
                
                all_results.append(result)
                
                # Save incrementally after each question
                save_incremental_results(output_path, all_results, args, dataset_name)
                
                # Brief progress update
                if (idx + 1) % 10 == 0:
                    correct = sum(1 for r in all_results if is_answer_correct(r.get('ground_truth'), r.get('final_answer_key'), dataset_name))
                    accuracy = correct / len(all_results)
                    print(f"  Progress: {idx + 1}/{len(dataset)} | Accuracy: {accuracy:.1%}")

        total_experiment_time = time.time() - experiment_start_time
        
        # Analyze LLM performance
        analyze_llm_performance(llm_handler, all_results)
        
        # Run evaluation
        print(f"\n{'='*80}")
        print("EVALUATING RESULTS...")
        print(f"{'='*80}")
        
        evaluator = Evaluator(all_results)
        metrics = evaluator.run_evaluation()
        
        print("\nFINAL EVALUATION METRICS:")
        print(json.dumps(metrics, indent=2))
        
        # Performance summary
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total Experiment Time: {total_experiment_time:.2f} seconds")
        print(f"Average Time per Question: {total_experiment_time / len(dataset):.2f} seconds")
        print(f"Overall Accuracy: {metrics.get('accuracy', 0):.1%}")
        print(f"Success Rate: {sum(1 for r in all_results if r.get('success', False)) / len(all_results):.1%}")
        
        # Save final comprehensive results with full evaluation
        save_incremental_results(output_path, all_results, args, dataset_name, metrics)
        
        print(f"\nExperiment completed successfully!")
        print(f"Final results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

