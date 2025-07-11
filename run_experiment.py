# run_experiment.py
import argparse
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from common.config_utils import load_dataset_config
from data_processing.data_loader import load_data
from causal_cot.llm_handler import get_llm_handler, validate_config
from causal_cot.knowledge_prober import Knowledge_Prober
from causal_cot.pipeline import EnhancedCausalCoTPipeline
from evaluation.evaluator import Evaluator

def validate_setup(args):
    """Validate configuration files and setup before running experiments."""
    print("Validating setup...")
    
    # Validate model configuration
    model_validation = validate_config(args.model_config)
    if not model_validation["valid"]:
        print("Model configuration validation failed:")
        for error in model_validation["errors"]:
            print(f"  - {error}")
        return False
    
    if model_validation["warnings"]:
        print("Model configuration warnings:")
        for warning in model_validation["warnings"]:
            print(f"  - {warning}")
    
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

def process_single_question(idx, item, pipeline, total_questions):
    """Process a single question with detailed logging."""
    question_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Processing Question {idx + 1}/{total_questions} | ID: {item['id']}")
    print(f"{'='*80}")
    print(f"Question: {item['question']}")
    print(f"Ground Truth: {item['answerKey']}")
    
    # Run the enhanced pipeline
    result = pipeline.run(item['question'])
    
    # Add metadata
    result['id'] = item['id']
    result['question'] = item['question']
    result['ground_truth'] = item['answerKey']
    result['processing_time'] = time.time() - question_start_time
    
    # Display detailed trace information
    display_detailed_trace(result)
    
    return result

def display_detailed_trace(result):
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
    accuracy = result.get('final_answer_key') == result.get('ground_truth')
    print(f"Success: {success} | Accuracy: {'Correct' if accuracy else 'Incorrect'}")

def analyze_llm_performance(llm_handler, results):
    """Analyze and display LLM performance statistics."""
    stats = llm_handler.get_stats()
    
    print(f"\n{'='*80}")
    print("LLM PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total LLM Queries: {stats['total_queries']}")
    print(f"Failed Queries: {stats['failed_queries']}")
    print(f"Success Rate: {stats['success_percentage']}")
    
    # Calculate average queries per question
    if results:
        avg_queries = stats['total_queries'] / len(results)
        print(f"Average Queries per Question: {avg_queries:.1f}")
    
    # Intervention statistics
    total_interventions = sum(result.get('trace', {}).get('interventions', 0) for result in results)
    intervention_rate = (total_interventions / len(results)) if results else 0
    print(f"Total Self-Corrections: {total_interventions}")
    print(f"Average Interventions per Question: {intervention_rate:.2f}")

def save_detailed_results(all_results, metrics, args):
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
            "correct_answers": sum(1 for r in all_results if r.get('final_answer_key') == r.get('ground_truth')),
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
        
        # Limit questions if specified (useful for testing)
        if args.max_questions:
            dataset = dataset[:args.max_questions]
            print(f"Limited to {len(dataset)} questions for testing")
        
        # Initialize components
        print("Initializing enhanced pipeline components...")
        llm_handler = get_llm_handler(args.model_config)
        prober = Knowledge_Prober(llm_handler)
        pipeline = EnhancedCausalCoTPipeline(llm_handler, prober)
        
        # Display experiment information
        display_experiment_info(args, dataset_config, dataset)
        
        # Process all questions
        print(f"\nStarting enhanced processing of {len(dataset)} questions...")
        experiment_start_time = time.time()
        
        all_results = []
        
        # Use tqdm for progress tracking unless verbose mode
        if args.verbose:
            # Verbose mode: detailed logging for each question
            for idx, item in enumerate(dataset):
                result = process_single_question(idx, item, pipeline, len(dataset))
                all_results.append(result)
        else:
            # Normal mode: progress bar with minimal logging
            for idx, item in enumerate(tqdm(dataset, desc="Processing Questions")):
                question_start_time = time.time()
                result = pipeline.run(item['question'])
                
                # Add metadata
                result['id'] = item['id']
                result['question'] = item['question']
                result['ground_truth'] = item['answerKey']
                result['processing_time'] = time.time() - question_start_time
                
                all_results.append(result)
                
                # Brief progress update
                if (idx + 1) % 10 == 0:
                    correct = sum(1 for r in all_results if r.get('final_answer_key') == r.get('ground_truth'))
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
        
        # Save comprehensive results
        output_path = save_detailed_results(all_results, metrics, args)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
    
