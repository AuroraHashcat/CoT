# causal_cot/pipeline.py
import re
import json
from typing import Dict, List, Tuple, Any
from .llm_handler import LLMHandler
from .knowledge_prober import Knowledge_Prober
from . import prompts

class CausalCoTPipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

class IntegratedCausalValidator:
    """
    Unified validator that combines causal analysis with step validation
    in a single coherent reasoning process - the major architectural improvement.
    """
    
    def __init__(self, llm_handler: LLMHandler, knowledge_prober: Knowledge_Prober):
        self.llm = llm_handler
        self.prober = knowledge_prober

    def integrated_step_analysis(
        self, 
        reasoning_step: str, 
        previous_validated_steps: list,
        original_question: str,
        verbose: bool = False
    ) -> dict:
        """
        Revolutionary unified analysis that combines causal structure discovery 
        with step validation in a single coherent LLM reasoning process.
        
        This eliminates the fragmented analyze → summarize → validate pipeline.
        """
        
        if verbose:
            print(f"Integrated Analysis: '{reasoning_step[:50]}...'")
        
        try:
            # Phase 1: Build causal knowledge graph
            probe_result = self.prober.probe_with_structure(
                sentence=reasoning_step,
                verbose=verbose
            )
            
            if "error" in probe_result:
                return self._handle_probe_error(probe_result, reasoning_step)
            
            # 新增：如果结构为 0，允许低置信度 accept
            structures_found = probe_result.get("structures_found", {})
            total_structures = sum(structures_found.values())
            if total_structures == 0:
                return {
                    "reasoning_step": reasoning_step,
                    "probe_result": probe_result,
                    "integrated_analysis": {
                        "validation_decision": True,
                        "confidence_level": "low",
                        "key_reasoning": "No causal structures found, but step is accepted with low confidence due to lack of evidence.",
                        "recommended_action": "accept",
                        "detailed_analysis": "No meaningful causal structures were detected in the knowledge graph. This may be due to limitations in commonsense knowledge bases or overly generic step phrasing. Accepting the step with low confidence, but recommend human review."
                    },
                    "is_valid": True,
                    "confidence": "low",
                    "detailed_reasoning": "No causal structures found, accepted with low confidence.",
                    "recommended_action": "accept"
                }
            
            # Phase 2: Single unified causal analysis + validation
            integrated_analysis = self._unified_causal_validation(
                reasoning_step=reasoning_step,
                probe_result=probe_result,
                previous_steps=previous_validated_steps,
                original_question=original_question
            )
            
            return {
                "reasoning_step": reasoning_step,
                "probe_result": probe_result,
                "integrated_analysis": integrated_analysis,
                "is_valid": integrated_analysis.get("validation_decision", False),
                "confidence": integrated_analysis.get("confidence_level", "medium"),
                "detailed_reasoning": integrated_analysis.get("detailed_analysis", ""),
                "recommended_action": integrated_analysis.get("recommended_action", "accept")
            }
            
        except Exception as e:
            return {
                "reasoning_step": reasoning_step,
                "error": f"Integrated analysis failed: {str(e)}",
                "is_valid": False,
                "confidence": "low"
            }

    def _unified_causal_validation(
        self,
        reasoning_step: str,
        probe_result: dict,
        previous_steps: list,
        original_question: str
    ) -> dict:
        """
        The core innovation: unified prompt that does causal analysis 
        AND step validation together in one coherent reasoning process.
        """
        
        # Prepare context information
        graph_stats = probe_result.get("graph_stats", {})
        structures_found = probe_result.get("structures_found", {})
        keywords = probe_result.get("keywords", [])
        
        # Build comprehensive context
        previous_context = self._build_previous_context(previous_steps)
        causal_context = self._build_causal_context(probe_result)
        
        # Single unified prompt - this is the breakthrough
        unified_prompt = prompts.INTEGRATED_CAUSAL_VALIDATION_PROMPT.format(
            original_question=original_question,
            previous_context=previous_context,
            reasoning_step=reasoning_step,
            keywords=', '.join(keywords),
            num_nodes=graph_stats.get('nodes', 0),
            num_edges=graph_stats.get('edges', 0),
            causal_context=causal_context
        )
        
        # Get unified LLM response
        llm_response = self.llm.query(unified_prompt)
        
        # Parse the structured response
        return self._parse_integrated_response(llm_response)

    def _build_previous_context(self, previous_steps: list) -> str:
        """Build context from previously validated steps."""
        if not previous_steps:
            return "This is the first step in the reasoning chain."
        
        context = "Previously validated reasoning steps:\n"
        for i, step in enumerate(previous_steps, 1):
            context += f"{i}. {step}\n"
        return context

    def _build_causal_context(self, probe_result: dict) -> str:
        """Build rich causal context from probe results."""
        structures_found = probe_result.get("structures_found", {})
        
        if not any(structures_found.values()):
            return "No significant causal structures were identified in the knowledge graph."
        
        context_parts = []
        
        # Add structure information
        total_structures = sum(structures_found.values())
        context_parts.append(f"Found {total_structures} causal structures:")
        
        for structure_type, count in structures_found.items():
            if count > 0:
                context_parts.append(f"  • {structure_type.replace('_', ' ').title()}: {count}")
        
        # Add causal analysis if available
        if "causal_analysis" in probe_result:
            context_parts.append(f"\nCausal Analysis Summary:")
            context_parts.append(probe_result["causal_analysis"])
        
        # Add confidence indicator
        confidence = probe_result.get("confidence", "medium")
        context_parts.append(f"\nCausal Analysis Confidence: {confidence.upper()}")
        
        return "\n".join(context_parts)

    def _parse_integrated_response(self, llm_response: str) -> dict:
        """Parse the structured LLM response into components."""
        
        # Default values
        result = {
            "validation_decision": False,
            "confidence_level": "medium", 
            "detailed_analysis": llm_response,
            "recommended_action": "regenerate_completely"
        }
        
        try:
            # Extract decision
            decision_match = re.search(r"DECISION:\s*(ACCEPT|REJECT)", llm_response, re.IGNORECASE)
            if decision_match:
                result["validation_decision"] = decision_match.group(1).upper() == "ACCEPT"
            
            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", llm_response, re.IGNORECASE)
            if confidence_match:
                result["confidence_level"] = confidence_match.group(1).lower()
            
            # Extract key reasoning
            reasoning_match = re.search(r"KEY_REASONING:\s*(.*?)(?=RECOMMENDED_ACTION|$)", llm_response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result["key_reasoning"] = reasoning_match.group(1).strip()
            
            # Extract recommended action
            action_match = re.search(r"RECOMMENDED_ACTION:\s*(.*?)(?=DETAILED_ANALYSIS|$)", llm_response, re.IGNORECASE | re.DOTALL)
            if action_match:
                result["recommended_action"] = action_match.group(1).strip()
            
            # Extract detailed analysis
            analysis_match = re.search(r"DETAILED_ANALYSIS:\s*(.*)", llm_response, re.IGNORECASE | re.DOTALL)
            if analysis_match:
                result["detailed_analysis"] = analysis_match.group(1).strip()
        
        except Exception as e:
            print(f"Warning: Could not fully parse LLM response: {e}")
        
        return result

    def _handle_probe_error(self, probe_result: dict, reasoning_step: str) -> dict:
        """Handle cases where causal probing failed."""
        return {
            "reasoning_step": reasoning_step,
            "probe_error": probe_result.get("error", "Unknown probe error"),
            "is_valid": True,  # Default to accepting if we can't analyze
            "confidence": "low",
            "detailed_reasoning": f"Causal analysis failed: {probe_result.get('error', 'Unknown error')}. Accepting step by default.",
            "recommended_action": "accept"
        }


class EnhancedCausalCoTPipeline:
    """
    Enhanced pipeline using integrated causal validation instead of 
    the fragmented analyze → summarize → validate approach.
    """
    
    def __init__(self, llm_handler: LLMHandler, prober: Knowledge_Prober, max_interventions: int = 5):
        self.llm = llm_handler
        self.prober = prober
        self.max_interventions = max_interventions
        
        # Initialize integrated validator - the key innovation
        self.integrated_validator = IntegratedCausalValidator(llm_handler, prober)

    def _parse_cot(self, text: str) -> Tuple[List[str], str]:
        """Parse Chain of Thought response to extract steps and conclusion."""
        try:
            steps = re.findall(r"Step \d+: (.*)", text, re.IGNORECASE)
            conclusion_match = re.search(r"Conclusion: .*?([A-E])", text, re.IGNORECASE)
            conclusion = conclusion_match.group(1).upper() if conclusion_match else "N/A"
            return steps, conclusion
        except Exception as e:
            print(f"Error parsing CoT: {e}")
            return [], "N/A"

    def run(self, question: str) -> dict:
        """
        Main pipeline execution using integrated causal analysis + validation.
        """
        try:
            # Generate initial Chain of Thought
            print("Generating initial Chain of Thought...")
            cot_prompt = prompts.COT_GENERATION_PROMPT.format(question_and_choices=question)
            initial_cot_raw = self.llm.query(cot_prompt)
            cot_queue, final_conclusion = self._parse_cot(initial_cot_raw)
            
            if not cot_queue:
                raise CausalCoTPipelineError("Failed to generate initial reasoning steps")
            
            # Initialize comprehensive tracking
            trace = {
                "initial_cot": cot_queue.copy(),
                "initial_conclusion": final_conclusion,
                "validated_steps": [],
                "interventions": 0,
                "integrated_analyses": [],  # New: store detailed analyses
                "errors": []
            }
            
            print(f"Processing {len(cot_queue)} reasoning steps...")
            
            # Process each step with integrated validation
            i = 0
            while i < len(cot_queue) and trace["interventions"] < self.max_interventions:
                current_step = cot_queue[i].strip()
                
                if not current_step:  # Skip empty steps
                    i += 1
                    continue
                
                print(f"\nIntegrated Analysis of Step {i+1}/{len(cot_queue)}: '{current_step[:50]}...'")
                
                # NEW: Single integrated analysis call instead of fragmented pipeline
                analysis_result = self.integrated_validator.integrated_step_analysis(
                    reasoning_step=current_step,
                    previous_validated_steps=trace["validated_steps"],
                    original_question=question,
                    verbose=True
                )
                
                trace["integrated_analyses"].append(analysis_result)
                
                if analysis_result["is_valid"]:
                    print(f"Step {i+1} ACCEPTED (confidence: {analysis_result['confidence']})")
                    trace["validated_steps"].append(current_step)
                    i += 1
                else:
                    print(f"Step {i+1} REJECTED")
                    
                    # Display rejection reasoning
                    key_reasoning = analysis_result.get('key_reasoning', 'No specific reason provided')
                    print(f"   Reason: {key_reasoning}")
                    
                    # Handle regeneration based on integrated analysis
                    action = analysis_result.get("recommended_action", "regenerate_completely")
                    if "regenerate" in action.lower():
                        trace["interventions"] += 1
                        print(f"Regenerating from Step {i+1} (intervention #{trace['interventions']})...")
                        
                        new_steps, new_conclusion = self._regenerate_from_step(
                            question, 
                            trace["validated_steps"], 
                            current_step, 
                            analysis_result,
                            i
                        )
                        
                        if new_steps:
                            cot_queue = trace["validated_steps"] + new_steps
                            final_conclusion = new_conclusion
                            i = len(trace["validated_steps"])
                            print(f"   Generated {len(new_steps)} new steps")
                        else:
                            print("   Regeneration failed - terminating")
                            break
                    else:
                        print("   No regeneration recommended - terminating")
                        break
            
            # Check termination conditions
            if trace["interventions"] >= self.max_interventions:
                print(f"Maximum interventions ({self.max_interventions}) reached")
                trace["errors"].append("Maximum interventions reached")
                
            print(f"\nFinal answer: {final_conclusion}")
            print(f"Pipeline stats: {len(trace['validated_steps'])} validated steps, {trace['interventions']} interventions")
            
            return {
                "final_answer_key": final_conclusion,
                "final_cot": trace["validated_steps"],
                "trace": trace,
                "success": len(trace["validated_steps"]) > 0 and final_conclusion != "N/A"
            }
            
        except Exception as e:
            error_msg = f"Pipeline execution error: {str(e)}"
            print(f"Error: {error_msg}")
            return {
                "final_answer_key": "N/A",
                "final_cot": [],
                "trace": {"errors": [error_msg]},
                "success": False
            }

    def _regenerate_from_step(self, question, validated_steps, failed_step, analysis_result, step_index):
        """Generate new reasoning based on integrated analysis feedback."""
        
        validated_context = "\n".join([f"{i+1}. {step}" for i, step in enumerate(validated_steps)])
        failure_reason = analysis_result.get("detailed_reasoning", "Step was rejected")
        
        regeneration_prompt = prompts.REFLECTION_AND_REGENERATION_PROMPT.format(
            question_and_choices=question,
            validated_facts=f"Previously validated reasoning:\n{validated_context}" if validated_context else "Based on the initial question premise.",
            failed_step=failed_step,
            failure_reason=failure_reason,
            step_index=step_index + 1,
            step_index_plus_1=step_index + 2
        )
        
        try:
            regenerated_response = self.llm.query(regeneration_prompt)
            new_steps, new_conclusion = self._parse_cot(regenerated_response)
            return new_steps, new_conclusion
        except Exception as e:
            print(f"Regeneration error: {e}")
            return [], "N/A"


# Legacy support - maintain compatibility with existing code
class CausalCoTPipeline(EnhancedCausalCoTPipeline):
    """
    Legacy wrapper for backward compatibility.
    New code should use EnhancedCausalCoTPipeline directly.
    """
    
    def __init__(self, llm_handler: LLMHandler, prober: Knowledge_Prober, max_interventions: int = 5):
        super().__init__(llm_handler, prober, max_interventions)
        print("Using legacy CausalCoTPipeline wrapper. Consider upgrading to EnhancedCausalCoTPipeline.")

    # Maintain old method name for compatibility
    def run_legacy(self, question: str) -> dict:
        """Legacy method name - redirects to new implementation."""
        return self.run(question)