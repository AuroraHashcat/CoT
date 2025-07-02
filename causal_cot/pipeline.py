# causal_cot/pipeline.py
import re
import json
from .llm_handler import LLMHandler
from .knowledge_prober import KnowledgeProber
from . import prompts

class CausalCoTPipeline:
    def __init__(self, llm_handler: LLMHandler, prober: KnowledgeProber):
        self.llm = llm_handler
        self.prober = prober

    def _parse_cot(self, text: str) -> tuple[list[str], str]:
        steps = re.findall(r"Step \d+: (.*)", text)
        conclusion_match = re.search(r"Conclusion: .*?([A-E])", text, re.IGNORECASE)
        conclusion = conclusion_match.group(1).upper() if conclusion_match else "N/A"
        return steps, conclusion

    def _extract_entities_with_llm(self, sentence: str) -> dict:
        """Uses an LLM call to extract core entities from a sentence."""
        prompt = prompts.ENTITY_EXTRACTION_PROMPT.format(sentence=sentence)
        response = self.llm.query(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Warning: Failed to extract entities for sentence: '{sentence}'")
            return {"entity1": "", "entity2": ""}

    def run(self, question: str) -> dict:
        cot_prompt = prompts.COT_GENERATION_PROMPT.format(question_and_choices=question)
        initial_cot_raw = self.llm.query(cot_prompt)
        cot_queue, final_conclusion = self._parse_cot(initial_cot_raw)
        
        trace = {
            "initial_cot": cot_queue.copy(),
            "initial_conclusion": final_conclusion,
            "validated_steps": [],
            "interventions": 0,
            "probe_history": []
        }
        validated_facts = "Based on the initial question premise."

        i = 0
        while i < len(cot_queue):
            current_step = cot_queue[i].strip()
            
            # **新增步骤: LLM驱动的实体提取**
            entities = self._extract_entities_with_llm(current_step)
            
            # **修改步骤: 将实体传递给Prober**
            probe_result, kg_evidence = self.prober.probe(current_step, validated_facts, entities)
            
            trace["probe_history"].append({
                "step": current_step,
                "extracted_entities": entities, # <-- 记录提取的实体
                "kg_evidence": kg_evidence,
                "result": probe_result
            })

            if probe_result.get("should_include"):
                trace["validated_steps"].append(current_step)
                validated_facts += f"\n- {current_step}"
                i += 1
            else: # Reflection & Regeneration
                trace["interventions"] += 1
                reflection_prompt = prompts.REFLECTION_AND_REGENERATION_PROMPT.format(
                    question_and_choices=question,
                    validated_facts=validated_facts,
                    failed_step=current_step,
                    failure_reason=probe_result.get("explanation", "No reason provided."),
                    i=i + 1
                )
                regenerated_cot_raw = self.llm.query(reflection_prompt)
                new_steps, new_conclusion = self._parse_cot(regenerated_cot_raw)
                
                cot_queue = trace["validated_steps"] + new_steps
                final_conclusion = new_conclusion
                i = len(trace["validated_steps"])
                
                if not new_steps: break

        return {
            "final_answer_key": final_conclusion,
            "final_cot": trace["validated_steps"],
            "trace": trace
        }