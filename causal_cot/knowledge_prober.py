# causal_cot/knowledge_prober.py
import requests
import json
import re
from .llm_handler import LLMHandler
from . import prompts

class KnowledgeProber:
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        self.api_url = "http://api.conceptnet.io"

    def _get_concept_uri(self, text: str, lang: str = 'en') -> str:
        """Converts text to a ConceptNet URI."""
        return f"/c/{lang}/{text.lower().replace(' ', '_')}"

    def _query_conceptnet(self, entity1: str, entity2: str) -> list[str]:
        """Uses ConceptNet's /query endpoint to find relationships."""
        if not entity1 or not entity2:
            return ["Entities for KG lookup were not extracted."]
        
        uri1 = self._get_concept_uri(entity1)
        uri2 = self._get_concept_uri(entity2)
        params = {'node': uri1, 'other': uri2, 'limit': 10}
        
        try:
            response = requests.get(f"{self.api_url}/query", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            edges = [f"({e.get('start', {}).get('label', 'N/A')})-[:{e.get('rel', {}).get('label', 'N/A')}, w={e.get('weight', 0)}]->({e.get('end', {}).get('label', 'N/A')})" for e in data.get('edges', [])]
            
            return edges if edges else ["No direct relations found between the entities in ConceptNet."]
        except requests.exceptions.RequestException as e:
            return [f"ConceptNet API Error: {e}"]

    def _extract_claim_entities(self, step_text: str) -> tuple[str, str]:
        """A simple heuristic to extract subject and object for KG query."""
        # This can be replaced by a more sophisticated NER LLM call if needed.
        words = re.findall(r'\b[a-zA-Z]+\b', step_text)
        # Avoid common verbs or trivial words
        stop_words = {'is', 'a', 'an', 'the', 'to', 'of', 'in', 'it', 'for', 'as'}
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return (filtered_words[0], filtered_words[-1]) if len(filtered_words) >= 2 else ("", "")

    def probe(self, step_text: str, validated_facts: str) -> tuple[dict, list[str]]:
        """Probes a single reasoning step and returns the judgment and evidence."""
        subject, obj = self._extract_claim_entities(step_text)
        kg_evidence = self._query_conceptnet(subject, obj)

        verification_prompt = prompts.EXPERT_CAUSAL_VERIFICATION_PROMPT.format(
            step_to_verify=step_text,
            claim_subject=subject,
            claim_object=obj,
            validated_facts=validated_facts,
            kg_evidence="\n".join(kg_evidence)
        )
        verification_raw = self.llm.query(verification_prompt)
        
        try:
            json_str_match = re.search(r'```json\n(.*?)\n```', verification_raw, re.DOTALL)
            if json_str_match:
                result_json = json_str_match.group(1)
            else:
                result_json = verification_raw
            verification_result = json.loads(result_json)
        except json.JSONDecodeError:
            verification_result = {
                "should_include": False,
                "causal_structure": "Error",
                "explanation": "Verifier LLM returned a non-JSON response.",
                "collider_suggestion": "N/A"
            }
        
        return verification_result, kg_evidence