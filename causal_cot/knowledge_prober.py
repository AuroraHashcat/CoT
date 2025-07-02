
### **2. `causal_cot/knowledge_prober.py` (重大升级)**

# causal_cot/knowledge_prober.py
import requests
import json
import re
import random
from .llm_handler import LLMHandler
from . import prompts

class KnowledgeProber:
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        self.api_url = "http://api.conceptnet.io"

    def _get_concept_uri(self, text: str, lang: str = 'en') -> str:
        """Converts text to a ConceptNet URI."""
        return f"/c/{lang}/{text.lower().replace(' ', '_')}"

    def _query_uri(self, uri: str) -> list:
        """Queries ConceptNet for all edges connected to a given URI."""
        try:
            response = requests.get(f"{self.api_url}{uri}", timeout=5)
            response.raise_for_status()
            return response.json().get('edges', [])
        except requests.exceptions.RequestException:
            return []

    def _perform_random_walk(self, start_entity: str, walk_length: int = 2, walks_per_entity: int = 5) -> set:
        """
        Performs random walks starting from an entity to build a local graph.
        Returns a set of formatted edge strings.
        """
        if not start_entity:
            return set()

        start_uri = self._get_concept_uri(start_entity)
        visited_edges = set()
        
        for _ in range(walks_per_entity):
            current_uri = start_uri
            for _ in range(walk_length):
                edges = self._query_uri(current_uri)
                if not edges:
                    break
                
                chosen_edge = random.choice(edges)
                
                start_node = chosen_edge.get('start', {})
                end_node = chosen_edge.get('end', {})
                rel = chosen_edge.get('rel', {})
                
                start_label = start_node.get('label', 'N/A')
                end_label = end_node.get('label', 'N/A')
                rel_label = rel.get('label', 'N/A')
                
                formatted_edge = f"({start_label}) -[:{rel_label}]-> ({end_label})"
                visited_edges.add(formatted_edge)

                # Decide the next node for the walk
                if start_node.get('@id') == current_uri:
                    current_uri = end_node.get('@id')
                else:
                    current_uri = start_node.get('@id')
                
                if not current_uri:
                    break
                    
        return visited_edges

    def probe(self, step_text: str, validated_facts: str, entities: dict) -> tuple[dict, list[str]]:
        """Probes a single reasoning step using random walks and returns the judgment and evidence."""
        entity1 = entities.get("entity1")
        entity2 = entities.get("entity2")

        # Build a richer local graph with random walks from both entities
        local_graph1 = self._perform_random_walk(entity1)
        local_graph2 = self._perform_random_walk(entity2)
        combined_graph = sorted(list(local_graph1.union(local_graph2)))
        
        if not combined_graph:
            combined_graph = ["No knowledge graph evidence could be constructed."]

        # Run the enhanced verification prompt
        verification_prompt = prompts.EXPERT_CAUSAL_VERIFICATION_PROMPT.format(
            step_to_verify=step_text,
            local_knowledge_graph="\n".join(combined_graph)
        )
        verification_raw = self.llm.query(verification_prompt)
        
        try:
            json_str_match = re.search(r'```json\n(.*?)\n```', verification_raw, re.DOTALL)
            result_json = json_str_match.group(1) if json_str_match else verification_raw
            verification_result = json.loads(result_json)
        except json.JSONDecodeError:
            verification_result = {
                "should_include": False,
                "causal_structure": "Error",
                "explanation": "Verifier LLM returned a non-JSON response.",
            }
        
        return verification_result, combined_graph