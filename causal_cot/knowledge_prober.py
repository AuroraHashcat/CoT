<<<<<<< HEAD
### **2. `causal_cot/knowledge_prober.py` **

=======
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
# causal_cot/knowledge_prober.py
import requests
import random
import networkx as nx
import json
<<<<<<< HEAD
from typing import List, Tuple, Dict, Any, Optional, Union
import re

class Knowledge_Prober:
    def __init__(self, llm_handler: Any):
        self.llm = llm_handler
        self.api_url = "http://api.conceptnet.io"
        self.step_cache: Dict[str, Tuple[List[str], nx.DiGraph, Dict[str, Any]]] = {}

    def get_concept_uri(self, keyword: str, lang: str = "en") -> str:
        """
        Generates a ConceptNet URI for a given keyword and language.

        ConceptNet uses a specific URI format (e.g., /c/en/apple).
        This function normalizes the keyword by lowercasing it and replacing
        spaces with underscores to match ConceptNet's convention.

        Args:
            keyword (str): The word or phrase to convert into a ConceptNet URI.
            lang (str): The two-letter language code (e.g., "en" for English).

        Returns:
            str: The formatted ConceptNet URI.
        """
        normalized_keyword = keyword.lower().replace(" ", "_")
        return f"/c/{lang}/{normalized_keyword}"

    def get_edges_for_concept(self, concept_uri: str, limit: int = 100) -> List[dict]:
        """
        Fetches edges (relationships) for a given ConceptNet concept URI using the ConceptNet 5 API.

        Args:
            concept_uri (str): The ConceptNet URI of the concept to query.
            limit (int): The maximum number of edges to retrieve from the API for this concept.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary represents an edge
                        (relationship) connected to the concept. Returns an empty list on error.
        """
        url = f"http://api.conceptnet.io{concept_uri}"
        params = {"limit": limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('edges', [])
        except Exception as e:
            print(f"Error fetching edges for {concept_uri}: {e}")
            return []

    def _perform_random_walk(
        self,
        keywords: List[str],
        max_depth: int = 3,
        max_edges_per_hop: int = 5,
        language: str = "en",
        api_limit_per_request: int = 100,
        min_weight: float = 1.0,
        forward_only: bool = False,
        relation_type_weights: Dict[str, float] = None,
        avoid_backtracking: bool = True,
        verbose: bool = True
    ) -> nx.DiGraph:
        """
        Performs a random walk on ConceptNet to build a network of concepts
        related to a set of initial keywords. This version includes improvements
        for weighted edge selection and more controlled exploration.

        Args:
            keywords (List[str]):
                A list of strings, representing the starting concepts for the walk.
            max_depth (int):
                The maximum number of "hops" (levels of relationships) from the
                initial keywords to explore.
            max_edges_per_hop (int):
                From each concept visited, this is the maximum number of *randomly selected*
                relevant edges to follow to new concepts. This introduces the
                "random" element and controls the breadth of the exploration.
            language (str):
                The two-letter language code to filter concepts and relationships by
                (e.g., "en" for English).
            api_limit_per_request (int):
                The maximum number of edges to request from the ConceptNet API
                for a single concept lookup.
            min_weight (float):
                The minimum weight (strength/confidence score) an edge must have
                to be considered relevant and included in the network.
            forward_only (bool):
                If `True`, the walk will only follow relationships where the
                `current_concept_uri` is the 'start' node of the ConceptNet edge.
                If `False`, it will follow relationships where the `current_concept_uri`
                is either the 'start' or the 'end' node.
            relation_type_weights (Dict[str, float], optional):
                A dictionary to assign custom weight multipliers to specific relation types.
                For example, `{"IsA": 2.0, "AtLocation": 0.5}` would make "IsA" relations
                twice as likely to be chosen and "AtLocation" half as likely (relative to others).
                Relations not in this dict default to a multiplier of 1.0. Defaults to None (no custom weights).
            avoid_backtracking (bool):
                If `True`, the walk will not immediately return to the concept it just came from.
                This helps prevent trivial back-and-forth loops. Defaults to True.
            verbose (bool):
                If `True`, prints progress messages during the walk. Defaults to True.

        Returns:
            nx.DiGraph:
                A NetworkX directed graph object representing the discovered network.
                - Nodes: ConceptNet URIs. Each node has `'name'` and `'type'` attributes.
                - Edges: Represent relationships. Each edge has `'relation'` and `'weight'` attributes.
        """
        if relation_type_weights is None:
            relation_type_weights = {}

        graph = nx.DiGraph()
        visited_concepts = set()
        # Queue stores (concept_uri, current_depth, previous_concept_uri_in_path)
        queue: List[Tuple[str, int, Union[str, None]]] = []

        # --- Initialization ---
        for keyword in keywords:
            concept_uri = self.get_concept_uri(keyword, language)
            if concept_uri not in visited_concepts:
                graph.add_node(concept_uri, name=keyword, type='keyword')
                queue.append((concept_uri, 0, None))
                visited_concepts.add(concept_uri)
                if verbose:
                    print(f"Starting with keyword: {keyword} ({concept_uri})")
            elif verbose:
                print(f"Skipping duplicate keyword: {keyword}")

        # --- BFS-like Traversal Loop ---
        head = 0
        while head < len(queue):
            current_concept_uri, current_depth, prev_concept_uri = queue[head]
            head += 1

            if verbose:
                print(f"\nExploring: {current_concept_uri} (Depth: {current_depth})")

            # Stop if max depth is reached for this path
            if current_depth >= max_depth:
                if verbose:
                    print(f"  Max depth ({max_depth}) reached for {current_concept_uri}. Skipping further exploration.")
                continue

            # Fetch all raw edges connected to the current concept
            raw_edges = self.get_edges_for_concept(current_concept_uri, api_limit_per_request)
            if not raw_edges:
                if verbose:
                    print(f"  No edges found for {current_concept_uri} from API.")
                continue

            # --- Filter Edges and Prepare for Weighted Selection ---
            eligible_edges = []
            for edge in raw_edges:
                # Validate basic structure of the edge response
                if not all(k in edge and isinstance(edge[k], dict) for k in ['start', 'end', 'rel']):
                    continue
                if not all(k in edge['start'] and k in edge['end'] for k in ['@id', 'label', 'language']):
                    continue
                if '@id' not in edge['rel']:
                    continue

                start_uri = edge['start']['@id']
                end_uri = edge['end']['@id']
                relation_full_uri = edge['rel']['@id']
                relation_name = relation_full_uri.split('/')[-1]
                weight = edge.get('weight', 0)

                # Filter by language
                if edge['start']['language'] != language or edge['end']['language'] != language:
                    continue

                # Filter by minimum weight
                if weight < min_weight:
                    continue

                # Apply 'forward_only' filter
                if forward_only and start_uri != current_concept_uri:
                    continue

                # Determine the target URI for this edge
                target_uri = end_uri if start_uri == current_concept_uri else start_uri

                # Avoid immediate backtracking
                if avoid_backtracking and target_uri == prev_concept_uri:
                    if verbose:
                        print(f"  Skipping backtracking edge: {current_concept_uri} -> {target_uri}")
                    continue

                # Skip self-loops
                if target_uri == current_concept_uri:
                    continue

                # Apply relation type weight multiplier
                adjusted_weight = weight * relation_type_weights.get(relation_name, 1.0)
                if adjusted_weight <= 0:
                     continue

                eligible_edges.append({
                    'edge_data': edge,
                    'start_uri': start_uri,
                    'end_uri': end_uri,
                    'relation_name': relation_name,
                    'original_weight': weight,
                    'adjusted_weight': adjusted_weight,
                    'target_uri': target_uri
                })

            if not eligible_edges:
                if verbose:
                    print(f"  No eligible edges found for {current_concept_uri} after filtering.")
                continue

            # --- Weighted Random Selection of Edges ---
            num_to_select = min(max_edges_per_hop, len(eligible_edges))
            selection_weights = [item['adjusted_weight'] for item in eligible_edges]

            # Use random.choices for weighted sampling
            try:
                selected_items = random.choices(eligible_edges, weights=selection_weights, k=num_to_select)
            except ValueError:
                if verbose:
                    print(f"  All eligible edges have zero adjusted weight for {current_concept_uri}. Falling back to uniform selection.")
                selected_items = random.sample(eligible_edges, num_to_select)

            # --- Add Selected Edges and New Concepts to Graph/Queue ---
            for item in selected_items:
                edge = item['edge_data']
                start_uri = item['start_uri']
                end_uri = item['end_uri']
                relation_name = item['relation_name']
                original_weight = item['original_weight']
                target_uri = item['target_uri']

                # Add nodes to graph
                start_label = edge['start'].get('label', start_uri)
                end_label = edge['end'].get('label', end_uri)

                if start_uri not in graph:
                    graph.add_node(start_uri, name=start_label, type='concept')
                if end_uri not in graph:
                    graph.add_node(end_uri, name=end_label, type='concept')

                # Add edge to graph
                if not graph.has_edge(start_uri, end_uri):
                    graph.add_edge(start_uri, end_uri, relation=relation_name, weight=original_weight)
                    if verbose:
                        print(f"  Added: {start_label} --({relation_name}, w={original_weight:.2f})--> {end_label}")

                # Enqueue target concept for next depth level
                if target_uri not in visited_concepts:
                    queue.append((target_uri, current_depth + 1, current_concept_uri))
                    visited_concepts.add(target_uri)
                    if verbose:
                        print(f"    Queued {graph.nodes[target_uri]['name']} for exploration at depth {current_depth + 1}")
                elif verbose:
                    print(f"    {graph.nodes[target_uri]['name']} already visited/queued.")

        return graph

    def find_causal_structures(self, graph: nx.DiGraph) -> Dict[str, List[Tuple[Tuple[str, str, str], Dict[str, Any]]]]:
        """
        Identifies potential causal chain, fork, and collider structures in a directed graph.
        Returns URIs for the nodes to ensure uniqueness and correct edge retrieval.

        Args:
            graph (nx.DiGraph): The networkx directed graph.

        Returns:
            Dict[str, List[Tuple[Tuple[str, str, str], Dict[str, Any]]]]: A dictionary where keys are 'chains', 'forks',
            and 'colliders', and values are lists of ((A_uri, B_uri, C_uri), edge_details) tuples
            representing the structure. edge_details is a dictionary containing relevant edge data
            for the structure.
        """
        chains = []
        forks = []
        colliders = []

        # Find chains: A -> B -> C
        for node_A_uri in graph.nodes():
            for node_B_uri in graph.successors(node_A_uri):
                edge_ab = graph.get_edge_data(node_A_uri, node_B_uri)
                for node_C_uri in graph.successors(node_B_uri):
                    edge_bc = graph.get_edge_data(node_B_uri, node_C_uri)
                    if node_A_uri != node_C_uri:
                        chains.append(((node_A_uri, node_B_uri, node_C_uri), {'edge_ab': edge_ab, 'edge_bc': edge_bc}))

        # Find forks: B <- A -> C (A is common cause)
        for node_A_uri in graph.nodes():
            successors_of_A = list(graph.successors(node_A_uri))
            if len(successors_of_A) >= 2:
                for j in range(len(successors_of_A)):
                    node_B_uri = successors_of_A[j]
                    edge_ab = graph.get_edge_data(node_A_uri, node_B_uri)
                    for k in range(j + 1, len(successors_of_A)):
                        node_C_uri = successors_of_A[k]
                        edge_ac = graph.get_edge_data(node_A_uri, node_C_uri)
                        forks.append(((node_A_uri, node_B_uri, node_C_uri), {'edge_ab': edge_ab, 'edge_ac': edge_ac}))

        # Find colliders: A -> C <- B (C is common effect)
        for node_C_uri in graph.nodes():
            predecessors_of_C = list(graph.predecessors(node_C_uri))
            if len(predecessors_of_C) >= 2:
                for j in range(len(predecessors_of_C)):
                    node_A_uri = predecessors_of_C[j]
                    edge_ac = graph.get_edge_data(node_A_uri, node_C_uri)
                    for k in range(j + 1, len(predecessors_of_C)):
                        node_B_uri = predecessors_of_C[k]
                        edge_bc = graph.get_edge_data(node_B_uri, node_C_uri)
                        colliders.append(((node_A_uri, node_C_uri, node_B_uri), {'edge_ac': edge_ac, 'edge_bc': edge_bc}))

        return {
            'chains': chains,
            'forks': forks,
            'colliders': colliders
        }

    def analyze_causal_structure_with_llm(
        self,
        structure_type: str,
        node_names_tuple: Tuple[str, str, str],
        edge_details: Dict[str, Any],
    ) -> str:
        """
        Generates a prompt for an LLM based on the causal structure and its details,
        then calls the LLM and returns its response.
        """
        node1, node2, node3 = node_names_tuple
        prompt = ""

        if structure_type == 'chain':
            edge_ab = edge_details.get('edge_ab', {})
            edge_bc = edge_details.get('edge_bc', {})
            prompt = f"""You are an expert in causal inference and network analysis.
            Analyze the following proposed causal chain and explain its implications based on Judea Pearl's causality framework.

            Causal Chain:
              - A: {node1}
              - B: {node2}
              - C: {node3}

            Relationship A to B: '{edge_ab.get('relation')}' (ConceptNet weight: {edge_ab.get('weight', 0):.2f}).
            Relationship B to C: '{edge_bc.get('relation')}' (ConceptNet weight: {edge_bc.get('weight', 0):.2f}).

            Provide:
            1. A classification of the causal strength and type (e.g., 'Direct Causal Path', 'Mediated Association').
            2. A brief explanation (3-5 sentences) of why this is a chain, focusing on how B mediates the effect of A on C, and mention d-separation (A ⊥ C | B).
            3. Suggest whether this specific chain is plausible in a real-world context and why, considering the given relations and their ConceptNet weights.
            """
        elif structure_type == 'fork':
            node_A_cause, node_B_effect1, node_C_effect2 = node1, node2, node3
            edge_ab = edge_details.get('edge_ab', {})
            edge_ac = edge_details.get('edge_ac', {})

            prompt = f"""You are an expert in causal inference and network analysis.
            Analyze the following proposed causal fork (common cause structure) and explain its implications based on Judea Pearl's causality framework.

            Causal Fork:
            - Common Cause (A): {node_A_cause}
            - Effect 1 (B): {node_B_effect1}
            - Effect 2 (C): {node_C_effect2}

            Relationship A to B: '{edge_ab.get('relation')}' (ConceptNet weight: {edge_ab.get('weight', 0):.2f}).
            Relationship A to C: '{edge_ac.get('relation')}' (ConceptNet weight: {edge_ac.get('weight', 0):.2f}).

            Provide:
            1. A classification of the causal strength and type (e.g., 'Strong Common Cause', 'Confounding Structure').
            2. A brief explanation (3-5 sentences) of why this is a fork, focusing on how A confounds the relationship between B and C. Explain d-separation (B ⊥ C | A).
            3. Suggest whether this specific fork is plausible in a real-world context and why, considering the given relations and their ConceptNet weights.
            """
        elif structure_type == 'collider':
            node_A_cause1, node_C_effect, node_B_cause2 = node1, node2, node3
            edge_ac = edge_details.get('edge_ac', {})
            edge_bc = edge_details.get('edge_bc', {})

            prompt = f"""You are an expert in causal inference and network analysis.
            Analyze the following proposed causal collider structure and explain its implications based on Judea Pearl's causality framework.

            Causal Collider:
            - Cause 1 (A): {node_A_cause1}
            - Common Effect (C): {node_C_effect}
            - Cause 2 (B): {node_B_cause2}

            Relationship A to C: '{edge_ac.get('relation')}' (ConceptNet weight: {edge_ac.get('weight', 0):.2f}).
            Relationship B to C: '{edge_bc.get('relation')}' (ConceptNet weight: {edge_bc.get('weight', 0):.2f}).

            Provide:
            1. A classification of the causal strength and type (e.g., 'Potential Collider Bias', 'Independent Causes, Common Effect').
            2. A brief explanation (3-5 sentences) of why this is a collider, focusing on how conditioning on C (or its descendants) can induce spurious correlations between A and B. Explain collider bias and A ⊥ B (unconditionally).
            3. Suggest whether this specific collider is plausible in a real-world context and why, emphasizing potential biases if conditioned upon, considering the given relations and their ConceptNet weights.
            """
        else:
            return "Invalid structure type for LLM analysis."

        print(f"\n--- Sending to LLM for {structure_type} analysis ---")
        print(f"Prompt:\n{prompt}")
        llm_response = self.llm.query(prompt)
        return llm_response

    def extract_keywords_from_sentence(self, sentence: str) -> List[str]:
        """
        Use LLM to extract key concepts or entities from a natural language sentence.
        """
        keyword_prompt = (
            f'Given the following sentence:\n\n'
            f'"{sentence}"\n\n'
            'Extract 2-5 key concepts that are central to the meaning and reasoning of the sentence. '
            'Return only the concepts as a JSON list of strings, for example: ["alcohol", "sleep", "fatigue"]'
        )
        try:
            response = self.llm.query(keyword_prompt)
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                keyword_list = json.loads(json_match.group())
                return [kw.strip() for kw in keyword_list]
            else:
                print("No keyword list detected in LLM output.")
                return []
        except Exception as e:
            print(f"Failed to parse keyword list: {e}")
            return []

    def build_knowledge_graph(self, keywords: List[str], max_depth: int = 2, max_edges_per_hop: int = 6, min_weight: float = 1.0) -> nx.DiGraph:
        graph = nx.DiGraph()
        visited_concepts = set()
        queue: List[Tuple[str, int, Union[str, None]]] = []
        for keyword in keywords:
            concept_uri = self.get_concept_uri(keyword)
=======
import re
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from . import prompts

class KnowledgeProberError(Exception):
    """Custom exception for KnowledgeProber errors"""
    pass

class Knowledge_Prober:
    """
    Knowledge Prober with adaptive search strategy and simplified structure detection.
    Uses staged exploration: shallow → moderate → deep, only escalating when needed.
    """
    
    def __init__(self, llm_handler: Any, max_retries: int = 3, retry_delay: float = 1.0):
        self.llm = llm_handler
        self.api_url = "http://api.conceptnet.io"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Adaptive search strategy: Start shallow, go deeper if needed
        self.search_stages = [
            # Stage 1: Fast initial exploration (should handle 80% of cases)
            {
                "name": "shallow",
                "max_depth": 2,
                "max_edges_per_hop": 5,
                "min_weight": 0.8,
                "min_structures_needed": 2,
                "min_graph_size": 8
            },
            # Stage 2: Moderate exploration (for 15% of cases)
            {
                "name": "moderate", 
                "max_depth": 3,
                "max_edges_per_hop": 7,
                "min_weight": 0.5,
                "min_structures_needed": 1,
                "min_graph_size": 5
            },
            # Stage 3: Deep exploration (for 5% of complex cases)
            {
                "name": "deep",
                "max_depth": 3,
                "max_edges_per_hop": 10,
                "min_weight": 0.3,
                "min_structures_needed": 0,
                "min_graph_size": 3
            }
        ]
        
        # Enhanced relation weights focusing on core causal types
        self.relation_weights = {
            "Causes": 5.0,           # Direct causation - highest priority
            "HasSubevent": 4.0,      # Sequential causation
            "HasPrerequisite": 3.5,  # Necessary conditions
            "ObstructedBy": 3.0,     # Negative causation
            "MotivatedByGoal": 3.0,  # Intentional causation
            "IsA": 2.5,              # Taxonomic (sometimes causal)
            "HasProperty": 2.0,      # Property relationships
            "UsedFor": 1.8,          # Functional relationships
            "CapableOf": 1.5,        # Capability relationships
            "PartOf": 1.2,           # Part-whole relationships
            "HasContext": 0.8,       # Contextual information
            "RelatedTo": 0.2,        # Generic relationships
            "Synonym": 0.1,          # Language relationships
        }

    def get_concept_uri(self, keyword: str, lang: str = "en") -> str:
        """Generate ConceptNet URI for a keyword."""
        normalized_keyword = keyword.lower().replace(" ", "_")
        return f"/c/{lang}/{normalized_keyword}"

    def get_edges_for_concept(self, concept_uri: str, limit: int = 100) -> List[dict]:
        """
        Fetch edges for a concept with retry logic and better error handling.
        """
        url = f"{self.api_url}{concept_uri}"
        params = {"limit": limit}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get('edges', [])
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {concept_uri}: {e}")
                break
        
        return []

    def extract_keywords_from_sentence(self, sentence: str) -> List[str]:
        """Extract keywords using LLM with better error handling."""
        try:
            prompt = prompts.KEYWORD_EXTRACTION_PROMPT.format(sentence=sentence)
            response = self.llm.query(prompt)
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                keyword_list = json.loads(json_match.group())
                return [kw.strip() for kw in keyword_list if isinstance(kw, str)]
            else:
                return []
                
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

    def probe_with_structure(self, sentence: str, verbose: bool = False) -> dict:
        """
        Main probing function with adaptive search strategy.
        Starts shallow and goes deeper only if needed.
        """
        if verbose:
            print(f"Probing: '{sentence[:50]}...'")
        
        try:
            # Phase 1: Keyword extraction
            keywords = self.extract_keywords_from_sentence(sentence)
            if not keywords:
                return {"error": "No keywords extracted", "sentence": sentence}
            
            if verbose:
                print(f"Keywords: {keywords}")
            
            # Phase 2: Adaptive graph construction
            graph, stage_used = self._graph_construction(keywords, verbose)
            
            if verbose:
                print(f"Graph ({stage_used}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Phase 3: Causal structure detection
            structures = self.find_causal_structures(graph, verbose)
            
            # Phase 4: Generate causal analysis if structures found
            if any(structures.values()):
                causal_analysis = self.generate_causal_analysis(structures, graph)
            else:
                causal_analysis = "No meaningful causal structures detected for analysis."
            
            return {
                "sentence": sentence,
                "keywords": keywords,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "exploration_stage": stage_used,
                    "density": nx.density(graph) if graph.number_of_nodes() > 0 else 0
                },
                "structures_found": {k: len(v) for k, v in structures.items() if v},
                "causal_analysis": causal_analysis,
                "confidence": self._assess_simple_confidence(structures, graph, stage_used)
            }
            
        except Exception as e:
            return {"error": f"Probing error: {str(e)}", "sentence": sentence}

    def _graph_construction(self, keywords: list, verbose: bool = False):
        """
        Try shallow exploration first, go deeper only if results are insufficient.
        """
        for stage in self.search_stages:
            if verbose:
                print(f"  Trying {stage['name']} exploration...")
            
            graph = self._perform_random_walk(
                keywords=keywords,
                max_depth=stage["max_depth"],
                max_edges_per_hop=stage["max_edges_per_hop"], 
                min_weight=stage["min_weight"],
                relation_type_weights=self.relation_weights,
                verbose=False
            )
            
            # Check if this stage produced sufficient results
            structures = self.find_causal_structures(graph, verbose=False)
            total_structures = sum(len(v) for v in structures.values())
            
            is_sufficient = (
                graph.number_of_nodes() >= stage["min_graph_size"] and
                total_structures >= stage["min_structures_needed"]
            )
            
            if verbose:
                print(f"    {graph.number_of_nodes()} nodes, {total_structures} structures - {'Sufficient' if is_sufficient else 'Insufficient'}")
            
            if is_sufficient:
                return graph, stage["name"]
        
        # If we get here, use the deepest exploration result
        return graph, "deep"

    def _perform_random_walk(
        self,
        keywords: List[str],
        max_depth: int = 3,
        max_edges_per_hop: int = 5,
        language: str = "en",
        api_limit_per_request: int = 100,
        min_weight: float = 1.0,
        forward_only: bool = False,
        relation_type_weights: Optional[Dict[str, float]] = None,
        avoid_backtracking: bool = True,
        verbose: bool = False
    ) -> nx.DiGraph:
        """Perform random walk on ConceptNet (core logic unchanged but with better error handling)."""
        if relation_type_weights is None:
            relation_type_weights = self.relation_weights

        graph = nx.DiGraph()
        visited_concepts = set()
        queue: List[Tuple[str, int, Union[str, None]]] = []

        # Initialize with keywords
        for keyword in keywords:
            concept_uri = self.get_concept_uri(keyword, language)
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
            if concept_uri not in visited_concepts:
                graph.add_node(concept_uri, name=keyword, type='keyword')
                queue.append((concept_uri, 0, None))
                visited_concepts.add(concept_uri)
<<<<<<< HEAD
=======

        # BFS-like traversal
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
        head = 0
        while head < len(queue):
            current_concept_uri, current_depth, prev_concept_uri = queue[head]
            head += 1
<<<<<<< HEAD
            if current_depth >= max_depth:
                continue
            raw_edges = self.get_edges_for_concept(current_concept_uri)
            if not raw_edges:
                continue
            eligible_edges = []
            for edge in raw_edges:
                if not all(k in edge and isinstance(edge[k], dict) for k in ['start', 'end', 'rel']):
                    continue
                if not all(k in edge['start'] and k in edge['end'] for k in ['@id', 'label', 'language']):
                    continue
                if '@id' not in edge['rel']:
                    continue
=======

            if current_depth >= max_depth:
                continue

            raw_edges = self.get_edges_for_concept(current_concept_uri, api_limit_per_request)
            if not raw_edges:
                continue

            # Filter and prepare edges
            eligible_edges = []
            for edge in raw_edges:
                if not self._is_valid_edge(edge, language, min_weight, forward_only, current_concept_uri, prev_concept_uri, avoid_backtracking):
                    continue
                
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
                start_uri = edge['start']['@id']
                end_uri = edge['end']['@id']
                relation_full_uri = edge['rel']['@id']
                relation_name = relation_full_uri.split('/')[-1]
                weight = edge.get('weight', 0)
<<<<<<< HEAD
                if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                    continue
                if weight < min_weight:
                    continue
                target_uri = end_uri if start_uri == current_concept_uri else start_uri
                if target_uri == prev_concept_uri:
                    continue
                if target_uri == current_concept_uri:
                    continue
=======
                target_uri = end_uri if start_uri == current_concept_uri else start_uri
                
                adjusted_weight = weight * relation_type_weights.get(relation_name, 1.0)
                if adjusted_weight <= 0:
                    continue

>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
                eligible_edges.append({
                    'edge_data': edge,
                    'start_uri': start_uri,
                    'end_uri': end_uri,
                    'relation_name': relation_name,
                    'original_weight': weight,
<<<<<<< HEAD
                    'target_uri': target_uri
                })
            num_to_select = min(max_edges_per_hop, len(eligible_edges))
            for item in eligible_edges[:num_to_select]:
                edge = item['edge_data']
                start_uri = item['start_uri']
                end_uri = item['end_uri']
                relation_name = item['relation_name']
                original_weight = item['original_weight']
                target_uri = item['target_uri']
                start_label = edge['start'].get('label', start_uri)
                end_label = edge['end'].get('label', end_uri)
                if start_uri not in graph:
                    graph.add_node(start_uri, name=start_label, type='concept')
                if end_uri not in graph:
                    graph.add_node(end_uri, name=end_label, type='concept')
                if not graph.has_edge(start_uri, end_uri):
                    graph.add_edge(start_uri, end_uri, relation=relation_name, weight=original_weight)
                if target_uri not in visited_concepts:
                    queue.append((target_uri, current_depth + 1, current_concept_uri))
                    visited_concepts.add(target_uri)
        return graph

    def probe_step(self, step: str) -> Tuple[List[str], nx.DiGraph, Dict[str, Any]]:
        """
        对单个step做关键词抽取、知识图谱检索、结构分析，自动缓存。
        返回 (keywords, graph, structures)
        """
        if step in self.step_cache:
            return self.step_cache[step]
        keywords = self.extract_keywords_from_sentence(step)
        graph = self.build_knowledge_graph(keywords)
        structures = self.find_causal_structures(graph)
        self.step_cache[step] = (keywords, graph, structures)
        return keywords, graph, structures
=======
                    'adjusted_weight': adjusted_weight,
                    'target_uri': target_uri
                })

            if not eligible_edges:
                continue

            # Weighted random selection
            num_to_select = min(max_edges_per_hop, len(eligible_edges))
            selection_weights = [item['adjusted_weight'] for item in eligible_edges]

            try:
                selected_items = random.choices(eligible_edges, weights=selection_weights, k=num_to_select)
            except ValueError:
                selected_items = random.sample(eligible_edges, num_to_select)

            # Add selected edges to graph
            for item in selected_items:
                self._add_edge_to_graph(graph, item, visited_concepts, queue, current_depth, current_concept_uri)

        return graph

    def _is_valid_edge(self, edge, language, min_weight, forward_only, current_concept_uri, prev_concept_uri, avoid_backtracking):
        """Check if an edge meets filtering criteria."""
        if not all(k in edge and isinstance(edge[k], dict) for k in ['start', 'end', 'rel']):
            return False
        if not all(k in edge['start'] and k in edge['end'] for k in ['@id', 'label', 'language']):
            return False
        if '@id' not in edge['rel']:
            return False

        start_uri = edge['start']['@id']
        end_uri = edge['end']['@id']
        weight = edge.get('weight', 0)

        # Language filter
        if edge['start']['language'] != language or edge['end']['language'] != language:
            return False

        # Weight filter
        if weight < min_weight:
            return False

        # Forward only filter
        if forward_only and start_uri != current_concept_uri:
            return False

        # Target URI
        target_uri = end_uri if start_uri == current_concept_uri else start_uri

        # Avoid backtracking
        if avoid_backtracking and target_uri == prev_concept_uri:
            return False

        # Avoid self-loops
        if target_uri == current_concept_uri:
            return False

        return True

    def _add_edge_to_graph(self, graph, item, visited_concepts, queue, current_depth, current_concept_uri):
        """Add an edge and its nodes to the graph."""
        edge = item['edge_data']
        start_uri = item['start_uri']
        end_uri = item['end_uri']
        relation_name = item['relation_name']
        original_weight = item['original_weight']
        target_uri = item['target_uri']

        # Add nodes
        start_label = edge['start'].get('label', start_uri)
        end_label = edge['end'].get('label', end_uri)

        if start_uri not in graph:
            graph.add_node(start_uri, name=start_label, type='concept')
        if end_uri not in graph:
            graph.add_node(end_uri, name=end_label, type='concept')

        # Add edge
        if not graph.has_edge(start_uri, end_uri):
            graph.add_edge(start_uri, end_uri, relation=relation_name, weight=original_weight)

        # Enqueue target for exploration
        if target_uri not in visited_concepts:
            queue.append((target_uri, current_depth + 1, current_concept_uri))
            visited_concepts.add(target_uri)

    def find_causal_structures(self, graph: nx.DiGraph, verbose: bool = False) -> dict:
        """
        Find causal structures with 4 core types.
        Uses heuristic approach focused on supporting LLM analysis.
        """
        structures = {
            'direct_links': [],    # Simple A → B relationships
            'chains': [],          # A → B → C+ (any length chain)
            'forks': [],          # A → {B,C+} (any number of effects)
            'colliders': [],      # {A,B+} → C (any number of causes)
        }
        
        # 1. Direct causal links
        structures['direct_links'] = self._find_direct_causal_links(graph)
        
        # 2. Chains of any length
        structures['chains'] = self._find_any_length_chains(graph)
        
        # 3. Forks with any number of effects
        structures['forks'] = self._find_any_size_forks(graph)
        
        # 4. Colliders with any number of causes
        structures['colliders'] = self._find_any_size_colliders(graph)
        
        if verbose:
            total = sum(len(v) for v in structures.values())
            print(f"Found {total} structures: {[f'{k}:{len(v)}' for k,v in structures.items() if v]}")
        
        return structures

    def _find_direct_causal_links(self, graph: nx.DiGraph) -> list:
        """Find strong direct A → B relationships."""
        direct_links = []
        strong_relations = {'Causes', 'HasSubevent', 'HasPrerequisite', 'ObstructedBy'}
        
        for source, target, data in graph.edges(data=True):
            if (data.get('relation') in strong_relations and 
                data.get('weight', 0) >= 1.0):
                
                direct_links.append({
                    'source': graph.nodes[source].get('name', source),
                    'target': graph.nodes[target].get('name', target),
                    'relation': data.get('relation'),
                    'weight': data.get('weight', 0)
                })
        
        return direct_links

    def _find_any_length_chains(self, graph: nx.DiGraph, max_length: int = 4) -> list:
        """Find causal chains of any reasonable length."""
        chains = []
        
        for path_length in [2, 3, 4]:
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            paths = list(nx.all_simple_paths(graph, source, target, cutoff=path_length-1))
                            for path in paths:
                                if len(path) == path_length and self._is_causal_path(graph, path):
                                    chains.append({
                                        'path': [graph.nodes[node].get('name', node) for node in path],
                                        'length': len(path),
                                        'strength': self._calculate_path_strength(graph, path)
                                    })
                        except:
                            continue
        
        return sorted(chains, key=lambda x: x['strength'], reverse=True)[:5]

    def _find_any_size_forks(self, graph: nx.DiGraph) -> list:
        """Find fork structures with 2+ effects."""
        forks = []
        
        for cause in graph.nodes():
            effects = []
            for effect in graph.successors(cause):
                edge_data = graph.get_edge_data(cause, effect)
                if self._is_causal_edge(edge_data) and isinstance(edge_data, dict):
                    effects.append({
                        'name': graph.nodes[effect].get('name', effect),
                        'relation': edge_data.get('relation'),
                        'weight': edge_data.get('weight', 0)
                    })
            
            if len(effects) >= 2:
                forks.append({
                    'cause': graph.nodes[cause].get('name', cause),
                    'effects': effects,
                    'num_effects': len(effects)
                })
        
        return sorted(forks, key=lambda x: x['num_effects'], reverse=True)[:3]

    def _find_any_size_colliders(self, graph: nx.DiGraph) -> list:
        """Find collider structures with 2+ causes.""" 
        colliders = []
        
        for effect in graph.nodes():
            causes = []
            for cause in graph.predecessors(effect):
                edge_data = graph.get_edge_data(cause, effect)
                if self._is_causal_edge(edge_data) and isinstance(edge_data, dict):
                    causes.append({
                        'name': graph.nodes[cause].get('name', cause),
                        'relation': edge_data.get('relation'),
                        'weight': edge_data.get('weight', 0)
                    })
            
            if len(causes) >= 2:
                colliders.append({
                    'effect': graph.nodes[effect].get('name', effect),
                    'causes': causes,
                    'num_causes': len(causes)
                })
        
        return sorted(colliders, key=lambda x: x['num_causes'], reverse=True)[:3]

    def _is_causal_edge(self, edge_data: Optional[Union[Dict[str, Any], Any]]) -> bool:
        """Check if an edge represents meaningful causal relationship."""
        if not edge_data:
            return False
        
        relation = edge_data.get('relation', '')
        weight = edge_data.get('weight', 0)
        
        # Strong causal indicators
        if relation in {'Causes', 'HasSubevent', 'HasPrerequisite', 'ObstructedBy'}:
            return weight >= 0.5
        
        # Moderate indicators need higher weight
        if relation in {'IsA', 'HasProperty', 'UsedFor'}:
            return weight >= 1.5
        
        return False

    def _is_causal_path(self, graph: nx.DiGraph, path: list) -> bool:
        """Check if a path represents meaningful causal sequence."""
        if len(path) < 2:
            return False
        
        causal_edges = 0
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            if self._is_causal_edge(edge_data) and isinstance(edge_data, dict):
                causal_edges += 1
        
        return causal_edges >= (len(path) - 1) // 2

    def _calculate_path_strength(self, graph: nx.DiGraph, path: list) -> float:
        """Calculate overall strength of a causal path.""" 
        if len(path) < 2:
            return 0.0
        
        weights = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            weights.append(edge_data.get('weight', 0))
        
        # Use minimum weight (weakest link) with length penalty
        return min(weights) * (0.9 ** (len(path) - 2))

    def generate_causal_analysis(self, structures: dict, graph: nx.DiGraph) -> str:
        """Generate causal analysis summary for use in integrated validation."""
        structure_summary = self._create_structure_summary(structures)
        
        if not structure_summary.strip():
            return "No meaningful causal structures detected."
        
        # Return structured summary instead of separate LLM call
        # This will be used directly in the integrated validation prompt
        return structure_summary

    def _create_structure_summary(self, structures: dict) -> str:
        """Create summary of all found structures."""
        summary_parts = []
        
        for structure_type, structure_list in structures.items():
            if not structure_list:
                continue
                
            if structure_type == 'direct_links':
                summary_parts.append(f"**Direct Causal Links ({len(structure_list)}):**")
                for link in structure_list[:3]:
                    summary_parts.append(f"  • {link['source']} --[{link['relation']}]--> {link['target']} (strength: {link['weight']:.2f})")
            
            elif structure_type == 'chains':
                summary_parts.append(f"**Causal Chains ({len(structure_list)}):**")
                for chain in structure_list[:2]:
                    path_str = " → ".join(chain['path'])
                    summary_parts.append(f"  • {path_str} (strength: {chain['strength']:.2f})")
            
            elif structure_type == 'forks':
                summary_parts.append(f"**Common Cause Patterns ({len(structure_list)}):**")
                for fork in structure_list[:2]:
                    effects_str = ", ".join([e['name'] for e in fork['effects'][:3]])
                    summary_parts.append(f"  • {fork['cause']} → [{effects_str}] ({fork['num_effects']} total effects)")
            
            elif structure_type == 'colliders':
                summary_parts.append(f"**Common Effect Patterns ({len(structure_list)}):**")
                for collider in structure_list[:2]:
                    causes_str = ", ".join([c['name'] for c in collider['causes'][:3]])
                    summary_parts.append(f"  • [{causes_str}] → {collider['effect']} ({collider['num_causes']} total causes)")
        
        return "\n".join(summary_parts) if summary_parts else ""

    def _assess_simple_confidence(self, structures: dict, graph: nx.DiGraph, stage_used: str) -> str:
        """Simple confidence assessment."""
        total_structures = sum(len(v) for v in structures.values())
        
        if total_structures >= 3 and graph.number_of_edges() >= 10 and stage_used in ['shallow', 'moderate']:
            return "high"
        elif total_structures >= 1 and graph.number_of_edges() >= 5:
            return "medium"
        else:
            return "low"
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
