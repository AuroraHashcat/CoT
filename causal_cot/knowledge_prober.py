### **2. `causal_cot/knowledge_prober.py` **

# causal_cot/knowledge_prober.py
import requests
import random
import networkx as nx
import json
from typing import List, Tuple, Dict, Any, Optional, Union
import re

class KnowledgeProber:
    def __init__(self, llm_handler: Any):
        self.llm = llm_handler
        self.api_url = "http://api.conceptnet.io"

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
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('edges', [])
        except requests.exceptions.RequestException as e:
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

        response = self.llm.query(keyword_prompt)

        try:
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

    def probe_with_structure(
        self,
        sentence: str,
        relation_type_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        1. Extract keywords from input sentence using LLM.
        2. Build a ConceptNet-based semantic graph with weighted random walk.
        3. Identify possible causal chains/forks/colliders.
        4. Analyze each with LLM using Pearl's causal reasoning framework.
        """
        # 1. Keyword extraction
        keywords = self.extract_keywords_from_sentence(sentence)
        if verbose:
            print(f"🔍 Extracted keywords: {keywords}")
        if not keywords:
            return {"error": "No keywords extracted"}

        # 2. Define default relation weights if none provided
        if relation_type_weights is None:
            relation_type_weights = {
                "Causes": 4.0,
                "HasSubevent": 2.0,
                "IsA": 2.5,
                "HasProperty": 2.0,
                "UsedFor": 1.5,
                "CapableOf": 1.0,
                "PartOf": 1.2,
                "RelatedTo": 0.1,
                "HasContext": 0.1,
                "Synonym": 0.5,
            }

        # 3. Build the concept graph
        graph = self._perform_random_walk(
            keywords=keywords,
            max_depth=2,
            max_edges_per_hop=6,
            min_weight=1.0,
            forward_only=False,
            relation_type_weights=relation_type_weights,
            avoid_backtracking=True,
            verbose=verbose
        )

        if verbose:
            print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # 4. Causal structure recognition
        structures = self.find_causal_structures(graph)

        # 5. For each structure, call LLM to analyze
        all_results = []
        for structure_type, structure_list in structures.items():
            for triple_uris, edge_details in structure_list:
                a_uri, b_uri, c_uri = triple_uris

                # Get the display names for the LLM prompt from the graph nodes
                A_name = graph.nodes[a_uri].get('name', a_uri.split('/')[-1].replace('_', ' '))
                B_name = graph.nodes[b_uri].get('name', b_uri.split('/')[-1].replace('_', ' '))
                C_name = graph.nodes[c_uri].get('name', c_uri.split('/')[-1].replace('_', ' '))

                llm_node_names_tuple = (A_name, B_name, C_name)

                result = self.analyze_causal_structure_with_llm(
                    structure_type=structure_type,
                    node_names_tuple=llm_node_names_tuple,
                    edge_details=edge_details
                )
                all_results.append({
                    "type": structure_type,
                    "nodes": llm_node_names_tuple,
                    "llm_analysis": result
                })

        return {
            "sentence": sentence,
            "keywords": keywords,
            "structures_analyzed": all_results
        }
