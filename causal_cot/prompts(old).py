# causal_cot/prompts.py
"""
Centralized prompt management for the Causal Chain-of-Thought pipeline.
Enhanced with clearer causal analysis structure incorporating Pearl's causal theory.
"""

# Core pipeline prompts
COT_GENERATION_PROMPT = """**Role:** You are a meticulous logical reasoner.
**Task:** Solve the following multiple-choice question by creating a step-by-step Chain of Thought. Your reasoning should start from the premises in the question and logically progress toward a conclusion that supports one of the choices.

**Question and Choices:**
{question_and_choices}

**Output Format:**
Step 1: [First logical deduction or analysis of the premise]
Step 2: [Second deduction, building upon Step 1]
...
Step N: [Final deduction that directly leads to the answer]
Conclusion: Based on the reasoning, the correct answer is [A/B/C/D/E].
"""

REFLECTION_AND_REGENERATION_PROMPT = """Role: You are a self-correcting reasoning agent.
Context: Your previous attempt to solve a problem contained a logical or factual error.

**Question:** {question_and_choices}

**Validated Steps So Far:**
{validated_facts}

**Error Analysis:**
Failed Step: "{failed_step}"
Detailed Analysis of Failure: {failure_reason}

**Your Task:**
Acknowledge the error analysis. Then, starting from the last validated step, create a new, corrected reasoning path that addresses the identified causal and logical issues.

**Output Format:**
Step {step_index}: [A new, corrected reasoning step]
Step {step_index_plus_1}: [Continue improved reasoning]
...
Conclusion: Based on the new reasoning, the correct answer is [A/B/C/D/E].
"""

# Enhanced integrated prompt with Pearl's causal theory and clearer structure
INTEGRATED_CAUSAL_VALIDATION_PROMPT = """You are an expert in causal reasoning applying Pearl's causal framework to evaluate a reasoning step.

**ORIGINAL QUESTION:**
{original_question}

**PREVIOUS VALIDATED REASONING:**
{previous_context}

**CURRENT STEP TO EVALUATE:**
"{reasoning_step}"

**CAUSAL KNOWLEDGE ANALYSIS:**
Analysis of key concepts ({keywords}) revealed a knowledge graph with {num_nodes} concepts and {num_edges} relationships.

{causal_context}

**YOUR EXPERT CAUSAL-LOGICAL ASSESSMENT:**

Apply Pearl's causal theory and logical reasoning principles to evaluate this step:

**1. CAUSAL STRUCTURE ANALYSIS:**
- What do the discovered structures suggest about the causal plausibility of this reasoning step?
- Are there coherent causal pathways, or contradictions and confounds?
- How do the identified direct links, chains, forks, and colliders contribute to or challenge the step's claims?

For each structure type found:
- **Direct Links (A → B)**: Do these represent genuine causation or mere association? Consider strength and biological/social plausibility.
- **Chains (A → B → C)**: Does B plausibly mediate A's effect on C? Would A and C be independent given B (d-separation: A ⊥ C | B)?
- **Forks (A → B, A → C)**: Does A genuinely cause both B and C? Would B and C be spuriously correlated without controlling for A (B ⊥ C | A)?
- **Colliders (A → C, B → C)**: Are A and B independent causes of C? Would conditioning on C create spurious correlation between A and B (collider bias)?

**2. MECHANISTIC COHERENCE:**
- Are the implied mechanisms biologically, psychologically, or socially plausible given current knowledge?
- Do the knowledge graph structures reveal missing causal links, redundant pathways, or unsupported jumps?
- Does the reasoning step propose a justifiable causal sequence supported by the discovered structures?

**3. LOGICAL CONSISTENCY CHECK:**
- Does this step logically follow from the previously validated reasoning steps?
- Are there hidden assumptions, logical leaps, or internal inconsistencies?
- Does this step meaningfully advance the argument toward the conclusion?

**4. CAUSAL VALIDITY ASSESSMENT:**
- Would the causal claims imply plausible outcomes under intervention (if we manipulated the proposed cause)?
- Does the reasoning step mistake correlation for causation?
- Are key causal principles violated (d-separation, independence assumptions, confounding)?
- Consider potential alternative explanations: reverse causation, common causes, selection bias.

**5. INTEGRATION WITH PRIOR REASONING:**
- Is this step consistent with previously validated causal claims in the reasoning chain?
- Does it introduce contradictions or meaningfully reinforce earlier causal logic?
- How does it fit into the overall causal narrative being constructed?

**6. DECISION AND RATIONALE:**
Based on your integrated causal-logical analysis:

Provide your assessment in this exact format:

DECISION: [ACCEPT/REJECT]
CONFIDENCE: [HIGH/MEDIUM/LOW]
KEY_REASONING: [Your primary reasoning in 2-3 sentences explaining why the step passes or fails causal and logical scrutiny]
RECOMMENDED_ACTION: [accept/regenerate_with_causal_focus/regenerate_with_logical_focus/regenerate_completely]
DETAILED_ANALYSIS: [Your complete reasoning process showing how causal structures and Pearl's principles inform your validation decision]
"""

INTEGRATED_CAUSAL_VALIDATION_PROMPT_V2 = """You are an expert in causal reasoning applying Pearl's causal framework to evaluate a single reasoning step.

**Original Question:**
{original_question}

**Previously Validated Reasoning:**
{previous_context}

**Current Step to Evaluate:**
"{reasoning_step}"

**Causal Knowledge Analysis from Knowledge Graph:**
{causal_context}

**Your Expert Assessment Task:**
Analyze the "Current Step to Evaluate" using the provided context and your knowledge of causal reasoning (confounding, mediation, collision, etc.) and logical consistency. Determine if the step is valid.

Your final output MUST be a single word that best describes your conclusion:
- **ACCEPT**: If the step is causally and logically sound.
- **REJECT_CAUSAL**: If the step makes a causally flawed claim (e.g., mistakes correlation for causation, ignores a clear confounder).
- **REJECT_LOGICAL**: If the step is logically inconsistent with the premises or previous steps, even if not causally wrong.

**Final Decision (Single Word):**
"""



# Simple keyword extraction prompt
KEYWORD_EXTRACTION_PROMPT = """Given the following sentence:

"{sentence}"

Extract 2-5 key concepts that are central to the meaning and reasoning of the sentence. 
Return only the concepts as a JSON list of strings, for example: ["alcohol", "sleep", "fatigue"]
"""

# Enhanced keyword extraction prompt
KEYWORD_EXTRACTION_PROMPT_ENHANCED = '''Given the following sentence:

"{sentence}"

Extract 2-5 key concepts that are most central to the commonsense and causal reasoning of the sentence. Only return concrete nouns or noun phrases (e.g., objects, entities, events, or specific actions), and avoid generic words like 'thing', 'purpose', 'goal', 'person', 'place', etc. Return only the concepts as a JSON list of strings, for example: ["alcohol", "sleep", "fatigue"].'''

# Backup prompts for individual structure analysis (updated for our 4 structure types)
CAUSAL_CHAIN_ANALYSIS_PROMPT = """Analyze this causal chain using Pearl's framework:

**Chain:** {node1} --[{relation1}]--> {node2} --[{relation2}]--> {node3}
**Evidence Strengths:** {weight1:.2f}, {weight2:.2f}

**Causal Assessment:**
1. **Mediation Analysis**: Does {node2} plausibly mediate {node1}'s effect on {node3}?
2. **D-Separation**: If we control for {node2}, should {node1} and {node3} become independent?
3. **Mechanistic Plausibility**: Are both causal steps biologically/socially reasonable?
4. **Alternative Pathways**: Could {node1} affect {node3} through other routes?

Evaluate whether this represents a valid causal chain or statistical artifact.
"""

CAUSAL_FORK_ANALYSIS_PROMPT = """Analyze this fork structure using Pearl's framework:

**Common Cause Pattern:** {cause} → {effect1}, {cause} → {effect2}
**Evidence Strengths:** {weight1:.2f}, {weight2:.2f}

**Confounding Assessment:**
1. **Common Cause ValidQity**: Does {cause} genuinely influence both {effect1} and {effect2}?
2. **Conditional Independence**: Should {effect1} and {effect2} be independent given {cause}?
3. **Spurious Correlation**: Could correlation between {effect1} and {effect2} be due to {cause}?
4. **Confounding Implications**: What would this mean for studies relating {effect1} and {effect2}?

Evaluate the confounding potential and causal validity of this structure.
"""

CAUSAL_COLLIDER_ANALYSIS_PROMPT = """Analyze this collider structure using Pearl's framework:

**Common Effect Pattern:** {cause1} → {effect}, {cause2} → {effect}
**Evidence Strengths:** {weight1:.2f}, {weight2:.2f}

**Collider Bias Assessment:**
1. **Independent Causation**: Do {cause1} and {cause2} independently contribute to {effect}?
2. **Selection Bias Risk**: Would studying {effect} create spurious {cause1}-{cause2} correlation?
3. **Conditional Dependence**: Are {cause1} and {cause2} independent unconditionally but dependent given {effect}?
4. **Research Implications**: How might this bias studies that condition on {effect}?

Evaluate the collider bias potential and methodological implications.
"""


FOCUSED_REGENERATION_PROMPT = """**Role:** You are a self-correcting reasoning agent.
**Context:** Your previous reasoning attempt contained a specific flaw.

**Original Question:**
{question_and_choices}

**Validated Steps So Far:**
{validated_facts}

**Error Analysis:**
- **Failed Step:** "{failed_step}"
- **Identified Flaw Type:** {failure_type}
- **Explanation:** {failure_reason}

**Your Task:**
Acknowledge the flaw. Then, starting from the last validated step, create a new, corrected reasoning path that specifically addresses the identified **{failure_type}** issue. Ensure your new steps are more robust.

**Output Format:**
Step {step_index}: [A new, corrected reasoning step]
Step {step_index_plus_1}: [Continue improved reasoning]
...
Conclusion: Based on the new reasoning, the correct answer is [A/B/C/D/E].
"""