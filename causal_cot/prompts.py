# causal_cot/prompts.py
"""
Centralized prompt management for the Causal Chain-of-Thought pipeline.
Enhanced with clearer causal analysis structure incorporating Pearl's causal theory.
"""

def get_task_context(dataset_type):
    """根据数据集类型获取任务上下文描述"""
    if dataset_type == "multiple_choice":
        return "This is a multiple choice question. You need to select the best option from the given choices."
    else:
        return "This is a fill-in-the-blank or open-ended question. Provide a specific and accurate answer."

def get_conclusion_instruction(dataset_type):
    """根据数据集类型获取结论指令"""
    if dataset_type == "multiple_choice":
        return "Conclusion: The final answer is \\box{{option letter or number (A, B, C, D or 0, 1, 2, 3)}}. My answer is: [option letter or number]"
    else:
        return "Conclusion: The final answer is \\box{{your final answer, based strictly on the above causal reasoning}}. My answer is: [your final answer]"

def get_additional_instruction(dataset_type):
    """根据数据集类型获取额外指令"""
    if dataset_type == "multiple_choice":
        return "\n**IMPORTANT:** For multiple choice questions, your final answer MUST be only the option letter (A, B, C, D) or number (0, 1, 2, 3). Do not include any additional text. The last sentence must clearly restate your answer as 'My answer is: [letter or number]'."
    else:
        return "\n**IMPORTANT:** The last sentence must clearly restate your answer as 'My answer is: [your answer]' for easy extraction."

def create_cot_generation_prompt(dataset_type="fill_in_blank"):
    """根据数据集类型创建COT生成prompt"""
    task_context = get_task_context(dataset_type)
    conclusion_instruction = get_conclusion_instruction(dataset_type)
    additional_instruction = get_additional_instruction(dataset_type)
    
    return f"""**Role:** You are a meticulous logical reasoner.
**Task:** {task_context}
Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.

**Question:**
{{question_and_choices}}

**Output Format:**
Step 1: [First causal deduction or analysis]
Step 2: [Second deduction, building upon Step 1]
...
Step N: [Final deduction that directly leads to the answer]
===FINAL_ANSWER_START===
{conclusion_instruction}
===FINAL_ANSWER_END==={additional_instruction}
"""

# Core pipeline prompts (保持向后兼容)
COT_GENERATION_PROMPT = """**Role:** You are a meticulous logical reasoner.
**Task:** Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.

**Question:**
{question_and_choices}

**Output Format:**
Step 1: [First causal deduction or analysis]
Step 2: [Second deduction, building upon Step 1]
...
Step N: [Final deduction that directly leads to the answer]
===FINAL_ANSWER_START===
Conclusion: The final answer is \box{{your final answer, based strictly on the above causal reasoning}}.
===FINAL_ANSWER_END===
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
Conclusion: Based on the new reasoning, the correct answer is [your answer].
"""

REFLECTION_AND_REGENERATION_PROMPT_FORCE_FIX = """Role: You are a self-correcting reasoning agent.
Context: Your previous attempt to solve a problem contained a logical or factual error.

**Question:** {question_and_choices}

**Validated Steps So Far:**
{validated_facts}

**Error Analysis:**
Failed Step: "{failed_step}"
Detailed Analysis of Failure: {failure_reason}

**Your Task:**
You MUST assume that the failed step and the current reasoning path are incorrect or causally invalid. You are REQUIRED to make a substantive correction. Do NOT simply repeat or rephrase the previous reasoning. Instead, generate a new, corrected reasoning path that addresses the identified causal/logical issues and leads to a potentially different answer if needed.

**IMPORTANT:** If your new reasoning is essentially the same as the previous one, it will be considered INVALID. You must provide a reasoning path and conclusion that is clearly different from the original, and your answer must be causally justified.

**Output Format:**
Step {step_index}: [A new, corrected reasoning step]
Step {step_index_plus_1}: [Continue improved reasoning]
...
Conclusion: Based on the new reasoning, the correct answer is [your answer].
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

**2. LOGICAL CONSISTENCY:**
- Is the reasoning step internally consistent?
- Does it contradict established facts or previous validated steps?
- Are there logical fallacies or unsupported assumptions?

**3. FACTUAL ACCURACY:**
- Are the factual claims in this step correct?
- Is the step grounded in domain knowledge?
- Are there contradictions with well-established facts?

**4. CAUSAL VALIDITY:**
Based on the causal structures and logical analysis, does this step represent:
a) **Valid causal reasoning** with appropriate evidence
b) **Weak but plausible** causal inference with some support
c) **Invalid or misleading** reasoning that should be rejected

**FINAL VERDICT:**
Provide your verdict as one of: "VALID", "WEAKLY_VALID", or "INVALID"

If INVALID or WEAKLY_VALID, explain the specific causal or logical issues that need correction.
"""

# Keyword extraction prompt
KEYWORD_EXTRACTION_PROMPT = """Given the following sentence:

"{sentence}"

Extract 2-5 key concepts that are central to the meaning and reasoning of the sentence. Focus on concrete entities, processes, or concepts that could have causal relationships.

Return only the concepts as a JSON list of strings, for example: ["alcohol", "sleep", "fatigue"]"""

# Causal structure analysis prompts
CAUSAL_CHAIN_ANALYSIS_PROMPT = """Analyze this chain structure using Pearl's framework:

**Causal Chain Pattern:** {node1} → {node2} → {node3}
**Evidence Strengths:** {weight1:.2f}, {weight2:.2f}

**Mediation Assessment:**
1. **Direct Effects**: Does {node1} plausibly cause {node2}? Does {node2} plausibly cause {node3}?
2. **D-Separation**: If we control for {node2}, should {node1} and {node3} become independent?
3. **Mechanistic Plausibility**: Are both causal steps biologically/socially reasonable?
4. **Alternative Pathways**: Could {node1} affect {node3} through other routes?

Evaluate whether this represents a valid causal chain or statistical artifact.
"""

CAUSAL_FORK_ANALYSIS_PROMPT = """Analyze this fork structure using Pearl's framework:

**Common Cause Pattern:** {cause} → {effect1}, {cause} → {effect2}
**Evidence Strengths:** {weight1:.2f}, {weight2:.2f}

**Confounding Assessment:**
1. **Common Cause Validity**: Does {cause} genuinely influence both {effect1} and {effect2}?
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

CAUSAL_STRUCTURE_ANALYSIS_PROMPT = """You are an expert in causal inference. Analyze the following structure and provide a detailed causal reasoning. If you cannot find any meaningful causal relationship, state clearly: 'No meaningful causal structure found.' Otherwise, explain the causal logic in detail.

Structure: {structure_desc}

Your analysis must include explicit causal reasoning, not just associations or correlations. You must clearly state whether a genuine causal relationship exists, and justify your answer with reference to the structure and context.
"""

# Legacy prompts for backward compatibility
STEP_VALIDATION_PROMPT = """You are an expert in causal reasoning and logical analysis.

**Question Context:**
{original_question}

**Previous Validated Steps:**
{previous_context}

**Step to Evaluate:**
"{reasoning_step}"

**Causal Knowledge Context:**
{causal_context}

**Task:**
Evaluate whether this reasoning step is causally valid and logically sound. Consider:
1. Is it causally plausible given the knowledge context?
2. Is it logically consistent with previous steps?
3. Are there any factual errors or unsupported assumptions?

**Response Format:**
Status: [VALID/INVALID]
Explanation: [Detailed reasoning for your assessment]
"""

SIMPLIFIED_VALIDATION_PROMPT = """Evaluate this reasoning step for causal validity:

**Question:** {original_question}
**Step:** "{reasoning_step}"
**Context:** {causal_context}

Is this step causally valid? Respond with VALID or INVALID and explain why.
"""
