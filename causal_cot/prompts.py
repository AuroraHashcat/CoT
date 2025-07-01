# causal_cot/prompts.py
import re

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

EXPERT_CAUSAL_VERIFICATION_PROMPT = """
**Role:** You are an expert in causal inference, applying Judea Pearl's structural causal model framework. Your task is to rigorously analyze a single reasoning step.

**Context:**
- We are verifying the step: "{step_to_verify}"
- The core claim appears to be: **"{claim_subject} -> {claim_object}"**
- The following facts have already been validated in previous steps:
{validated_facts}
- The following evidence has been retrieved from the ConceptNet knowledge graph regarding the claim's entities:
{kg_evidence}

**Your Detailed Task (Follow these 4 steps):**

**1. Identify Causal Structure:**
Based on the KG evidence and common sense, classify the relationship between "{claim_subject}" (A) and "{claim_object}" (B). Choose the most likely structure:
   - **Chain (A -> M -> B):** A causes B through a mediator M. The path is valid.
   - **Fork (A <- C -> B):** A and B are spuriously correlated due to a common cause (confounder) C. The direct path A -> B is invalid.
   - **Collider (A -> C <- B):** A and B are independent causes of a common effect C. The path is invalid, but conditioning on C could create a spurious correlation.
   - **Direct Causal Path (A -> B):** The evidence strongly supports a direct causal link without obvious mediators or confounders.
   - **Unsupported:** The evidence does not support any clear causal link.

**2. Explain Reasoning (Pearl's Theory):**
Provide a concise explanation for your classification.
   - If **Fork**, identify the likely confounder C.
   - If **Chain**, identify the mediator M.
   - If **Collider**, explain why it's a collider and the implication of conditioning on C.

**3. Decide on Inclusion:**
Based on your causal analysis, should this reasoning step be included in our final, sound reasoning chain? (true/false)

**4. Suggestion for Colliders (If applicable):**
If you identified a collider, state whether additional background knowledge is necessary to disambiguate the relationship. For example, knowing whether A and B are independent *before* observing C.

**Output Format (Strictly JSON within ```json ... ``` block):**
```json
{{
  "causal_structure": "Chain | Fork | Collider | Direct | Unsupported",
  "explanation": "Your detailed reasoning based on Pearl's theory, identifying any M or C.",
  "should_include": true,
  "collider_suggestion": "Your suggestion here, or 'N/A' if not a collider."
}}
"""

REFLECTION_AND_REGENERATION_PROMPT = """Role: You are a self-correcting reasoning agent.
Context: Your previous attempt to solve a problem contained a logical or factual error.
Question: {question_and_choices}
Validated Steps So Far:
{validated_facts}
Error Analysis:
Failed Step: "{failed_step}"
Reason for Failure: {failure_reason}
Your Task:
Acknowledge the error. Then, starting from the last validated step, create a new, corrected reasoning path to solve the question.
Output Format:
Corrected Step {i}: [A new, corrected reasoning step]
...
Conclusion: Based on the new reasoning, the correct answer is [A/B/C/D/E].
"""