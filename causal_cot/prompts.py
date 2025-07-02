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



# ... COT_GENERATION_PROMPT 和 REFLECTION_AND_REGENERATION_PROMPT 保持不变 ...

# --- 新增：LLM驱动的实体提取 ---
ENTITY_EXTRACTION_PROMPT = """
**Role:** You are a highly accurate Natural Language Understanding (NLU) system.
**Task:** From the given sentence, extract the two most important entities that form the core assertion or relationship. Focus on nouns or core concepts.

**Sentence:**
"{sentence}"

**Output Format (Strictly JSON only):**
{{
  "entity1": "first_core_entity",
  "entity2": "second_core_entity"
}}
"""

# --- 重大升级：专家级因果验证器 ---
EXPERT_CAUSAL_VERIFICATION_PROMPT = """
**Role:** You are an expert in causal inference, applying Judea Pearl's structural causal model framework. Your task is to rigorously analyze the causal claim implied in a reasoning step, supported by a local knowledge graph constructed via random walks.

**Context:**
- We are verifying the reasoning step: "{step_to_verify}"
- A local knowledge graph was constructed by performing random walks from the core entities. Here is the evidence found in ConceptNet:
{local_knowledge_graph}

**Your Detailed Task (Follow these 4 steps):**

**1. Identify Core Causal Claim:**
First, analyze the sentence "{step_to_verify}" and state the primary causal claim it implies in the form "A -> B" or "A is related to B". Let's call the potential cause 'A' and the potential effect 'B'.

**2. Analyze Causal Structure from Graph:**
Now, examine the provided local knowledge graph. Trace the paths between the entities you identified as A and B. Classify the most likely causal structure that explains their relationship:
   - **Chain (A -> M -> B):** A causes B through a mediator M. The path is valid and explanatory.
   - **Fork (A <- C -> B):** A and B are likely spuriously correlated due to a common cause (confounder) C. The direct path A -> B is likely invalid.
   - **Collider (A -> C <- B):** A and B are independent causes of a common effect C. The path is invalid, and conditioning on C could create misleading associations.
   - **Direct Causal Path (A -> B):** The evidence strongly supports a direct causal link (e.g., via a 'Causes' or 'HasSubevent' relation).
   - **Unsupported/Ambiguous:** The graph is too sparse, disconnected, or contradictory to support a clear causal link.

**3. Explain Reasoning (Pearl's Theory):**
Provide a concise explanation for your classification.
   - If **Fork**, name the likely confounder C.
   - If **Chain**, name the mediator M.
   - If **Collider**, explain the implication of conditioning on C.
   - If **Unsupported**, explain why the evidence is insufficient.

**4. Final Judgment & Decision:**
Based on your analysis, is the original reasoning step "{step_to_verify}" causally sound enough to be included in a rigorous line of reasoning?

**Output Format (Strictly JSON within ```json ... ``` block):**
```json
{{
  "identified_claim": "Your identified 'A -> B' claim",
  "causal_structure": "Chain | Fork | Collider | Direct | Unsupported/Ambiguous",
  "explanation": "Your detailed reasoning based on Pearl's theory, identifying any M, C, or lack of evidence.",
  "should_include": true | false
}}
"""