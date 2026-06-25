class PromptBuilder:

    @staticmethod
    def build_prompt(
        query,
        expert_context
    ):

        evidence = ""

        for i, doc in enumerate(expert_context, start=1):

            evidence += f"""
==============================
Evidence {i}
==============================

Similarity Score:
{doc["score"]:.4f}

Question:
{doc["question"]}

Abstract:
{doc["abstract"]}

Reference Answer:
{doc["answer"]}

"""

        prompt = f"""
You are a trustworthy medical AI assistant.

Your task is to answer the user's question ONLY using the expert medical evidence.

Rules:

1. Use ONLY the evidence provided below.

2. If the evidence does not directly answer the question, reply EXACTLY:

INSUFFICIENT_EVIDENCE

3. Do NOT guess.

4. Do NOT use outside medical knowledge.

5. Do NOT generate information that is not supported by the evidence.

6. Keep the answer concise and evidence-based.

==============================
Expert Medical Evidence
==============================

{evidence}

==============================
User Question
==============================

{query}

==============================
Final Answer
==============================
"""

        return prompt