"""
Foundation Models - Prompt Engineering Techniques

Demonstrates various prompt engineering patterns and strategies.
Shows zero-shot, few-shot, chain-of-thought, and self-consistency approaches.
Focuses on prompt structure and format - no actual LLM API calls.

No external dependencies.
"""

import json
import re
from typing import List, Dict, Any


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt: direct instruction without examples."""

    def __init__(self, task_description: str):
        template = f"{task_description}\n\nInput: {{input}}\n\nOutput:"
        super().__init__(template)


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt: instruction + examples."""

    def __init__(self, task_description: str, examples: List[Dict[str, str]]):
        self.task_description = task_description
        self.examples = examples

        # Build template
        template_parts = [task_description, "\n"]

        for i, ex in enumerate(examples, 1):
            template_parts.append(f"Example {i}:\n")
            template_parts.append(f"Input: {ex['input']}\n")
            template_parts.append(f"Output: {ex['output']}\n\n")

        template_parts.append("Now solve this:\n")
        template_parts.append("Input: {input}\n\nOutput:")

        super().__init__(''.join(template_parts))


class ChainOfThoughtPrompt(PromptTemplate):
    """Chain-of-thought: encourages step-by-step reasoning."""

    def __init__(self, task_description: str, examples: List[Dict[str, Any]]):
        self.task_description = task_description
        self.examples = examples

        template_parts = [task_description, "\n"]

        for i, ex in enumerate(examples, 1):
            template_parts.append(f"Example {i}:\n")
            template_parts.append(f"Q: {ex['question']}\n")
            template_parts.append(f"A: Let's think step by step.\n")
            template_parts.append(f"{ex['reasoning']}\n")
            template_parts.append(f"Therefore, the answer is {ex['answer']}.\n\n")

        template_parts.append("Now solve this:\n")
        template_parts.append("Q: {question}\n")
        template_parts.append("A: Let's think step by step.\n")

        super().__init__(''.join(template_parts))


class StructuredOutputPrompt(PromptTemplate):
    """Prompt for generating structured output (JSON)."""

    def __init__(self, task_description: str, schema: Dict[str, str]):
        self.schema = schema

        template_parts = [
            task_description,
            "\n\nOutput format (JSON):\n",
            json.dumps(schema, indent=2),
            "\n\nInput: {input}\n\nOutput JSON:\n"
        ]

        super().__init__(''.join(template_parts))


class RolePrompt(PromptTemplate):
    """Role-based prompt: assign persona to model."""

    def __init__(self, role: str, task: str):
        template = (
            f"You are {role}.\n\n"
            f"{task}\n\n"
            f"Input: {{input}}\n\nResponse:"
        )
        super().__init__(template)


class SelfConsistencyPrompt:
    """
    Self-consistency: generate multiple reasoning paths and vote.
    (Demonstrates structure, actual sampling would need LLM API)
    """

    def __init__(self, base_prompt: ChainOfThoughtPrompt, num_samples: int = 3):
        self.base_prompt = base_prompt
        self.num_samples = num_samples

    def format(self, **kwargs) -> List[str]:
        """Generate multiple prompts with temperature variation."""
        prompts = []
        base = self.base_prompt.format(**kwargs)

        for i in range(self.num_samples):
            # In practice, would use different random seeds or temperatures
            prompts.append(f"[Sample {i+1}]\n{base}")

        return prompts

    @staticmethod
    def aggregate_answers(answers: List[str]) -> str:
        """Find most common answer (simple majority vote)."""
        from collections import Counter

        # Extract final answers (simplified)
        extracted = [a.strip() for a in answers]
        vote = Counter(extracted)

        return vote.most_common(1)[0][0]


# ============================================================
# Demonstrations
# ============================================================

def demo_zero_shot():
    """Demonstrate zero-shot prompting."""
    print("=" * 60)
    print("DEMO 1: Zero-Shot Prompting")
    print("=" * 60)

    prompt = ZeroShotPrompt(
        "Classify the sentiment of the following text as Positive, Negative, or Neutral."
    )

    test_inputs = [
        "This product is amazing! Highly recommend.",
        "Terrible experience, would not buy again.",
        "The item arrived on time and matches the description.",
    ]

    for inp in test_inputs:
        formatted = prompt.format(input=inp)
        print(f"\n{formatted}\n")
        print("-" * 60)


def demo_few_shot():
    """Demonstrate few-shot prompting."""
    print("\n" + "=" * 60)
    print("DEMO 2: Few-Shot Prompting")
    print("=" * 60)

    examples = [
        {"input": "I love this!", "output": "Positive"},
        {"input": "This is awful.", "output": "Negative"},
        {"input": "It's okay.", "output": "Neutral"},
    ]

    prompt = FewShotPrompt(
        "Classify the sentiment of the text as Positive, Negative, or Neutral.",
        examples
    )

    test_input = "Great value for money!"
    formatted = prompt.format(input=test_input)

    print(f"\n{formatted}\n")


def demo_chain_of_thought():
    """Demonstrate chain-of-thought prompting."""
    print("\n" + "=" * 60)
    print("DEMO 3: Chain-of-Thought Prompting")
    print("=" * 60)

    examples = [
        {
            "question": "If a train travels 60 miles in 1 hour, how far does it travel in 2.5 hours?",
            "reasoning": "The train travels 60 miles per hour. For 2.5 hours, we multiply: 60 × 2.5 = 150 miles.",
            "answer": "150 miles"
        },
        {
            "question": "A store sells apples for $2 each. If I buy 3 apples and pay with a $10 bill, how much change do I get?",
            "reasoning": "Cost of 3 apples = 3 × $2 = $6. Change = $10 - $6 = $4.",
            "answer": "$4"
        }
    ]

    prompt = ChainOfThoughtPrompt(
        "Solve the following math word problems.",
        examples
    )

    test_question = "A book costs $15. If I buy 4 books and have a 20% discount, what is the total cost?"
    formatted = prompt.format(question=test_question)

    print(f"\n{formatted}\n")


def demo_structured_output():
    """Demonstrate structured output prompting."""
    print("\n" + "=" * 60)
    print("DEMO 4: Structured Output Prompting")
    print("=" * 60)

    schema = {
        "entity_name": "string",
        "entity_type": "person|organization|location",
        "sentiment": "positive|negative|neutral",
        "confidence": "float (0-1)"
    }

    prompt = StructuredOutputPrompt(
        "Extract entities from the text and analyze their sentiment.",
        schema
    )

    test_input = "Apple Inc. released a great new product yesterday in California."
    formatted = prompt.format(input=test_input)

    print(f"\n{formatted}\n")


def demo_role_based():
    """Demonstrate role-based prompting."""
    print("\n" + "=" * 60)
    print("DEMO 5: Role-Based Prompting")
    print("=" * 60)

    roles = [
        ("a helpful AI assistant", "Answer the following question concisely."),
        ("a technical expert in machine learning", "Explain the following concept in detail."),
        ("a friendly teacher explaining to a 10-year-old", "Explain the following concept simply."),
    ]

    test_input = "What is neural network?"

    for role, task in roles:
        prompt = RolePrompt(role, task)
        formatted = prompt.format(input=test_input)

        print(f"\n{formatted}\n")
        print("-" * 60)


def demo_self_consistency():
    """Demonstrate self-consistency approach."""
    print("\n" + "=" * 60)
    print("DEMO 6: Self-Consistency")
    print("=" * 60)

    examples = [
        {
            "question": "Is 17 a prime number?",
            "reasoning": "A prime number is only divisible by 1 and itself. Let's check divisors of 17: 2, 3, 5, 7, 11, 13 don't divide 17 evenly. Only 1 and 17 divide it.",
            "answer": "Yes"
        }
    ]

    cot_prompt = ChainOfThoughtPrompt(
        "Determine if the number is prime.",
        examples
    )

    sc_prompt = SelfConsistencyPrompt(cot_prompt, num_samples=3)

    test_question = "Is 21 a prime number?"
    prompts = sc_prompt.format(question=test_question)

    print("\nGenerating multiple reasoning paths:")
    print("=" * 60)

    for i, p in enumerate(prompts, 1):
        print(f"\n{p}\n")
        print("-" * 60)

    # Simulate different answers
    simulated_answers = ["No", "No", "Yes"]  # Majority: No

    aggregated = SelfConsistencyPrompt.aggregate_answers(simulated_answers)
    print(f"\nSimulated answers: {simulated_answers}")
    print(f"Aggregated answer (majority vote): {aggregated}")


def demo_instruction_optimization():
    """Demonstrate instruction clarity impact."""
    print("\n" + "=" * 60)
    print("DEMO 7: Instruction Optimization")
    print("=" * 60)

    test_input = "The movie was boring and too long."

    # Poor instruction
    poor = "Sentiment?"
    print(f"Poor instruction:\n{poor}\nInput: {test_input}\n")

    # Better instruction
    better = "What is the sentiment of this text? Answer with one word: Positive, Negative, or Neutral."
    print(f"\nBetter instruction:\n{better}\nInput: {test_input}\n")

    # Best instruction
    best = """Classify the sentiment of the following movie review.

Instructions:
1. Read the review carefully
2. Identify emotional tone and opinion
3. Classify as: Positive, Negative, or Neutral
4. Respond with only the classification label

Review: {input}

Classification:"""

    print(f"\nBest instruction (detailed):\n{best.format(input=test_input)}\n")


def demo_constraint_specification():
    """Demonstrate specifying output constraints."""
    print("\n" + "=" * 60)
    print("DEMO 8: Output Constraint Specification")
    print("=" * 60)

    base_task = "Summarize the following text."
    test_input = "Artificial intelligence has transformed many industries. Machine learning algorithms can now perform complex tasks that previously required human intelligence. Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn representations of data."

    # Without constraints
    prompt1 = f"{base_task}\n\nText: {test_input}\n\nSummary:"
    print(f"Without constraints:\n{prompt1}\n")

    # With length constraint
    prompt2 = f"{base_task} Use exactly one sentence, maximum 20 words.\n\nText: {test_input}\n\nSummary:"
    print(f"\nWith length constraint:\n{prompt2}\n")

    # With format constraint
    prompt3 = f"{base_task} Format as bullet points (max 3 points).\n\nText: {test_input}\n\nSummary:\n-"
    print(f"\nWith format constraint:\n{prompt3}\n")

    # With style constraint
    prompt4 = f"{base_task} Explain like I'm 5 years old.\n\nText: {test_input}\n\nSummary:"
    print(f"\nWith style constraint:\n{prompt4}\n")


def demo_prompt_chaining():
    """Demonstrate multi-step prompt chaining."""
    print("\n" + "=" * 60)
    print("DEMO 9: Prompt Chaining")
    print("=" * 60)

    input_text = "I bought a laptop yesterday but the screen is broken. I want a refund."

    # Step 1: Extract intent
    step1 = f"""Identify the customer's intent from this message.
Possible intents: refund_request, product_inquiry, complaint, compliment

Message: {input_text}

Intent:"""

    print(f"Step 1 - Intent Detection:\n{step1}\n")

    # Simulated output from step 1
    intent = "refund_request"

    # Step 2: Extract entities (using output from step 1)
    step2 = f"""Extract relevant entities for a {intent}.
Required entities: product, issue, timeframe

Message: {input_text}

Entities (JSON):"""

    print(f"\nStep 2 - Entity Extraction:\n{step2}\n")

    # Simulated output from step 2
    entities = {"product": "laptop", "issue": "broken screen", "timeframe": "yesterday"}

    # Step 3: Generate response (using outputs from steps 1 & 2)
    step3 = f"""Generate a customer service response for a {intent}.

Details:
- Product: {entities['product']}
- Issue: {entities['issue']}
- Purchase time: {entities['timeframe']}

Response:"""

    print(f"\nStep 3 - Response Generation:\n{step3}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: Prompt Engineering")
    print("=" * 60)

    demo_zero_shot()
    demo_few_shot()
    demo_chain_of_thought()
    demo_structured_output()
    demo_role_based()
    demo_self_consistency()
    demo_instruction_optimization()
    demo_constraint_specification()
    demo_prompt_chaining()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Zero-shot: Direct instruction (simple tasks)")
    print("2. Few-shot: Add examples (better accuracy)")
    print("3. Chain-of-thought: Step-by-step reasoning (complex problems)")
    print("4. Structured output: Specify exact format (JSON, etc.)")
    print("5. Role-based: Set persona for appropriate tone/style")
    print("6. Self-consistency: Multiple samples + voting (robustness)")
    print("7. Clear instructions > vague instructions")
    print("8. Constraints: Length, format, style specifications")
    print("9. Chaining: Break complex tasks into steps")
    print("=" * 60)
