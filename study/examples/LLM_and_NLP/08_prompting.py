"""
08. 프롬프트 엔지니어링 예제

다양한 프롬프팅 기법과 최적화 전략
"""

print("=" * 60)
print("프롬프트 엔지니어링")
print("=" * 60)


# ============================================
# 1. 프롬프트 템플릿 클래스
# ============================================
print("\n[1] 프롬프트 템플릿")
print("-" * 40)

class PromptTemplate:
    """재사용 가능한 프롬프트 템플릿"""

    def __init__(self, template: str, input_variables: list = None):
        self.template = template
        self.input_variables = input_variables or []

    def format(self, **kwargs) -> str:
        """변수를 채워 프롬프트 생성"""
        return self.template.format(**kwargs)

    @classmethod
    def from_file(cls, path: str):
        """파일에서 템플릿 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls(f.read())

# 기본 템플릿
basic_template = PromptTemplate(
    template="""You are a {role}.
Task: {task}
Input: {input}
Output:""",
    input_variables=["role", "task", "input"]
)

prompt = basic_template.format(
    role="helpful assistant",
    task="translate to Korean",
    input="Hello, world!"
)
print("기본 템플릿 예시:")
print(prompt)


# ============================================
# 2. Zero-shot vs Few-shot
# ============================================
print("\n[2] Zero-shot vs Few-shot")
print("-" * 40)

# Zero-shot 프롬프트
zero_shot = """다음 리뷰의 감성을 분석해주세요.
리뷰: "이 영화는 정말 지루했어요."
감성:"""

print("Zero-shot:")
print(zero_shot)

# Few-shot 프롬프트
few_shot = """다음 리뷰의 감성을 분석해주세요.

리뷰: "정말 재미있는 영화였어요!"
감성: 긍정

리뷰: "최악의 영화, 시간 낭비"
감성: 부정

리뷰: "그냥 그랬어요"
감성: 중립

리뷰: "이 영화는 정말 지루했어요."
감성:"""

print("\nFew-shot:")
print(few_shot)


# ============================================
# 3. Few-shot 프롬프트 빌더
# ============================================
print("\n[3] Few-shot 프롬프트 빌더")
print("-" * 40)

class FewShotPromptTemplate:
    """Few-shot 프롬프트 생성기"""

    def __init__(
        self,
        examples: list,
        example_template: str,
        prefix: str = "",
        suffix: str = "",
        separator: str = "\n\n"
    ):
        self.examples = examples
        self.example_template = example_template
        self.prefix = prefix
        self.suffix = suffix
        self.separator = separator

    def format(self, **kwargs) -> str:
        # 예시들 포맷팅
        formatted_examples = [
            self.example_template.format(**ex)
            for ex in self.examples
        ]

        # 조합
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.extend(formatted_examples)
        if self.suffix:
            parts.append(self.suffix.format(**kwargs))

        return self.separator.join(parts)

# 감성 분석 Few-shot
sentiment_examples = [
    {"text": "정말 좋아요!", "sentiment": "긍정"},
    {"text": "별로예요", "sentiment": "부정"},
    {"text": "그냥 보통이에요", "sentiment": "중립"},
]

sentiment_prompt = FewShotPromptTemplate(
    examples=sentiment_examples,
    example_template="텍스트: {text}\n감성: {sentiment}",
    prefix="다음 텍스트의 감성을 분석하세요.",
    suffix="텍스트: {text}\n감성:"
)

result = sentiment_prompt.format(text="오늘 기분이 좋네요")
print("Few-shot 감성 분석 프롬프트:")
print(result)


# ============================================
# 4. Chain-of-Thought (CoT)
# ============================================
print("\n[4] Chain-of-Thought (CoT)")
print("-" * 40)

# Zero-shot CoT
zero_shot_cot = """Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

Let's think step by step."""

print("Zero-shot CoT:")
print(zero_shot_cot)

# Few-shot CoT
few_shot_cot = """Q: There are 15 trees in the grove. Grove workers plant trees today.
   After they are done, there will be 21 trees. How many trees did they plant?

A: Let's think step by step.
1. Started with 15 trees.
2. After planting, there are 21 trees.
3. Trees planted = 21 - 15 = 6 trees.
The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive,
   how many cars are in the parking lot?

A: Let's think step by step.
1. Started with 3 cars.
2. 2 more cars arrive.
3. Total = 3 + 2 = 5 cars.
The answer is 5.

Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

A: Let's think step by step."""

print("\nFew-shot CoT:")
print(few_shot_cot)


# ============================================
# 5. 역할 기반 프롬프팅
# ============================================
print("\n[5] 역할 기반 프롬프팅")
print("-" * 40)

class RolePrompt:
    """역할 기반 프롬프트 생성"""

    ROLES = {
        "developer": """You are a senior software developer with 10 years of experience.
You write clean, efficient, and well-documented code.
You always consider edge cases and security implications.""",

        "teacher": """You are a patient and encouraging teacher.
You explain complex concepts using simple analogies.
You always check for understanding and provide examples.""",

        "reviewer": """You are a thorough code reviewer.
You check for:
- Code readability
- Potential bugs
- Performance issues
- Security vulnerabilities
You provide constructive feedback.""",

        "translator": """You are a professional translator.
You translate text while preserving:
- Original meaning
- Tone and style
- Cultural context
You provide notes for idiomatic expressions."""
    }

    @classmethod
    def get_system_prompt(cls, role: str) -> str:
        return cls.ROLES.get(role, "You are a helpful assistant.")

    @classmethod
    def create_prompt(cls, role: str, task: str) -> dict:
        return {
            "system": cls.get_system_prompt(role),
            "user": task
        }

# 역할 프롬프트 예시
prompt = RolePrompt.create_prompt(
    role="reviewer",
    task="""다음 코드를 리뷰해주세요:

def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
)
print("코드 리뷰어 역할:")
print(f"System: {prompt['system'][:100]}...")
print(f"User: {prompt['user']}")


# ============================================
# 6. 구조화된 출력 프롬프트
# ============================================
print("\n[6] 구조화된 출력")
print("-" * 40)

# JSON 출력 프롬프트
json_prompt = """다음 텍스트에서 인물과 장소를 추출해주세요.

텍스트: "철수는 서울에서 영희를 만나 부산으로 여행을 떠났다."

다음 JSON 형식으로 응답해주세요:
{
  "persons": ["인물1", "인물2"],
  "locations": ["장소1", "장소2"]
}"""

print("JSON 출력 프롬프트:")
print(json_prompt)

# 마크다운 구조화 출력
markdown_prompt = """다음 기사를 분석해주세요.

## 요약
(2-3문장으로 요약)

## 핵심 포인트
- 포인트 1
- 포인트 2
- 포인트 3

## 감성
(긍정/부정/중립 중 선택)

## 신뢰도
(높음/중간/낮음 중 선택, 이유 설명)"""

print("\n마크다운 구조화 출력:")
print(markdown_prompt)


# ============================================
# 7. 출력 파서
# ============================================
print("\n[7] 출력 파서")
print("-" * 40)

import json
import re
from typing import Any, Optional

class OutputParser:
    """LLM 출력 파싱"""

    @staticmethod
    def parse_json(text: str) -> Optional[dict]:
        """JSON 추출 및 파싱"""
        # JSON 블록 찾기
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            # JSON 객체 직접 찾기
            json_pattern = r'\{[^{}]*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def parse_list(text: str) -> list:
        """리스트 항목 추출"""
        # 번호 매긴 항목
        numbered = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
        if numbered:
            return numbered

        # 불릿 항목
        bulleted = re.findall(r'^[-*]\s*(.+)$', text, re.MULTILINE)
        return bulleted

    @staticmethod
    def parse_key_value(text: str) -> dict:
        """키-값 쌍 추출"""
        pattern = r'^([^:]+):\s*(.+)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        return {k.strip(): v.strip() for k, v in matches}

# 테스트
sample_output = """분석 결과:
- 주제: 인공지능
- 감성: 긍정
- 신뢰도: 높음

1. 첫 번째 포인트
2. 두 번째 포인트
3. 세 번째 포인트"""

parser = OutputParser()
print("리스트 파싱:", parser.parse_list(sample_output))
print("키-값 파싱:", parser.parse_key_value(sample_output))


# ============================================
# 8. Self-Consistency
# ============================================
print("\n[8] Self-Consistency")
print("-" * 40)

from collections import Counter

class SelfConsistency:
    """Self-Consistency: 여러 추론 경로의 다수결"""

    def __init__(self, model_fn, n_samples: int = 5):
        self.model_fn = model_fn
        self.n_samples = n_samples

    def generate_with_consistency(self, prompt: str) -> tuple:
        """여러 샘플 생성 후 다수결"""
        responses = []

        for _ in range(self.n_samples):
            # temperature > 0 으로 다양한 응답 생성
            response = self.model_fn(prompt, temperature=0.7)
            answer = self._extract_answer(response)
            responses.append(answer)

        # 다수결
        counter = Counter(responses)
        most_common = counter.most_common(1)[0]

        return most_common[0], most_common[1] / self.n_samples

    def _extract_answer(self, response: str) -> str:
        """응답에서 최종 답 추출"""
        # "The answer is X" 패턴
        match = re.search(r'answer is[:\s]*(\d+)', response, re.IGNORECASE)
        if match:
            return match.group(1)

        # 마지막 숫자
        numbers = re.findall(r'\d+', response)
        return numbers[-1] if numbers else response

# 모의 함수
def mock_model(prompt, temperature=0.7):
    import random
    # 실제로는 LLM 호출
    answers = ["42", "42", "42", "41", "42"]
    return f"The answer is {random.choice(answers)}"

sc = SelfConsistency(mock_model, n_samples=5)
print("Self-Consistency 예시 (모의):")
print("여러 추론 경로 생성 후 다수결로 최종 답 선택")


# ============================================
# 9. ReAct 패턴
# ============================================
print("\n[9] ReAct (Reasoning + Acting)")
print("-" * 40)

react_prompt = """Answer the following question using this format:

Question: {question}

Thought: (your reasoning about what to do)
Action: (one of: Search[query], Calculate[expression], Lookup[term], Finish[answer])
Observation: (result of the action)

Repeat Thought/Action/Observation until you have the answer.

Example:
Question: What is the capital of the country where the Eiffel Tower is located?

Thought: I need to find where the Eiffel Tower is located.
Action: Search[Eiffel Tower location]
Observation: The Eiffel Tower is located in Paris, France.

Thought: Now I know it's in France. I need to find the capital of France.
Action: Search[capital of France]
Observation: The capital of France is Paris.

Thought: I have the answer now.
Action: Finish[Paris]

Now answer:
Question: {question}
"""

print("ReAct 프롬프트 패턴:")
print(react_prompt.format(question="What year was Python created?"))


# ============================================
# 10. Tree of Thoughts
# ============================================
print("\n[10] Tree of Thoughts")
print("-" * 40)

class TreeOfThoughts:
    """Tree of Thoughts: 여러 사고 경로 탐색"""

    def __init__(self, model_fn, evaluator_fn):
        self.model_fn = model_fn
        self.evaluator_fn = evaluator_fn

    def solve(self, problem: str, depth: int = 3, branches: int = 3) -> str:
        """트리 탐색으로 문제 해결"""
        thoughts = self._generate_thoughts(problem, branches)

        # 각 생각 평가
        scored_thoughts = [
            (thought, self.evaluator_fn(problem, thought))
            for thought in thoughts
        ]

        # 상위 생각 선택
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        best_thoughts = scored_thoughts[:2]

        if depth > 1:
            # 재귀적으로 확장
            for thought, _ in best_thoughts:
                extended = self.solve(
                    f"{problem}\n\nPartial solution: {thought}",
                    depth - 1,
                    branches
                )
                thoughts.append(extended)

        # 최종 선택
        return scored_thoughts[0][0]

    def _generate_thoughts(self, problem: str, n: int) -> list:
        """n개의 서로 다른 접근법 생성"""
        prompt = f"""Problem: {problem}

Generate {n} different approaches to solve this problem.
Each approach should be a distinct strategy.

Approach 1:"""

        # 실제로는 LLM 호출
        return [f"Approach {i+1}: ..." for i in range(n)]

print("Tree of Thoughts 패턴:")
print("- 여러 사고 경로를 트리 형태로 탐색")
print("- 각 노드(생각)를 평가하고 유망한 경로 확장")
print("- 복잡한 추론 문제에 효과적")


# ============================================
# 11. 프롬프트 최적화 전략
# ============================================
print("\n[11] 프롬프트 최적화")
print("-" * 40)

optimization_strategies = """
프롬프트 최적화 전략:

1. 명확성 (Clarity)
   Bad:  "텍스트를 정리해줘"
   Good: "다음 텍스트를 3문장으로 요약하고, 핵심 키워드 5개를 추출해주세요."

2. 구체성 (Specificity)
   Bad:  "좋은 코드를 작성해줘"
   Good: "Python 3.10+, 타입 힌트 사용, PEP 8 준수, 에러 처리 포함"

3. 제약 조건 (Constraints)
   - 출력 길이: "100단어 이내로"
   - 출력 형식: "JSON 형식으로"
   - 스타일: "공식적인 어조로"

4. 예시 제공 (Examples)
   - 원하는 출력의 예시 1-3개 제공
   - 형식과 스타일 명확히 전달

5. 단계별 분해 (Decomposition)
   - 복잡한 태스크를 작은 단계로 분해
   - 각 단계별로 명확한 지시

6. 네거티브 프롬프팅 (Negative Prompting)
   - "~하지 마세요" 지시 추가
   - 원치 않는 출력 방지
"""
print(optimization_strategies)


# ============================================
# 12. 프롬프트 A/B 테스트
# ============================================
print("\n[12] 프롬프트 A/B 테스트")
print("-" * 40)

class PromptABTest:
    """프롬프트 A/B 테스트 프레임워크"""

    def __init__(self, model_fn, evaluator_fn):
        self.model_fn = model_fn
        self.evaluator_fn = evaluator_fn

    def run_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: list,
        n_trials: int = 1
    ) -> dict:
        """A/B 테스트 실행"""
        results = {"A": 0, "B": 0, "tie": 0}
        details = []

        for case in test_cases:
            scores_a = []
            scores_b = []

            for _ in range(n_trials):
                # 프롬프트 A
                response_a = self.model_fn(prompt_a.format(**case))
                score_a = self.evaluator_fn(response_a, case.get("expected"))
                scores_a.append(score_a)

                # 프롬프트 B
                response_b = self.model_fn(prompt_b.format(**case))
                score_b = self.evaluator_fn(response_b, case.get("expected"))
                scores_b.append(score_b)

            avg_a = sum(scores_a) / len(scores_a)
            avg_b = sum(scores_b) / len(scores_b)

            if avg_a > avg_b:
                results["A"] += 1
                winner = "A"
            elif avg_b > avg_a:
                results["B"] += 1
                winner = "B"
            else:
                results["tie"] += 1
                winner = "tie"

            details.append({
                "case": case,
                "score_a": avg_a,
                "score_b": avg_b,
                "winner": winner
            })

        return {
            "summary": results,
            "details": details,
            "winner": "A" if results["A"] > results["B"] else "B"
        }

print("프롬프트 A/B 테스트 프레임워크")
print("- 두 프롬프트의 성능 비교")
print("- 다양한 테스트 케이스에서 평가")
print("- 통계적으로 유의미한 결과 도출")


# ============================================
# 13. 도메인별 프롬프트 템플릿
# ============================================
print("\n[13] 도메인별 프롬프트 템플릿")
print("-" * 40)

PROMPT_TEMPLATES = {
    "classification": """Classify the following text into one of these categories: {categories}

Text: {text}

Think step by step:
1. Identify key features of the text
2. Match features to categories
3. Select the best category

Category:""",

    "summarization": """Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.
Maintain the original tone.

Text:
{text}

Summary:""",

    "qa": """Answer the question based on the context below.
If the answer cannot be found in the context, say "I don't know."
Do not make up information.

Context: {context}

Question: {question}

Answer:""",

    "code_generation": """Write a {language} function that {task_description}.

Requirements:
{requirements}

Include:
- Type hints (if applicable)
- Docstring
- Example usage
- Error handling

Code:
```{language}
""",

    "translation": """Translate the following {source_lang} text to {target_lang}.
Preserve the original tone and meaning.
For idiomatic expressions, provide a note.

Original ({source_lang}):
{text}

Translation ({target_lang}):""",

    "extraction": """Extract the following information from the text:
{fields}

Text:
{text}

Output as JSON:
{{
{json_template}
}}"""
}

# 사용 예시
classification_prompt = PROMPT_TEMPLATES["classification"].format(
    categories="긍정, 부정, 중립",
    text="오늘 날씨가 정말 좋네요!"
)
print("분류 프롬프트:")
print(classification_prompt[:200] + "...")


# ============================================
# 14. 프롬프트 체이닝
# ============================================
print("\n[14] 프롬프트 체이닝")
print("-" * 40)

class PromptChain:
    """프롬프트를 연결하여 복잡한 태스크 수행"""

    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.steps = []

    def add_step(self, name: str, prompt_template: str, parser=None):
        """체인에 단계 추가"""
        self.steps.append({
            "name": name,
            "template": prompt_template,
            "parser": parser
        })
        return self

    def run(self, initial_input: dict) -> dict:
        """체인 실행"""
        context = initial_input.copy()
        results = {}

        for step in self.steps:
            # 프롬프트 생성
            prompt = step["template"].format(**context)

            # LLM 호출
            response = self.model_fn(prompt)

            # 파싱 (선택적)
            if step["parser"]:
                response = step["parser"](response)

            # 결과 저장
            results[step["name"]] = response
            context[step["name"]] = response

        return results

# 체인 예시
chain = PromptChain(lambda x: "Mock response")
chain.add_step(
    "summary",
    "Summarize this text: {text}"
).add_step(
    "keywords",
    "Extract keywords from: {summary}"
).add_step(
    "title",
    "Create a title based on keywords: {keywords}"
)

print("프롬프트 체이닝 예시:")
print("1. 텍스트 요약")
print("2. 키워드 추출")
print("3. 제목 생성")
print("→ 각 단계의 출력이 다음 단계의 입력으로 사용")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("프롬프트 엔지니어링 정리")
print("=" * 60)

summary = """
프롬프팅 기법 선택 가이드:

| 상황                | 추천 기법           |
|---------------------|---------------------|
| 간단한 태스크       | Zero-shot           |
| 특정 형식 필요      | Few-shot + 형식지정 |
| 복잡한 추론         | Chain-of-Thought    |
| 신뢰성 필요         | Self-Consistency    |
| 도구 사용 필요      | ReAct               |
| 매우 복잡한 문제    | Tree of Thoughts    |

핵심 원칙:
1. 명확하고 구체적인 지시
2. 적절한 예시 제공
3. 출력 형식 명시
4. 단계별 사고 유도
5. 반복적인 개선과 테스트

프롬프트 구조:
    [시스템 지시] + [컨텍스트] + [태스크] + [출력 형식]
"""
print(summary)
