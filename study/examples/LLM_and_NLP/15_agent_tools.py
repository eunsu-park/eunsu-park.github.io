"""
15. LLM 에이전트 (LLM Agents) 예제

ReAct 패턴, Tool Use, LangChain Agent 실습
"""

import json
import re
from typing import Dict, List, Callable, Any

print("=" * 60)
print("LLM 에이전트 (LLM Agents)")
print("=" * 60)


# ============================================
# 1. 도구 정의
# ============================================
print("\n[1] 도구 (Tools) 정의")
print("-" * 40)

class Tool:
    """도구 기본 클래스"""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_str: str) -> str:
        """도구 실행"""
        try:
            return str(self.func(input_str))
        except Exception as e:
            return f"Error: {str(e)}"


# 도구 구현
def calculator(expression: str) -> float:
    """안전한 수학 계산"""
    # 간단한 안전 체크
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)

def search(query: str) -> str:
    """검색 시뮬레이션"""
    # 실제로는 API 호출
    results = {
        "파이썬 창시자": "파이썬은 1991년 귀도 반 로섬(Guido van Rossum)이 개발했습니다.",
        "인공지능": "인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템입니다.",
        "서울 인구": "서울의 인구는 약 950만 명입니다 (2024년 기준).",
    }
    for key, value in results.items():
        if key in query:
            return value
    return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."

def get_weather(city: str) -> str:
    """날씨 조회 시뮬레이션"""
    weather_data = {
        "서울": {"temp": 25, "condition": "맑음"},
        "부산": {"temp": 28, "condition": "흐림"},
        "제주": {"temp": 27, "condition": "구름 조금"},
    }
    if city in weather_data:
        data = weather_data[city]
        return f"{city} 날씨: {data['temp']}도, {data['condition']}"
    return f"{city}의 날씨 정보를 찾을 수 없습니다."


# 도구 등록
tools = [
    Tool("calculator", "수학 계산. 입력: 수학 표현식 (예: '2 + 3 * 4')", calculator),
    Tool("search", "정보 검색. 입력: 검색어", search),
    Tool("get_weather", "날씨 조회. 입력: 도시 이름", get_weather),
]

print("사용 가능한 도구:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")


# ============================================
# 2. ReAct 패턴 시뮬레이션
# ============================================
print("\n[2] ReAct 패턴 시뮬레이션")
print("-" * 40)

class ReActAgent:
    """ReAct (Reasoning + Acting) 에이전트"""

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
        self.history = []

    def think(self, question: str, observations: List[str]) -> Dict[str, str]:
        """
        생각 단계 (실제로는 LLM 호출)
        여기서는 규칙 기반으로 시뮬레이션
        """
        question_lower = question.lower()

        # 규칙 기반 의사결정 (실제로는 LLM이 수행)
        if "날씨" in question_lower:
            # 도시 추출
            cities = ["서울", "부산", "제주"]
            for city in cities:
                if city in question:
                    return {
                        "thought": f"{city}의 날씨를 확인해야 합니다.",
                        "action": "get_weather",
                        "action_input": city
                    }

        if "계산" in question_lower or any(op in question for op in ["+", "-", "*", "/"]):
            # 수식 추출
            numbers = re.findall(r'[\d\+\-\*\/\.\(\)\s]+', question)
            if numbers:
                expr = numbers[0].strip()
                return {
                    "thought": f"'{expr}'을 계산해야 합니다.",
                    "action": "calculator",
                    "action_input": expr
                }

        if any(keyword in question_lower for keyword in ["누구", "무엇", "어디", "검색"]):
            return {
                "thought": f"'{question}'에 대해 검색해야 합니다.",
                "action": "search",
                "action_input": question
            }

        # 이미 충분한 정보가 있으면 최종 답변
        if observations:
            return {
                "thought": "충분한 정보를 수집했습니다.",
                "final_answer": " ".join(observations)
            }

        return {
            "thought": "질문을 이해하지 못했습니다.",
            "final_answer": "죄송합니다, 질문을 처리할 수 없습니다."
        }

    def act(self, action: str, action_input: str) -> str:
        """행동 단계"""
        if action in self.tools:
            return self.tools[action].run(action_input)
        return f"Error: Unknown tool '{action}'"

    def run(self, question: str, max_steps: int = 5) -> str:
        """에이전트 실행"""
        observations = []

        print(f"\n질문: {question}")
        print("-" * 30)

        for step in range(max_steps):
            # 생각
            result = self.think(question, observations)

            print(f"\n[Step {step + 1}]")
            print(f"Thought: {result.get('thought', '')}")

            # 최종 답변 확인
            if "final_answer" in result:
                print(f"Final Answer: {result['final_answer']}")
                return result["final_answer"]

            # 행동
            action = result.get("action")
            action_input = result.get("action_input")

            if action:
                print(f"Action: {action}")
                print(f"Action Input: {action_input}")

                observation = self.act(action, action_input)
                observations.append(observation)
                print(f"Observation: {observation}")

        return "최대 단계 도달"


# 테스트
agent = ReActAgent(tools)

questions = [
    "서울 날씨 어때?",
    "15 + 27 * 3을 계산해줘",
    "파이썬 창시자가 누구야?",
]

for q in questions:
    result = agent.run(q)


# ============================================
# 3. Function Calling 형식
# ============================================
print("\n" + "=" * 60)
print("[3] Function Calling 형식")
print("-" * 40)

# OpenAI Function Calling 형식
function_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "도시 이름 (예: Seoul, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "온도 단위"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "웹에서 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색어"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

print("Function Calling 정의:")
print(json.dumps(function_definitions, indent=2, ensure_ascii=False))


# ============================================
# 4. 멀티 에이전트 시뮬레이션
# ============================================
print("\n" + "=" * 60)
print("[4] 멀티 에이전트 시뮬레이션")
print("-" * 40)

class ResearcherAgent:
    """연구 에이전트"""

    def __init__(self):
        self.name = "Researcher"

    def research(self, topic: str) -> str:
        """주제 연구 (시뮬레이션)"""
        research_db = {
            "인공지능": "AI는 1956년 다트머스 회의에서 시작되었으며, "
                       "머신러닝, 딥러닝, 자연어처리 등으로 발전했습니다.",
            "파이썬": "파이썬은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어로, "
                     "간결한 문법과 풍부한 라이브러리가 특징입니다.",
        }
        for key, value in research_db.items():
            if key in topic:
                return value
        return f"{topic}에 대한 연구 결과를 찾을 수 없습니다."


class WriterAgent:
    """작문 에이전트"""

    def __init__(self):
        self.name = "Writer"

    def write(self, research_results: str, style: str = "formal") -> str:
        """문서 작성 (시뮬레이션)"""
        if style == "formal":
            return f"## 연구 보고서\n\n{research_results}\n\n본 내용은 연구 결과를 바탕으로 작성되었습니다."
        else:
            return f"# 요약\n\n{research_results}"


class ReviewerAgent:
    """검토 에이전트"""

    def __init__(self):
        self.name = "Reviewer"

    def review(self, document: str) -> str:
        """문서 검토 (시뮬레이션)"""
        issues = []
        if len(document) < 100:
            issues.append("내용이 너무 짧습니다.")
        if "참고문헌" not in document:
            issues.append("참고문헌이 없습니다.")

        if issues:
            return "검토 결과:\n" + "\n".join(f"- {issue}" for issue in issues)
        return "검토 결과: 수정 필요 없음"


class MultiAgentSystem:
    """멀티 에이전트 시스템"""

    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def create_document(self, topic: str) -> str:
        """문서 생성 파이프라인"""
        print(f"\n토픽: {topic}")

        # 1. 연구
        print(f"\n[{self.researcher.name}] 연구 중...")
        research = self.researcher.research(topic)
        print(f"연구 결과: {research[:50]}...")

        # 2. 작성
        print(f"\n[{self.writer.name}] 작성 중...")
        document = self.writer.write(research)
        print(f"작성 결과: {document[:50]}...")

        # 3. 검토
        print(f"\n[{self.reviewer.name}] 검토 중...")
        review = self.reviewer.review(document)
        print(f"검토 결과: {review}")

        return document


# 테스트
system = MultiAgentSystem()
doc = system.create_document("인공지능의 역사")
print(f"\n최종 문서:\n{doc}")


# ============================================
# 5. LangChain Agent 코드 예시
# ============================================
print("\n" + "=" * 60)
print("[5] LangChain Agent 코드 예시")
print("-" * 40)

langchain_code = '''
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 도구 정의
@tool
def calculator(expression: str) -> str:
    """수학 계산. 입력: 수학 표현식 (예: '2 + 3 * 4')"""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """현재 시간 반환"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]

# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI입니다."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 에이전트 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "현재 시간과 15 + 27의 결과를 알려줘"})
print(result["output"])
'''
print(langchain_code)


# ============================================
# 6. 자율 에이전트 (AutoGPT 스타일)
# ============================================
print("\n" + "=" * 60)
print("[6] 자율 에이전트 (AutoGPT 스타일)")
print("-" * 40)

class AutoGPTLikeAgent:
    """AutoGPT 스타일 자율 에이전트"""

    def __init__(self, tools: List[Tool], goals: List[str]):
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self) -> Dict[str, Any]:
        """다음 작업 계획 (규칙 기반 시뮬레이션)"""
        # 아직 달성하지 않은 목표 확인
        remaining_goals = [g for g in self.goals if g not in self.completed_tasks]

        if not remaining_goals:
            return {"task": "COMPLETE", "summary": "모든 목표 달성!"}

        current_goal = remaining_goals[0]

        # 간단한 규칙 기반 계획
        if "날씨" in current_goal:
            return {
                "task": current_goal,
                "tool": "get_weather",
                "input": "서울"
            }
        elif "계산" in current_goal or "숫자" in current_goal:
            return {
                "task": current_goal,
                "tool": "calculator",
                "input": "100 + 200"
            }
        else:
            return {
                "task": current_goal,
                "tool": "search",
                "input": current_goal
            }

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """계획 실행"""
        if plan.get("task") == "COMPLETE":
            return {"status": "complete", "summary": plan["summary"]}

        tool_name = plan.get("tool")
        tool_input = plan.get("input")

        if tool_name in self.tools:
            result = self.tools[tool_name].run(tool_input)
            return {"status": "success", "result": result}

        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    def run(self, max_iterations: int = 5) -> str:
        """에이전트 실행"""
        print(f"\n목표: {self.goals}")
        print("-" * 30)

        for i in range(max_iterations):
            print(f"\n=== Iteration {i + 1} ===")

            # 계획
            plan = self.plan()
            print(f"Plan: {plan}")

            if plan.get("task") == "COMPLETE":
                print(f"완료: {plan['summary']}")
                return plan["summary"]

            # 실행
            result = self.execute(plan)
            print(f"Result: {result}")

            # 메모리 및 완료 목록 업데이트
            self.memory.append({"plan": plan, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(plan["task"])

        return "최대 반복 횟수 도달"


# 테스트
auto_agent = AutoGPTLikeAgent(
    tools=tools,
    goals=["서울 날씨 확인", "간단한 계산"]
)
auto_agent.run()


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("LLM 에이전트 정리")
print("=" * 60)

summary = """
LLM 에이전트 핵심 개념:

1. ReAct 패턴:
   Thought -> Action -> Observation -> ... -> Final Answer

2. 도구 정의:
   - 이름: 고유 식별자
   - 설명: LLM이 사용 시점 판단용
   - 함수: 실제 실행 로직

3. Function Calling (OpenAI):
   - tools 파라미터로 함수 정의
   - tool_choice="auto"로 자동 선택
   - 결과를 role="tool"로 전달

4. 멀티 에이전트:
   - 역할 분담 (연구, 작성, 검토)
   - 파이프라인 연결
   - 협업 프로토콜

5. 자율 에이전트:
   - 목표 기반 계획
   - 메모리 유지
   - 반복적 실행

에이전트 설계 체크리스트:
□ 명확한 도구 설명
□ 에러 처리
□ 무한 루프 방지 (max_steps)
□ 안전 장치 (위험한 작업 제한)
□ 로깅 및 디버깅
"""
print(summary)
