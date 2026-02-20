"""
12. 실전 챗봇 예제

RAG 기반 대화형 AI 시스템
"""

print("=" * 60)
print("실전 챗봇")
print("=" * 60)


# ============================================
# 1. 간단한 대화 챗봇 (메모리)
# ============================================
print("\n[1] 간단한 대화 챗봇")
print("-" * 40)

class SimpleChatbot:
    """히스토리를 유지하는 간단한 챗봇"""

    def __init__(self, system_prompt="You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.history = []

    def chat(self, user_message):
        """사용자 메시지 처리 (LLM 호출 시뮬레이션)"""
        # 히스토리에 추가
        self.history.append({"role": "user", "content": user_message})

        # 실제로는 LLM 호출
        # response = llm.invoke(messages)
        response = f"[응답] {user_message}에 대한 답변입니다."

        self.history.append({"role": "assistant", "content": response})
        return response

    def get_messages(self):
        """전체 메시지 구성"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages

    def clear_history(self):
        self.history = []

# 테스트
bot = SimpleChatbot()
print(bot.chat("안녕하세요"))
print(bot.chat("오늘 날씨 어때요?"))
print(f"히스토리 길이: {len(bot.history)}")


# ============================================
# 2. RAG 챗봇
# ============================================
print("\n[2] RAG 챗봇")
print("-" * 40)

import numpy as np

class RAGChatbot:
    """문서 기반 RAG 챗봇"""

    def __init__(self, documents):
        self.documents = documents
        self.history = []
        # 가상 임베딩 (실제로는 모델 사용)
        self.embeddings = np.random.randn(len(documents), 128)

    def retrieve(self, query, top_k=2):
        """관련 문서 검색"""
        query_emb = np.random.randn(128)
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def chat(self, question):
        """RAG 기반 답변 생성"""
        # 검색
        relevant_docs = self.retrieve(question)
        context = "\n".join(relevant_docs)

        # 프롬프트 구성
        prompt = f"""Context:
{context}

History:
{self._format_history()}

Question: {question}

Answer:"""

        # 실제로는 LLM 호출
        response = f"[컨텍스트 기반 응답] {relevant_docs[0][:50]}..."

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def _format_history(self, max_turns=3):
        recent = self.history[-max_turns*2:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

# 테스트
documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a type of artificial intelligence.",
    "Deep learning uses neural networks with many layers."
]

rag_bot = RAGChatbot(documents)
print(rag_bot.chat("What is Python?"))
print(rag_bot.chat("Tell me more about it"))


# ============================================
# 3. 의도 분류
# ============================================
print("\n[3] 의도 분류")
print("-" * 40)

class IntentClassifier:
    """규칙 기반 의도 분류 (실제로는 LLM 사용)"""

    def __init__(self):
        self.intents = {
            "greeting": ["hello", "hi", "hey", "안녕"],
            "goodbye": ["bye", "goodbye", "잘가"],
            "help": ["help", "도움", "how do i"],
            "question": ["what", "why", "how", "when", "무엇", "왜"]
        }

    def classify(self, text):
        text_lower = text.lower()
        for intent, keywords in self.intents.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return "general"

classifier = IntentClassifier()
test_texts = ["Hello!", "What is AI?", "Goodbye", "Help me please"]
for text in test_texts:
    intent = classifier.classify(text)
    print(f"  [{intent}] {text}")


# ============================================
# 4. 대화 상태 관리
# ============================================
print("\n[4] 대화 상태 관리")
print("-" * 40)

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any

class State(Enum):
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    DONE = "done"

@dataclass
class ConversationState:
    state: State = State.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)

class StatefulBot:
    def __init__(self):
        self.context = ConversationState()
        self.required_slots = ["name", "email"]

    def process(self, message):
        if self.context.state == State.GREETING:
            self.context.state = State.COLLECTING
            return "안녕하세요! 이름을 알려주세요."

        elif self.context.state == State.COLLECTING:
            # 슬롯 추출 (간단한 예시)
            if "name" not in self.context.slots:
                self.context.slots["name"] = message
                return "이메일 주소를 알려주세요."
            elif "email" not in self.context.slots:
                self.context.slots["email"] = message
                self.context.state = State.CONFIRMING
                return f"확인: {self.context.slots}. 맞습니까? (예/아니오)"

        elif self.context.state == State.CONFIRMING:
            if "예" in message.lower() or "yes" in message.lower():
                self.context.state = State.DONE
                return "감사합니다! 처리 완료되었습니다."
            else:
                self.context = ConversationState()
                return "처음부터 다시 시작합니다. 이름을 알려주세요."

        return "무엇을 도와드릴까요?"

# 테스트
stateful_bot = StatefulBot()
print(stateful_bot.process("시작"))
print(stateful_bot.process("홍길동"))
print(stateful_bot.process("hong@example.com"))
print(stateful_bot.process("예"))


# ============================================
# 5. OpenAI 챗봇 (코드 예시)
# ============================================
print("\n[5] OpenAI 챗봇 (코드)")
print("-" * 40)

openai_bot_code = '''
from openai import OpenAI

class OpenAIChatbot:
    def __init__(self, system_prompt="You are a helpful assistant."):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.history = []

    def chat(self, message):
        # 메시지 구성
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        # API 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_msg = response.choices[0].message.content

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def chat_stream(self, message):
        """스트리밍 응답"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})
'''
print(openai_bot_code)


# ============================================
# 6. FastAPI 서버 (코드)
# ============================================
print("\n[6] FastAPI 서버 (코드)")
print("-" * 40)

fastapi_code = '''
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = OpenAIChatbot()

    bot = sessions[request.session_id]
    response = bot.chat(request.message)

    return {"response": response}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}

# 실행: uvicorn main:app --reload
'''
print(fastapi_code)


# ============================================
# 7. Gradio UI (코드)
# ============================================
print("\n[7] Gradio UI (코드)")
print("-" * 40)

gradio_code = '''
import gradio as gr

def respond(message, history):
    # 챗봇 응답 생성
    response = bot.chat(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="AI Chatbot",
    description="Ask me anything!",
    examples=["Hello!", "What is AI?"],
    theme="soft"
)

demo.launch()
'''
print(gradio_code)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("챗봇 정리")
print("=" * 60)

summary = """
챗봇 구성요소:
    1. 대화 히스토리 관리
    2. 의도 분류
    3. 슬롯 추출
    4. 상태 관리
    5. RAG (문서 기반)
    6. LLM 호출

핵심 패턴:
    # 기본 대화
    messages = [system] + history + [user_message]
    response = llm.invoke(messages)

    # RAG
    context = retrieve(query)
    response = llm.invoke(context + query)

    # 스트리밍
    for chunk in llm.stream(messages):
        yield chunk

배포:
    - FastAPI: REST API 서버
    - Gradio: 빠른 UI 프로토타입
    - Streamlit: 대시보드 스타일
"""
print(summary)
