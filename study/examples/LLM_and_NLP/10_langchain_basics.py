"""
10. LangChain 기초 예제

LangChain을 사용한 LLM 애플리케이션
"""

print("=" * 60)
print("LangChain 기초")
print("=" * 60)

# ============================================
# 1. LangChain 구조 (코드 예시)
# ============================================
print("\n[1] LangChain 기본 구조")
print("-" * 40)

langchain_basic = '''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# 출력 파서
parser = StrOutputParser()

# 체인 구성 (LCEL)
chain = prompt | llm | parser

# 실행
result = chain.invoke({"topic": "programming"})
print(result)
'''
print(langchain_basic)


# ============================================
# 2. 프롬프트 템플릿
# ============================================
print("\n[2] 프롬프트 템플릿 예제")
print("-" * 40)

try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

    # 기본 템플릿
    template = PromptTemplate(
        input_variables=["product"],
        template="Write a marketing slogan for {product}."
    )
    print(f"기본 템플릿: {template.format(product='smartphone')}")

    # Chat 템플릿
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    messages = chat_template.format_messages(question="What is Python?")
    print(f"\nChat 템플릿: {messages}")

except ImportError:
    print("langchain 미설치 (pip install langchain langchain-core)")


# ============================================
# 3. Few-shot 프롬프트
# ============================================
print("\n[3] Few-shot 프롬프트")
print("-" * 40)

fewshot_code = '''
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_template = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\\nAntonym: {antonym}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of each word:",
    suffix="Word: {input}\\nAntonym:",
    input_variables=["input"]
)

prompt = few_shot_prompt.format(input="big")
'''
print(fewshot_code)


# ============================================
# 4. 출력 파서
# ============================================
print("\n[4] 출력 파서")
print("-" * 40)

parser_code = '''
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")

parser = JsonOutputParser(pydantic_object=Person)

# 프롬프트에 형식 지시 추가
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person info. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=format_instructions)

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 25 years old"})
# {'name': 'John', 'age': 25}
'''
print(parser_code)


# ============================================
# 5. 체인 (LCEL)
# ============================================
print("\n[5] LCEL 체인")
print("-" * 40)

lcel_code = '''
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 순차 체인
chain = prompt | llm | parser

# 병렬 체인
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain
)

# 분기 체인
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 실행
result = chain.invoke({"question": "What is AI?"})
'''
print(lcel_code)


# ============================================
# 6. RAG 체인
# ============================================
print("\n[6] RAG 체인")
print("-" * 40)

rag_chain_code = '''
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 벡터 스토어
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG 프롬프트
template = """Answer based on context:
Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 문서 포맷
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# RAG 체인
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# 실행
answer = rag_chain.invoke("What is machine learning?")
'''
print(rag_chain_code)


# ============================================
# 7. 에이전트
# ============================================
print("\n[7] 에이전트")
print("-" * 40)

agent_code = '''
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool

# 커스텀 도구
@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

tools = [calculate, search]

# ReAct 에이전트
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = executor.invoke({"input": "What is 2 + 2?"})
'''
print(agent_code)


# ============================================
# 8. 메모리
# ============================================
print("\n[8] 대화 메모리")
print("-" * 40)

memory_code = '''
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 메모리
memory = ConversationBufferMemory()

# 대화 체인
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 대화
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")
# "Your name is John"

# LCEL 메모리
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
'''
print(memory_code)


# ============================================
# 9. 간단한 실행 예제
# ============================================
print("\n[9] 실행 가능한 예제")
print("-" * 40)

try:
    from langchain_core.prompts import PromptTemplate

    # 프롬프트 템플릿만 테스트
    template = PromptTemplate.from_template(
        "Translate '{text}' to {language}."
    )

    # 포맷팅
    prompt = template.format(text="Hello", language="Korean")
    print(f"생성된 프롬프트: {prompt}")

    # 입력 변수
    print(f"입력 변수: {template.input_variables}")

except ImportError:
    print("langchain-core 미설치")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("LangChain 정리")
print("=" * 60)

summary = """
핵심 패턴:
    # 기본 체인
    chain = prompt | llm | output_parser

    # RAG 체인
    rag = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # 에이전트
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

주요 컴포넌트:
    - PromptTemplate: 프롬프트 구성
    - ChatOpenAI: LLM 래퍼
    - OutputParser: 출력 파싱
    - Retriever: 문서 검색
    - Memory: 대화 히스토리
    - Agent: 도구 사용
"""
print(summary)
