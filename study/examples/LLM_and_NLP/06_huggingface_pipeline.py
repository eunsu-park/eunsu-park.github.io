"""
06. HuggingFace Pipeline 예제

다양한 NLP 태스크를 Pipeline으로 수행
"""

print("=" * 60)
print("HuggingFace Pipeline")
print("=" * 60)

try:
    from transformers import pipeline

    # ============================================
    # 1. 감성 분석
    # ============================================
    print("\n[1] 감성 분석 (Sentiment Analysis)")
    print("-" * 40)

    classifier = pipeline("sentiment-analysis")

    texts = [
        "I love this product! It's amazing.",
        "This is terrible. I'm very disappointed.",
        "It's okay, nothing special."
    ]

    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"[{result['label']}] ({result['score']:.2%}) {text}")


    # ============================================
    # 2. 텍스트 생성
    # ============================================
    print("\n[2] 텍스트 생성 (Text Generation)")
    print("-" * 40)

    generator = pipeline("text-generation", model="gpt2")

    prompt = "Artificial intelligence will"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    print(f"프롬프트: {prompt}")
    print(f"생성: {result[0]['generated_text']}")


    # ============================================
    # 3. 질의응답 (QA)
    # ============================================
    print("\n[3] 질의응답 (Question Answering)")
    print("-" * 40)

    qa = pipeline("question-answering")

    context = """
    Python is a high-level, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python was created by Guido van Rossum and first released in 1991.
    """

    questions = [
        "Who created Python?",
        "When was Python released?",
        "What does Python emphasize?"
    ]

    for question in questions:
        result = qa(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.2%})")
        print()


    # ============================================
    # 4. 개체명 인식 (NER)
    # ============================================
    print("\n[4] 개체명 인식 (NER)")
    print("-" * 40)

    ner = pipeline("ner", grouped_entities=True)

    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
    entities = ner(text)

    print(f"텍스트: {text}")
    print("개체:")
    for entity in entities:
        print(f"  [{entity['entity_group']}] {entity['word']} ({entity['score']:.2%})")


    # ============================================
    # 5. 텍스트 요약
    # ============================================
    print("\n[5] 텍스트 요약 (Summarization)")
    print("-" * 40)

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    article = """
    Machine learning is a type of artificial intelligence that allows software applications
    to become more accurate at predicting outcomes without being explicitly programmed to do so.
    Machine learning algorithms use historical data as input to predict new output values.
    Recommendation engines are a common use case for machine learning. Other popular uses include
    fraud detection, spam filtering, malware threat detection, business process automation and
    predictive maintenance.
    """

    summary = summarizer(article, max_length=50, min_length=20)
    print(f"원문 길이: {len(article)} chars")
    print(f"요약: {summary[0]['summary_text']}")


    # ============================================
    # 6. Zero-shot 분류
    # ============================================
    print("\n[6] Zero-shot 분류")
    print("-" * 40)

    classifier = pipeline("zero-shot-classification")

    texts = [
        "I need to book a flight to New York",
        "Can you recommend a good restaurant?",
        "How do I reset my password?"
    ]
    labels = ["travel", "food", "tech_support"]

    for text in texts:
        result = classifier(text, candidate_labels=labels)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        print(f"[{top_label}] ({top_score:.2%}) {text}")


    # ============================================
    # 7. Fill-Mask (BERT MLM)
    # ============================================
    print("\n[7] Fill-Mask")
    print("-" * 40)

    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    text = "Python is a [MASK] programming language."
    results = fill_mask(text)

    print(f"입력: {text}")
    print("예측:")
    for r in results[:3]:
        print(f"  {r['token_str']}: {r['score']:.2%}")


    # ============================================
    # 8. 번역
    # ============================================
    print("\n[8] 번역 (Translation)")
    print("-" * 40)

    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

    texts = [
        "Hello, how are you?",
        "Machine learning is amazing."
    ]

    for text in texts:
        result = translator(text)
        print(f"EN: {text}")
        print(f"FR: {result[0]['translation_text']}")
        print()


    # ============================================
    # 정리
    # ============================================
    print("=" * 60)
    print("Pipeline 정리")
    print("=" * 60)

    summary = """
주요 Pipeline:
    - sentiment-analysis: 감성 분석
    - text-generation: 텍스트 생성
    - question-answering: 질의응답
    - ner: 개체명 인식
    - summarization: 요약
    - zero-shot-classification: 레이블 없는 분류
    - fill-mask: 마스크 예측
    - translation: 번역

사용법:
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this!")
"""
    print(summary)

except ImportError as e:
    print(f"필요 패키지 미설치: {e}")
    print("pip install transformers torch")
