"""
04. BERT 기초 - HuggingFace BERT 사용 예제

BERT 모델 로드, 임베딩, 분류
"""

print("=" * 60)
print("BERT 기초")
print("=" * 60)

try:
    import torch
    from transformers import BertTokenizer, BertModel, BertForSequenceClassification
    import torch.nn.functional as F

    # ============================================
    # 1. 토크나이저와 모델 로드
    # ============================================
    print("\n[1] BERT 모델 로드")
    print("-" * 40)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    print(f"어휘 크기: {tokenizer.vocab_size}")
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")


    # ============================================
    # 2. 텍스트 인코딩
    # ============================================
    print("\n[2] 텍스트 인코딩")
    print("-" * 40)

    text = "Hello, how are you?"

    # 토큰화
    tokens = tokenizer.tokenize(text)
    print(f"텍스트: {text}")
    print(f"토큰: {tokens}")

    # 인코딩
    encoded = tokenizer(text, return_tensors='pt')
    print(f"input_ids: {encoded['input_ids']}")
    print(f"attention_mask: {encoded['attention_mask']}")

    # 디코딩
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"디코딩: {decoded}")


    # ============================================
    # 3. BERT 임베딩 추출
    # ============================================
    print("\n[3] BERT 임베딩 추출")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)

    # 출력 구조
    last_hidden_state = outputs.last_hidden_state  # (batch, seq, hidden)
    pooler_output = outputs.pooler_output          # (batch, hidden) - [CLS] 변환

    print(f"last_hidden_state shape: {last_hidden_state.shape}")
    print(f"pooler_output shape: {pooler_output.shape}")

    # [CLS] 토큰 임베딩
    cls_embedding = last_hidden_state[0, 0]  # 첫 번째 토큰
    print(f"[CLS] 임베딩 shape: {cls_embedding.shape}")


    # ============================================
    # 4. 문장 쌍 인코딩
    # ============================================
    print("\n[4] 문장 쌍 인코딩")
    print("-" * 40)

    text_a = "How old are you?"
    text_b = "I am 25 years old."

    encoded_pair = tokenizer(text_a, text_b, return_tensors='pt')
    print(f"문장 A: {text_a}")
    print(f"문장 B: {text_b}")
    print(f"token_type_ids: {encoded_pair['token_type_ids']}")
    # [0, 0, ..., 0, 1, 1, ..., 1] - A는 0, B는 1


    # ============================================
    # 5. 문장 분류
    # ============================================
    print("\n[5] 문장 분류")
    print("-" * 40)

    # 감성 분석 모델 로드
    classifier = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    texts = [
        "I love this movie! It's amazing.",
        "This is terrible. I hate it.",
        "The weather is nice today."
    ]

    classifier.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

        with torch.no_grad():
            outputs = classifier(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        label = "Positive" if pred == 1 else "Negative"
        conf = probs[0, pred].item()
        print(f"[{label}] ({conf:.2%}) {text[:40]}...")


    # ============================================
    # 6. 배치 처리
    # ============================================
    print("\n[6] 배치 처리")
    print("-" * 40)

    texts = ["Hello world", "How are you?", "I'm fine, thanks!"]

    # 배치 인코딩
    batch_encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    print(f"배치 input_ids shape: {batch_encoded['input_ids'].shape}")

    # 배치 추론
    model.eval()
    with torch.no_grad():
        batch_outputs = model(**batch_encoded)

    print(f"배치 출력 shape: {batch_outputs.last_hidden_state.shape}")


    # ============================================
    # 7. 문장 유사도
    # ============================================
    print("\n[7] 문장 유사도")
    print("-" * 40)

    def get_sentence_embedding(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] 토큰 또는 평균 풀링
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    sentences = [
        "I love programming",
        "Coding is my passion",
        "I enjoy eating pizza"
    ]

    embeddings = [get_sentence_embedding(s, model, tokenizer) for s in sentences]

    print("문장 유사도:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            print(f"  '{sentences[i][:20]}...' vs '{sentences[j][:20]}...': {sim.item():.4f}")


    # ============================================
    # 정리
    # ============================================
    print("\n" + "=" * 60)
    print("BERT 정리")
    print("=" * 60)

    summary = """
BERT 사용 패턴:
    # 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 인코딩
    inputs = tokenizer(text, return_tensors='pt')

    # 임베딩
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]

    # 분류
    classifier = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )
    logits = classifier(**inputs).logits
"""
    print(summary)

except ImportError as e:
    print(f"필요 패키지 미설치: {e}")
    print("pip install torch transformers")
