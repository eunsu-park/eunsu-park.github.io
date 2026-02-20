"""
05. GPT 텍스트 생성 예제

GPT-2를 사용한 텍스트 생성
"""

print("=" * 60)
print("GPT 텍스트 생성")
print("=" * 60)

try:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch.nn.functional as F

    # ============================================
    # 1. GPT-2 로드
    # ============================================
    print("\n[1] GPT-2 모델 로드")
    print("-" * 40)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token

    print(f"어휘 크기: {tokenizer.vocab_size}")
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")


    # ============================================
    # 2. 기본 생성 (Greedy)
    # ============================================
    print("\n[2] Greedy 생성")
    print("-" * 40)

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=False  # Greedy
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"프롬프트: {prompt}")
    print(f"생성: {generated}")


    # ============================================
    # 3. 샘플링 생성
    # ============================================
    print("\n[3] Temperature 샘플링")
    print("-" * 40)

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for temp in [0.5, 1.0, 1.5]:
        output = model.generate(
            input_ids,
            max_length=40,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"temp={temp}: {generated[:60]}...")


    # ============================================
    # 4. Top-k / Top-p 샘플링
    # ============================================
    print("\n[4] Top-k / Top-p 샘플링")
    print("-" * 40)

    prompt = "In the year 2050"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Top-k
    output_topk = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Top-k (k=50): {tokenizer.decode(output_topk[0], skip_special_tokens=True)[:70]}...")

    # Top-p (Nucleus)
    output_topp = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Top-p (p=0.9): {tokenizer.decode(output_topp[0], skip_special_tokens=True)[:70]}...")


    # ============================================
    # 5. 고급 생성 파라미터
    # ============================================
    print("\n[5] 고급 생성 파라미터")
    print("-" * 40)

    prompt = "Python is a programming language"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=80,
        min_length=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=2,    # n-gram 반복 방지
        repetition_penalty=1.2,     # 반복 패널티
        num_return_sequences=2,     # 여러 시퀀스 생성
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"프롬프트: {prompt}")
    for i, out in enumerate(output):
        text = tokenizer.decode(out, skip_special_tokens=True)
        print(f"\n생성 {i+1}: {text}")


    # ============================================
    # 6. 수동 생성 루프
    # ============================================
    print("\n[6] 수동 생성 (Step-by-step)")
    print("-" * 40)

    def generate_manual(prompt, max_tokens=20, temperature=1.0):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # 마지막 토큰

            # Temperature 적용
            probs = F.softmax(logits / temperature, dim=-1)

            # 샘플링
            next_token = torch.multinomial(probs, num_samples=1)

            # EOS 체크
            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    result = generate_manual("The robot said", max_tokens=15, temperature=0.8)
    print(f"수동 생성: {result}")


    # ============================================
    # 7. 조건부 생성 (프롬프트 기반)
    # ============================================
    print("\n[7] 조건부 생성")
    print("-" * 40)

    prompts = [
        "Q: What is machine learning?\nA:",
        "Translate English to French: Hello, how are you? →",
        "Summarize: Artificial intelligence is transforming various industries. →"
    ]

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"입력: {prompt[:50]}...")
        print(f"출력: {result[len(prompt):len(prompt)+60]}...")
        print()


    # ============================================
    # 정리
    # ============================================
    print("=" * 60)
    print("GPT 생성 정리")
    print("=" * 60)

    summary = """
생성 전략:
    - Greedy: do_sample=False, 결정적
    - Temperature: 낮으면 결정적, 높으면 다양
    - Top-k: 상위 k개 토큰에서 샘플링
    - Top-p (Nucleus): 누적 확률 p까지 샘플링

핵심 코드:
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
"""
    print(summary)

except ImportError as e:
    print(f"필요 패키지 미설치: {e}")
    print("pip install torch transformers")
