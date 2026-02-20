"""
07. 파인튜닝 예제

HuggingFace Trainer를 사용한 모델 파인튜닝
"""

print("=" * 60)
print("파인튜닝")
print("=" * 60)


# ============================================
# 1. 기본 파인튜닝 (코드 예시)
# ============================================
print("\n[1] 기본 파인튜닝")
print("-" * 40)

basic_finetuning = '''
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 데이터 로드
dataset = load_dataset("imdb")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 학습 설정
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# 학습
trainer.train()
'''
print(basic_finetuning)


# ============================================
# 2. LoRA 파인튜닝
# ============================================
print("\n[2] LoRA 파인튜닝")
print("-" * 40)

lora_code = '''
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=8,                           # 랭크
    lora_alpha=32,                 # 스케일링
    target_modules=["query", "value"],  # 적용 모듈
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# 모델에 LoRA 적용
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# trainable: 0.27% (약 300K / 110M)

# 일반 Trainer로 학습
trainer = Trainer(model=model, args=args, ...)
trainer.train()
'''
print(lora_code)


# ============================================
# 3. QLoRA (양자화 + LoRA)
# ============================================
print("\n[3] QLoRA")
print("-" * 40)

qlora_code = '''
from transformers import BitsAndBytesConfig
import torch

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습
trainer = Trainer(model=model, ...)
'''
print(qlora_code)


# ============================================
# 4. 커스텀 메트릭
# ============================================
print("\n[4] 커스텀 메트릭")
print("-" * 40)

try:
    import evaluate
    import numpy as np

    # 메트릭 로드
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        }

    print("커스텀 메트릭 함수 정의 완료")

    # 테스트
    mock_pred = (np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]))
    result = compute_metrics(mock_pred)
    print(f"테스트 결과: {result}")

except ImportError:
    print("evaluate 미설치 (pip install evaluate)")


# ============================================
# 5. NER 파인튜닝
# ============================================
print("\n[5] NER 파인튜닝")
print("-" * 40)

ner_code = '''
from transformers import AutoModelForTokenClassification

# 레이블
label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

# 모델
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# 토큰 정렬 (서브워드 처리)
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 특수 토큰
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized
'''
print(ner_code)


# ============================================
# 6. QA 파인튜닝
# ============================================
print("\n[6] QA 파인튜닝")
print("-" * 40)

qa_code = '''
from transformers import AutoModelForQuestionAnswering

# 모델
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# 전처리 (시작/끝 위치 찾기)
def prepare_train_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 답변 위치를 토큰 위치로 변환
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        # 답변 시작/끝 문자 위치 → 토큰 위치
        ...

    return tokenized
'''
print(qa_code)


# ============================================
# 7. 학습 최적화 팁
# ============================================
print("\n[7] 학습 최적화 팁")
print("-" * 40)

optimization_tips = '''
# Gradient Checkpointing (메모리 절약)
model.gradient_checkpointing_enable()

# Mixed Precision (속도 향상)
args = TrainingArguments(
    ...,
    fp16=True,  # 또는 bf16=True
)

# Gradient Accumulation (큰 배치 효과)
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 실효 배치 = 32
)

# DeepSpeed (분산 학습)
args = TrainingArguments(
    ...,
    deepspeed="ds_config.json"
)

# Learning Rate Scheduler
args = TrainingArguments(
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
'''
print(optimization_tips)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("파인튜닝 정리")
print("=" * 60)

summary = """
파인튜닝 선택 가이드:
    - 충분한 GPU: Full Fine-tuning
    - 제한된 메모리: LoRA / QLoRA
    - 매우 적은 데이터: Prompt Tuning

핵심 코드:
    # Trainer
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    # LoRA
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8, target_modules=["query", "value"])
    model = get_peft_model(model, config)
"""
print(summary)
