"""
14. RLHF와 LLM 정렬 (Alignment) 예제

Reward Model, PPO, DPO, Constitutional AI 실습
"""

import numpy as np
import random

print("=" * 60)
print("RLHF와 LLM 정렬 (Alignment)")
print("=" * 60)


# ============================================
# 1. 선호도 데이터 이해
# ============================================
print("\n[1] 선호도 데이터 형식")
print("-" * 40)

# 선호도 데이터 예시
preference_data = [
    {
        "prompt": "인공지능이란 무엇인가요?",
        "chosen": "인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, "
                  "문제 해결 등의 작업을 수행하는 기술입니다. 머신러닝, 딥러닝, "
                  "자연어 처리 등 다양한 분야를 포함합니다.",
        "rejected": "AI는 컴퓨터가 똑똑해지는 것입니다."
    },
    {
        "prompt": "파이썬의 장점은?",
        "chosen": "파이썬의 주요 장점은 1) 읽기 쉬운 문법, 2) 풍부한 라이브러리, "
                  "3) 다양한 분야 적용 가능, 4) 활발한 커뮤니티입니다.",
        "rejected": "파이썬은 좋은 언어입니다."
    },
    {
        "prompt": "운동의 효과는?",
        "chosen": "규칙적인 운동은 심혈관 건강 개선, 체중 관리, 근력 강화, "
                  "정신 건강 향상, 수면 질 개선 등 다양한 효과가 있습니다.",
        "rejected": "운동하면 건강해집니다."
    }
]

print("선호도 데이터 예시:")
for i, data in enumerate(preference_data):
    print(f"\n{i+1}. 프롬프트: {data['prompt']}")
    print(f"   선호 응답: {data['chosen'][:50]}...")
    print(f"   비선호 응답: {data['rejected']}")


# ============================================
# 2. 간단한 Reward Model 시뮬레이션
# ============================================
print("\n[2] 간단한 Reward Model")
print("-" * 40)

class SimpleRewardModel:
    """간단한 규칙 기반 Reward Model (시뮬레이션용)"""

    def __init__(self):
        self.positive_factors = {
            "length": 0.3,        # 적절한 길이
            "detail": 0.3,        # 상세함
            "structure": 0.2,     # 구조화
            "politeness": 0.2     # 정중함
        }

    def compute_reward(self, prompt, response):
        """응답에 대한 보상 점수 계산"""
        score = 0.0

        # 1. 길이 점수 (50-300자 최적)
        length = len(response)
        if 50 <= length <= 300:
            score += self.positive_factors["length"]
        elif length > 300:
            score += self.positive_factors["length"] * 0.5

        # 2. 상세함 (숫자, 예시 포함)
        if any(c.isdigit() for c in response):
            score += self.positive_factors["detail"] * 0.5
        if "예를 들어" in response or "예시" in response:
            score += self.positive_factors["detail"] * 0.5

        # 3. 구조화 (쉼표, 마침표 사용)
        if response.count(',') >= 2:
            score += self.positive_factors["structure"]

        # 4. 정중함
        polite_words = ["입니다", "습니다", "됩니다"]
        if any(word in response for word in polite_words):
            score += self.positive_factors["politeness"]

        return score

# 테스트
reward_model = SimpleRewardModel()

print("Reward Model 테스트:")
for data in preference_data:
    chosen_reward = reward_model.compute_reward(data["prompt"], data["chosen"])
    rejected_reward = reward_model.compute_reward(data["prompt"], data["rejected"])
    print(f"\n프롬프트: {data['prompt']}")
    print(f"  선호 응답 점수: {chosen_reward:.2f}")
    print(f"  비선호 응답 점수: {rejected_reward:.2f}")
    print(f"  정렬 여부: {'OK' if chosen_reward > rejected_reward else 'FAIL'}")


# ============================================
# 3. Bradley-Terry 모델 (DPO 기반)
# ============================================
print("\n[3] Bradley-Terry 모델 (선호도 확률)")
print("-" * 40)

def bradley_terry_probability(reward_chosen, reward_rejected, beta=1.0):
    """
    Bradley-Terry 모델로 선호 확률 계산

    P(chosen > rejected) = sigmoid(beta * (r_chosen - r_rejected))
    """
    diff = reward_chosen - reward_rejected
    prob = 1 / (1 + np.exp(-beta * diff))
    return prob

def dpo_loss(reward_chosen, reward_rejected, beta=0.1):
    """
    DPO 손실 함수 (간단한 버전)

    L = -log(sigmoid(beta * (r_chosen - r_rejected)))
    """
    prob = bradley_terry_probability(reward_chosen, reward_rejected, beta)
    loss = -np.log(prob + 1e-10)
    return loss

# 테스트
print("Bradley-Terry 선호 확률:")
for r_c, r_r in [(0.8, 0.3), (0.5, 0.5), (0.3, 0.7)]:
    prob = bradley_terry_probability(r_c, r_r, beta=2.0)
    loss = dpo_loss(r_c, r_r, beta=2.0)
    print(f"  r_chosen={r_c}, r_rejected={r_r} -> P(chosen)={prob:.4f}, Loss={loss:.4f}")


# ============================================
# 4. PPO 개념 시뮬레이션
# ============================================
print("\n[4] PPO 개념 시뮬레이션")
print("-" * 40)

class SimplePPOSimulator:
    """PPO 개념 시뮬레이션"""

    def __init__(self, clip_epsilon=0.2, kl_coef=0.1):
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.policy_history = []

    def compute_ratio(self, new_prob, old_prob):
        """확률 비율 계산"""
        return new_prob / (old_prob + 1e-10)

    def clip_ratio(self, ratio):
        """PPO 클리핑"""
        return np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

    def compute_ppo_objective(self, ratio, advantage):
        """PPO 목적 함수"""
        clipped_ratio = self.clip_ratio(ratio)
        obj1 = ratio * advantage
        obj2 = clipped_ratio * advantage
        return min(obj1, obj2)  # 보수적 업데이트

    def compute_kl_penalty(self, new_prob, old_prob):
        """KL 페널티"""
        kl = new_prob * np.log(new_prob / (old_prob + 1e-10) + 1e-10)
        return self.kl_coef * kl

# 테스트
ppo = SimplePPOSimulator()
print("PPO 클리핑 예시:")

test_cases = [
    (0.8, 0.5, 1.0),   # 확률 증가, 양의 어드밴티지
    (0.3, 0.5, 1.0),   # 확률 감소, 양의 어드밴티지
    (0.8, 0.5, -1.0),  # 확률 증가, 음의 어드밴티지
]

for new_p, old_p, adv in test_cases:
    ratio = ppo.compute_ratio(new_p, old_p)
    clipped = ppo.clip_ratio(ratio)
    obj = ppo.compute_ppo_objective(ratio, adv)
    print(f"  new_p={new_p}, old_p={old_p}, adv={adv}")
    print(f"    ratio={ratio:.2f}, clipped={clipped:.2f}, objective={obj:.2f}")


# ============================================
# 5. SFT 데이터 형식
# ============================================
print("\n[5] SFT (Supervised Fine-Tuning) 데이터")
print("-" * 40)

# Alpaca 형식
alpaca_data = [
    {
        "instruction": "다음 텍스트를 요약하세요.",
        "input": "인공지능은 컴퓨터 과학의 한 분야로, 인간의 학습능력, 추론능력, "
                 "지각능력, 자연언어 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.",
        "output": "인공지능은 인간의 지적 능력을 컴퓨터로 구현한 기술입니다."
    },
    {
        "instruction": "다음 문장을 영어로 번역하세요.",
        "input": "안녕하세요, 오늘 날씨가 좋네요.",
        "output": "Hello, the weather is nice today."
    }
]

print("Alpaca 형식 예시:")
for item in alpaca_data:
    print(f"\n  Instruction: {item['instruction']}")
    print(f"  Input: {item['input'][:40]}...")
    print(f"  Output: {item['output']}")

# ChatML 형식
chatml_example = """
<|system|>
You are a helpful assistant.
<|user|>
What is the capital of Korea?
<|assistant|>
The capital of South Korea is Seoul.
"""

print(f"\nChatML 형식 예시:{chatml_example}")


# ============================================
# 6. Constitutional AI 시뮬레이션
# ============================================
print("\n[6] Constitutional AI 시뮬레이션")
print("-" * 40)

class ConstitutionalAI:
    """Constitutional AI 시뮬레이션"""

    def __init__(self):
        self.constitution = [
            "응답은 도움이 되어야 합니다.",
            "응답은 해로운 내용을 포함하지 않아야 합니다.",
            "응답은 정직하고 사실에 기반해야 합니다.",
            "차별적이거나 편견 있는 내용을 포함하지 않아야 합니다."
        ]

    def check_principles(self, response):
        """원칙 위반 확인 (간단한 규칙 기반)"""
        violations = []

        # 해로운 키워드 체크
        harmful_words = ["폭력", "위험한", "불법"]
        if any(word in response for word in harmful_words):
            violations.append("해로운 내용 포함 가능")

        # 너무 짧은 응답
        if len(response) < 20:
            violations.append("충분히 도움이 되지 않음")

        return violations

    def critique(self, prompt, response):
        """응답 비평"""
        violations = self.check_principles(response)

        critique = f"프롬프트: {prompt}\n응답: {response}\n\n원칙 검토:\n"
        for i, principle in enumerate(self.constitution, 1):
            critique += f"  {i}. {principle}\n"

        if violations:
            critique += f"\n위반 사항:\n"
            for v in violations:
                critique += f"  - {v}\n"
        else:
            critique += "\n모든 원칙 준수"

        return critique, violations

    def revise(self, response, violations):
        """응답 수정 (시뮬레이션)"""
        revised = response
        if "충분히 도움이 되지 않음" in violations:
            revised = response + " 추가적인 설명이 필요하시면 말씀해 주세요."
        return revised


# 테스트
cai = ConstitutionalAI()

test_responses = [
    ("파이썬 배우는 방법?", "책을 읽으세요."),
    ("운동의 효과?", "운동은 건강에 매우 좋습니다. 심혈관 기능 개선, 체중 관리, 정신 건강 향상 등 다양한 이점이 있습니다."),
]

print("Constitutional AI 검토:")
for prompt, response in test_responses:
    critique, violations = cai.critique(prompt, response)
    print(f"\n{'-'*30}")
    print(critique)
    if violations:
        revised = cai.revise(response, violations)
        print(f"수정된 응답: {revised}")


# ============================================
# 7. TRL 라이브러리 사용법 (코드만)
# ============================================
print("\n[7] TRL 라이브러리 코드 예시")
print("-" * 40)

trl_code = '''
# SFT (Supervised Fine-Tuning)
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir="./sft_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    ),
)
trainer.train()

# DPO (Direct Preference Optimization)
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,  # 온도 파라미터
    loss_type="sigmoid",
    max_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,  # prompt, chosen, rejected
    tokenizer=tokenizer,
)
trainer.train()

# PPO (Proximal Policy Optimization)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    ppo_epochs=4,
    target_kl=0.1,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 학습 루프
for batch in dataloader:
    query_tensors = tokenize(batch["prompt"])
    response_tensors = ppo_trainer.generate(query_tensors)
    rewards = reward_model(query_tensors, response_tensors)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
'''
print(trl_code)


# ============================================
# 8. Reward Model 학습 (코드만)
# ============================================
print("\n[8] Reward Model 학습 코드")
print("-" * 40)

reward_code = '''
from transformers import AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer

# Reward Model (분류 헤드 추가)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=1  # 스칼라 출력
)

# 학습
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# 보상 점수 계산
def get_reward(prompt, response):
    text = f"### Prompt: {prompt}\\n### Response: {response}"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze().item()
    return reward
'''
print(reward_code)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("RLHF 정리")
print("=" * 60)

summary = """
RLHF 파이프라인:

1. SFT (Supervised Fine-Tuning)
   - 고품질 데이터로 기본 능력 학습
   - 형식: instruction, input, output

2. Reward Model 학습
   - 선호도 데이터로 보상 함수 학습
   - 형식: prompt, chosen, rejected

3. PPO (강화학습)
   - Reward Model로 정책 최적화
   - KL 페널티로 기준 모델과의 거리 제한

4. DPO (Direct Preference Optimization)
   - Reward Model 없이 직접 선호도 학습
   - L = -log(sigmoid(β * (log π(y_w) - log π(y_l))))

5. Constitutional AI
   - 원칙 기반 자기 비평 및 수정
   - 안전성 향상

정렬 방법 선택:
- 간단한 정렬: DPO (추천)
- 복잡한 정렬: RLHF (PPO)
- 안전성 중요: Constitutional AI
"""
print(summary)
