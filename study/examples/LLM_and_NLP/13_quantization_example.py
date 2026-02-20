"""
13. 모델 양자화 (Model Quantization) 예제

INT8/INT4 양자화, bitsandbytes, GPTQ, AWQ 실습
"""

import numpy as np

print("=" * 60)
print("모델 양자화 (Model Quantization)")
print("=" * 60)


# ============================================
# 1. 기본 양자화 이해
# ============================================
print("\n[1] 기본 양자화 개념")
print("-" * 40)

def quantize_symmetric(tensor, bits=8):
    """대칭 양자화 (Symmetric Quantization)"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # 스케일 계산
    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax if abs_max != 0 else 1.0

    # 양자화
    quantized = np.round(tensor / scale).astype(np.int8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale

def dequantize(quantized, scale):
    """역양자화"""
    return quantized.astype(np.float32) * scale


# 테스트
original = np.array([0.5, -1.2, 0.3, 2.1, -0.8, 0.0], dtype=np.float32)
print(f"원본 텐서: {original}")

quantized, scale = quantize_symmetric(original, bits=8)
print(f"양자화됨 (INT8): {quantized}")
print(f"스케일: {scale:.6f}")

recovered = dequantize(quantized, scale)
print(f"복원됨: {recovered}")

error = np.abs(original - recovered).mean()
print(f"평균 양자화 오차: {error:.6f}")


# ============================================
# 2. 비대칭 양자화
# ============================================
print("\n[2] 비대칭 양자화")
print("-" * 40)

def quantize_asymmetric(tensor, bits=8):
    """비대칭 양자화 (Asymmetric Quantization)"""
    qmin = 0
    qmax = 2 ** bits - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = round(-min_val / scale) if scale != 0 else 0

    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale, zero_point

def dequantize_asymmetric(quantized, scale, zero_point):
    """비대칭 역양자화"""
    return (quantized.astype(np.float32) - zero_point) * scale


# 테스트
asym_quantized, asym_scale, zero_point = quantize_asymmetric(original, bits=8)
print(f"비대칭 양자화 (UINT8): {asym_quantized}")
print(f"스케일: {asym_scale:.6f}, Zero Point: {zero_point}")

asym_recovered = dequantize_asymmetric(asym_quantized, asym_scale, zero_point)
print(f"복원됨: {asym_recovered}")


# ============================================
# 3. 그룹별 양자화
# ============================================
print("\n[3] 그룹별 양자화 (Group Quantization)")
print("-" * 40)

def group_quantize(tensor, group_size=4, bits=4):
    """그룹별 양자화 - 정확도 향상"""
    flat = tensor.flatten()
    pad_size = (group_size - len(flat) % group_size) % group_size
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size))

    groups = flat.reshape(-1, group_size)
    quantized_groups = []
    scales = []

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    for group in groups:
        abs_max = np.abs(group).max()
        scale = abs_max / qmax if abs_max != 0 else 1.0
        q = np.round(group / scale).astype(np.int8)
        q = np.clip(q, qmin, qmax)
        quantized_groups.append(q)
        scales.append(scale)

    return np.array(quantized_groups), np.array(scales)

def group_dequantize(quantized_groups, scales):
    """그룹별 역양자화"""
    recovered = []
    for q, s in zip(quantized_groups, scales):
        recovered.append(q.astype(np.float32) * s)
    return np.concatenate(recovered)


# 테스트
larger_tensor = np.random.randn(16).astype(np.float32)
print(f"원본 (16개): {larger_tensor[:8]}...")

g_quantized, g_scales = group_quantize(larger_tensor, group_size=4, bits=4)
print(f"그룹 수: {len(g_scales)}, 그룹 크기: 4")
print(f"스케일들: {g_scales}")

g_recovered = group_dequantize(g_quantized, g_scales)
g_error = np.abs(larger_tensor - g_recovered).mean()
print(f"그룹 양자화 평균 오차: {g_error:.6f}")


# ============================================
# 4. 비트 정밀도 비교
# ============================================
print("\n[4] 비트 정밀도 비교")
print("-" * 40)

def compare_bit_precision(tensor):
    """다양한 비트 정밀도 비교"""
    results = {}

    for bits in [8, 4, 2]:
        q, s = quantize_symmetric(tensor, bits=bits)
        r = dequantize(q, s)
        error = np.abs(tensor - r).mean()
        results[f"INT{bits}"] = {
            "error": error,
            "range": (-(2**(bits-1)), 2**(bits-1)-1)
        }

    return results

comparison = compare_bit_precision(original)
print("비트별 양자화 비교:")
for name, result in comparison.items():
    print(f"  {name}: 오차={result['error']:.6f}, 범위={result['range']}")


# ============================================
# 5. bitsandbytes 예제 (코드만)
# ============================================
print("\n[5] bitsandbytes 사용법 (코드 예시)")
print("-" * 40)

bnb_code = '''
# bitsandbytes 8비트 양자화
from transformers import AutoModelForCausalLM, AutoTokenizer

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# bitsandbytes 4비트 양자화 (NF4)
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # 이중 양자화
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4bit 모델 메모리: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
'''
print(bnb_code)


# ============================================
# 6. GPTQ 예제 (코드만)
# ============================================
print("\n[6] GPTQ 양자화 (코드 예시)")
print("-" * 40)

gptq_code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# GPTQ 설정
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    dataset=calibration_data,
    tokenizer=tokenizer
)

# 양자화
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

model.save_pretrained("./llama-2-7b-gptq-4bit")

# 사전 양자화 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
'''
print(gptq_code)


# ============================================
# 7. AWQ 예제 (코드만)
# ============================================
print("\n[7] AWQ 양자화 (코드 예시)")
print("-" * 40)

awq_code = '''
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 모델 로드
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# AWQ 양자화 설정
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 양자화
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./llama-2-7b-awq")

# AWQ 모델 추론
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq",
    fuse_layers=True  # 레이어 퓨전으로 속도 향상
)
'''
print(awq_code)


# ============================================
# 8. QLoRA 예제 (코드만)
# ============================================
print("\n[8] QLoRA 파인튜닝 (코드 예시)")
print("-" * 40)

qlora_code = '''
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# k-bit 학습 준비
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력: trainable params: ~0.1%
'''
print(qlora_code)


# ============================================
# 9. 양자화 메모리 절약 시뮬레이션
# ============================================
print("\n[9] 양자화 메모리 절약 시뮬레이션")
print("-" * 40)

def estimate_model_size(params_billions, bits):
    """모델 크기 추정 (GB)"""
    bytes_per_param = bits / 8
    size_gb = params_billions * 1e9 * bytes_per_param / (1024**3)
    return size_gb

model_sizes = {
    "7B": 7,
    "13B": 13,
    "70B": 70,
}

precisions = {
    "FP32": 32,
    "FP16": 16,
    "INT8": 8,
    "INT4": 4,
}

print("모델 크기 추정 (GB):")
print("-" * 60)
header = "Model\t" + "\t".join(precisions.keys())
print(header)
print("-" * 60)

for model_name, params in model_sizes.items():
    sizes = [f"{estimate_model_size(params, bits):.1f}" for bits in precisions.values()]
    print(f"{model_name}\t" + "\t".join(sizes))


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("양자화 정리")
print("=" * 60)

summary = """
양자화 핵심 개념:

1. 대칭 양자화:
   - scale = max(|x|) / (2^(bits-1) - 1)
   - x_q = round(x / scale)
   - x' = x_q * scale

2. 비대칭 양자화:
   - scale = (max - min) / (2^bits - 1)
   - zero_point = round(-min / scale)
   - x_q = round(x / scale + zero_point)

3. 양자화 방법 비교:
   - bitsandbytes: 빠른 적용, 동적 양자화
   - GPTQ: 높은 품질, 캘리브레이션 필요
   - AWQ: 빠른 양자화, 활성화 기반
   - QLoRA: 양자화 + LoRA 파인튜닝

4. 선택 가이드:
   - 프로토타이핑: bitsandbytes (load_in_8bit)
   - 메모리 제한: bitsandbytes (load_in_4bit)
   - 프로덕션: GPTQ 또는 AWQ
   - 파인튜닝: QLoRA
"""
print(summary)
