"""
PyTorch Low-Level BERT 구현

nn.TransformerEncoder 미사용
F.linear, F.layer_norm 등 기본 연산만 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BertEmbeddings(nn.Module):
    """
    BERT Embeddings = Token + Segment + Position

    Token: 단어 의미
    Segment: 문장 A/B 구분
    Position: 위치 정보 (학습 가능)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position: int = 512,
        type_vocab_size: int = 2,  # 문장 A, B
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding tables (nn.Embedding 사용하지만 개념적으로 lookup)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # Layer Norm + Dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) 토큰 ID
            token_type_ids: (batch, seq_len) 세그먼트 ID (0 or 1)
            position_ids: (batch, seq_len) 위치 ID

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # 기본값 설정
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 세 가지 임베딩 합산
        word_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + position_emb + token_type_emb

        # Layer Norm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-Head Self-Attention (Low-Level)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections (nn.Linear 대신 파라미터 직접 관리 가능)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len) 또는 (batch, seq_len)

        Returns:
            context: (batch, seq_len, hidden_size)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V 계산
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Multi-head reshape: (batch, seq, hidden) → (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Attention mask 적용
        if attention_mask is not None:
            # (batch, seq) → (batch, 1, 1, seq)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 0 → -inf, 1 → 0
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask

        # Softmax + Dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Context: (batch, heads, seq, head_dim)
        context = torch.matmul(attention_weights, V)

        # Reshape back: (batch, seq, hidden)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)

        return context, attention_weights


class BertSelfOutput(nn.Module):
    """Attention Output (projection + residual + layer norm)"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: attention 출력
            input_tensor: residual connection용 원본 입력
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual + Layer Norm
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    """Feed-Forward 첫 번째 층 (확장)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # BERT는 GELU 사용

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Feed-Forward 두 번째 층 (축소) + Residual"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Single BERT Encoder Layer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Self-Attention
        self.attention = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attention_output = BertSelfOutput(hidden_size, dropout)

        # Feed-Forward
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention
        attention_output, attention_weights = self.attention(
            hidden_states, attention_mask
        )
        attention_output = self.attention_output(attention_output, hidden_states)

        # Feed-Forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_weights


class BertEncoder(nn.Module):
    """BERT Encoder (stacked layers)"""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(attention_weights)

        return hidden_states, all_attentions


class BertPooler(nn.Module):
    """[CLS] 토큰 풀링"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            pooled: (batch, hidden_size) - [CLS] 토큰의 표현
        """
        # [CLS] 토큰 (첫 번째 토큰)
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = torch.tanh(pooled)
        return pooled


class BertModel(nn.Module):
    """BERT Base Model"""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position, type_vocab_size, dropout
        )
        self.encoder = BertEncoder(
            num_layers, hidden_size, num_heads, intermediate_size, dropout
        )
        self.pooler = BertPooler(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            token_type_ids: (batch, seq_len) - 0 for sent A, 1 for sent B

        Returns:
            last_hidden_state: (batch, seq_len, hidden_size)
            pooler_output: (batch, hidden_size)
            attentions: optional list of attention weights
        """
        # Embeddings
        embeddings = self.embeddings(
            input_ids, token_type_ids=token_type_ids
        )

        # Encoder
        encoder_output, attentions = self.encoder(
            embeddings, attention_mask, output_attentions
        )

        # Pooler
        pooled_output = self.pooler(encoder_output)

        return {
            'last_hidden_state': encoder_output,
            'pooler_output': pooled_output,
            'attentions': attentions
        }


class BertForMaskedLM(nn.Module):
    """BERT for Masked Language Modeling"""

    def __init__(self, config: dict):
        super().__init__()
        self.bert = BertModel(**config)

        # MLM Head
        self.cls = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.LayerNorm(config['hidden_size'], eps=1e-12),
            nn.Linear(config['hidden_size'], config['vocab_size'])
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_states = outputs['last_hidden_state']

        # MLM predictions
        prediction_scores = self.cls(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': hidden_states
        }


class BertForSequenceClassification(nn.Module):
    """BERT for Sequence Classification"""

    def __init__(self, config: dict, num_labels: int):
        super().__init__()
        self.bert = BertModel(**config)
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs['pooler_output']

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


# 테스트
if __name__ == "__main__":
    print("=== BERT Low-Level Implementation Test ===\n")

    # 설정
    config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'intermediate_size': 3072,
        'max_position': 512,
        'dropout': 0.1
    }

    # 모델 생성
    model = BertModel(**config)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~110M for BERT-Base\n")

    # 테스트 입력
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Forward
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_attentions=True
    )

    print("Output shapes:")
    print(f"  last_hidden_state: {outputs['last_hidden_state'].shape}")
    print(f"  pooler_output: {outputs['pooler_output'].shape}")
    print(f"  attentions: {len(outputs['attentions'])} layers")
    print(f"  attention shape: {outputs['attentions'][0].shape}")

    # MLM 테스트
    print("\n=== MLM Test ===")
    mlm_model = BertForMaskedLM(config)

    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    labels[labels != 103] = -100  # [MASK] token만 예측

    mlm_outputs = mlm_model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"MLM Loss: {mlm_outputs['loss'].item():.4f}")
    print(f"Logits shape: {mlm_outputs['logits'].shape}")

    # Classification 테스트
    print("\n=== Classification Test ===")
    clf_model = BertForSequenceClassification(config, num_labels=2)

    labels = torch.randint(0, 2, (batch_size,))
    clf_outputs = clf_model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Classification Loss: {clf_outputs['loss'].item():.4f}")
    print(f"Logits shape: {clf_outputs['logits'].shape}")

    print("\nAll tests passed!")
