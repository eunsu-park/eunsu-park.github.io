"""
PyTorch Low-Level Variational Autoencoder (VAE) 구현

ELBO, Reparameterization Trick 직접 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """VAE 설정"""
    image_size: int = 28
    in_channels: int = 1
    latent_dim: int = 20
    hidden_dims: Tuple[int, ...] = (32, 64)
    beta: float = 1.0  # β-VAE


class Encoder(nn.Module):
    """VAE Encoder: x → (μ, log σ²)"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Convolutional layers
        modules = []
        in_channels = config.in_channels

        for h_dim in config.hidden_dims:
            modules.append(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 최종 feature map 크기 계산
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flatten_dim = config.hidden_dims[-1] * self.final_size * self.final_size

        # FC layers for μ and log σ²
        self.fc_mu = nn.Linear(self.flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Encode
        h = self.encoder(x)
        h = h.flatten(start_dim=1)

        # μ and log σ²
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder: z → x̂"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flatten_dim = config.hidden_dims[-1] * self.final_size * self.final_size

        # FC layer
        self.fc = nn.Linear(config.latent_dim, self.flatten_dim)

        # Transposed convolutions
        modules = []
        hidden_dims = list(config.hidden_dims)[::-1]  # 역순

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            modules.append(nn.ReLU())

        # Final layer
        modules.append(
            nn.ConvTranspose2d(
                hidden_dims[-1], config.in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        modules.append(nn.Sigmoid())  # [0, 1] 범위

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, C, H, W)
        """
        h = self.fc(z)
        h = h.view(-1, self.config.hidden_dims[-1], self.final_size, self.final_size)
        x_recon = self.decoder(h)
        return x_recon


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick

        z = μ + σ ⊙ ε, where ε ~ N(0, I)

        이렇게 하면 z의 그래디언트가 μ, σ를 통해 역전파됨
        """
        # σ = exp(log σ² / 2) = exp(logvar / 2)
        std = torch.exp(0.5 * logvar)

        # ε ~ N(0, I)
        eps = torch.randn_like(std)

        # z = μ + σ ⊙ ε
        z = mu + std * eps

        return z

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (B, C, H, W)
            return_latent: latent z 반환 여부

        Returns:
            x_recon: (B, C, H, W) 재구성 이미지
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            z: (optional) (B, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        if return_latent:
            return x_recon, mu, logvar, z

        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """인코딩만 수행"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """디코딩만 수행"""
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """잠재 공간에서 샘플링하여 이미지 생성"""
        # Prior에서 샘플링: z ~ N(0, I)
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        samples = self.decode(z)
        return samples


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE Loss: ELBO의 음수

    L = L_recon + β * L_KL

    Args:
        x: 원본 이미지 (B, C, H, W)
        x_recon: 재구성 이미지 (B, C, H, W)
        mu: 평균 (B, latent_dim)
        logvar: 로그 분산 (B, latent_dim)
        beta: KL 가중치 (β-VAE)
        reduction: "mean" or "sum"

    Returns:
        total_loss: 전체 손실
        recon_loss: 재구성 손실
        kl_loss: KL divergence
    """
    batch_size = x.size(0)

    # Reconstruction loss (Binary Cross-Entropy)
    # BCE는 각 픽셀을 독립적인 Bernoulli로 모델링
    recon_loss = F.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )

    # KL Divergence: KL(N(μ, σ²) || N(0, 1))
    # = -0.5 * Σ(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    if reduction == "mean":
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size

    return total_loss, recon_loss, kl_loss


class BetaVAE(VAE):
    """β-VAE: Disentanglement를 위한 변형"""

    def __init__(self, config: VAEConfig):
        super().__init__(config)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """β-VAE 손실 계산"""
        total_loss, recon_loss, kl_loss = vae_loss(
            x, x_recon, mu, logvar,
            beta=self.config.beta
        )

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item()
        }


class ConditionalVAE(nn.Module):
    """Conditional VAE: 조건부 생성"""

    def __init__(self, config: VAEConfig, num_classes: int = 10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # 클래스 임베딩
        self.class_embed = nn.Embedding(num_classes, config.latent_dim)

        # Encoder와 Decoder는 동일
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Encoder에 조건 추가를 위한 projection
        self.cond_proj = nn.Linear(config.latent_dim, config.in_channels * config.image_size * config.image_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
            labels: (B,) 클래스 레이블

        Returns:
            x_recon, mu, logvar
        """
        # 클래스 임베딩
        c = self.class_embed(labels)  # (B, latent_dim)

        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode with condition
        z_cond = z + c  # 조건 추가
        x_recon = self.decoder(z_cond)

        return x_recon, mu, logvar

    def sample(
        self,
        num_samples: int,
        labels: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """조건부 샘플링"""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        c = self.class_embed(labels)
        z_cond = z + c
        samples = self.decoder(z_cond)
        return samples


# 잠재 공간 시각화
def visualize_latent_space(
    model: VAE,
    data_loader,
    device: torch.device,
    num_samples: int = 1000
):
    """잠재 공간 2D 시각화 (latent_dim=2인 경우)"""
    import matplotlib.pyplot as plt

    model.eval()
    latents = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            if len(latents) * data.size(0) >= num_samples:
                break

            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu())
            labels_list.append(labels)

    latents = torch.cat(latents, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]

    # 2D 시각화 (처음 2차원만 사용)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        latents[:, 0].numpy(),
        latents[:, 1].numpy(),
        c=labels.numpy(),
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('VAE Latent Space')
    plt.savefig('vae_latent_space.png')
    print("Saved vae_latent_space.png")


def interpolate_latent(
    model: VAE,
    x1: torch.Tensor,
    x2: torch.Tensor,
    num_steps: int = 10
) -> torch.Tensor:
    """두 이미지 간 잠재 공간 보간"""
    model.eval()

    with torch.no_grad():
        # 두 이미지 인코딩
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # 선형 보간
        alphas = torch.linspace(0, 1, num_steps).to(mu1.device)
        interpolated = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            x_recon = model.decode(z)
            interpolated.append(x_recon)

        return torch.cat(interpolated, dim=0)


# 테스트
if __name__ == "__main__":
    print("=== VAE Low-Level Implementation ===\n")

    # 설정
    config = VAEConfig(
        image_size=28,
        in_channels=1,
        latent_dim=20,
        hidden_dims=(32, 64),
        beta=1.0
    )
    print(f"Config: {config}\n")

    # 모델 생성
    model = VAE(config)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # 테스트 입력
    batch_size = 8
    x = torch.rand(batch_size, 1, 28, 28)

    # Forward
    x_recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Loss 계산
    total_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"Recon Loss: {recon_loss.item():.4f}")
    print(f"KL Loss: {kl_loss.item():.4f}")

    # 샘플링 테스트
    samples = model.sample(16, x.device)
    print(f"\nSampled images shape: {samples.shape}")

    # β-VAE 테스트
    print("\n=== β-VAE Test ===")
    config_beta = VAEConfig(beta=4.0)  # β > 1 for disentanglement
    beta_vae = BetaVAE(config_beta)

    x_recon, mu, logvar = beta_vae(x)
    loss, metrics = beta_vae.compute_loss(x, x_recon, mu, logvar)
    print(f"β-VAE Loss: {metrics}")

    # Conditional VAE 테스트
    print("\n=== Conditional VAE Test ===")
    cvae = ConditionalVAE(config, num_classes=10)
    labels = torch.randint(0, 10, (batch_size,))

    x_recon, mu, logvar = cvae(x, labels)
    print(f"CVAE Reconstruction shape: {x_recon.shape}")

    # 조건부 샘플링
    cond_samples = cvae.sample(16, torch.arange(10).repeat(2)[:16], x.device)
    print(f"Conditional samples shape: {cond_samples.shape}")

    # 잠재 공간 보간
    print("\n=== Latent Interpolation ===")
    interp = interpolate_latent(model, x[:1], x[1:2], num_steps=5)
    print(f"Interpolated images shape: {interp.shape}")

    print("\nAll tests passed!")
