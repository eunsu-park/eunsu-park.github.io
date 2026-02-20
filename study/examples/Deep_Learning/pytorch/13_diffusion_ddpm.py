"""
DDPM (Denoising Diffusion Probabilistic Model) Implementation

This script implements a simple DDPM for image generation following
"Denoising Diffusion Probabilistic Models" (Ho et al., 2020).

Key concepts:
- Forward diffusion: gradually add Gaussian noise to data
- Reverse diffusion: learn to denoise and generate samples
- Linear beta schedule for noise variance
- Simple UNet architecture with time embedding

References:
- Ho et al. (2020): https://arxiv.org/abs/2006.11239
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================================
# Noise Schedule
# ============================================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule for beta (variance) from beta_start to beta_end.

    Args:
        timesteps: number of diffusion steps (T)
        beta_start: minimum noise variance
        beta_end: maximum noise variance

    Returns:
        betas: [T] tensor of noise variances
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(betas):
    """
    Precompute diffusion parameters for efficient sampling.

    Args:
        betas: [T] noise schedule

    Returns:
        Dictionary with precomputed parameters
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Posterior variance for reverse process
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
    }


# ============================================================================
# Time Embedding
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding similar to Transformer positional encoding.
    Maps timestep t to a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: [batch_size] timesteps
        Returns:
            [batch_size, dim] embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# ============================================================================
# Simple UNet Architecture
# ============================================================================

class SimpleUNet(nn.Module):
    """
    Simplified UNet for DDPM with time conditioning.

    Architecture:
    - Encoder: downsampling with conv blocks
    - Decoder: upsampling with conv blocks
    - Time embedding injected at each resolution
    """
    def __init__(self, in_channels=1, out_channels=1, time_dim=128, base_channels=64):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder (downsampling)
        self.enc1 = self._make_block(in_channels, base_channels, time_dim)
        self.enc2 = self._make_block(base_channels, base_channels * 2, time_dim)
        self.enc3 = self._make_block(base_channels * 2, base_channels * 4, time_dim)

        # Bottleneck
        self.bottleneck = self._make_block(base_channels * 4, base_channels * 4, time_dim)

        # Decoder (upsampling)
        self.dec3 = self._make_block(base_channels * 8, base_channels * 2, time_dim)
        self.dec2 = self._make_block(base_channels * 4, base_channels, time_dim)
        self.dec1 = self._make_block(base_channels * 2, base_channels, time_dim)

        # Output layer
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def _make_block(self, in_ch, out_ch, time_dim):
        """Create a residual block with time conditioning."""
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'time_proj': nn.Linear(time_dim, out_ch),
            'norm1': nn.GroupNorm(8, out_ch),
            'norm2': nn.GroupNorm(8, out_ch),
        })

    def _forward_block(self, x, t_emb, block):
        """Forward pass through a block with time embedding."""
        h = block['conv1'](x)
        h = block['norm1'](h)

        # Add time embedding
        t_proj = block['time_proj'](t_emb)[:, :, None, None]
        h = h + t_proj

        h = F.gelu(h)
        h = block['conv2'](h)
        h = block['norm2'](h)
        h = F.gelu(h)

        return h

    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] noisy images
            t: [B] timesteps
        Returns:
            [B, C, H, W] predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self._forward_block(x, t_emb, self.enc1)
        x2 = F.max_pool2d(x1, 2)

        x2 = self._forward_block(x2, t_emb, self.enc2)
        x3 = F.max_pool2d(x2, 2)

        x3 = self._forward_block(x3, t_emb, self.enc3)
        x4 = F.max_pool2d(x3, 2)

        # Bottleneck
        x4 = self._forward_block(x4, t_emb, self.bottleneck)

        # Decoder with skip connections
        x = F.interpolate(x4, scale_factor=2, mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x = self._forward_block(x, t_emb, self.dec3)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x = self._forward_block(x, t_emb, self.dec2)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x1], dim=1)
        x = self._forward_block(x, t_emb, self.dec1)

        return self.out(x)


# ============================================================================
# Diffusion Process
# ============================================================================

def forward_diffusion(x0, t, params, device):
    """
    Add noise to data according to forward diffusion process.

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Args:
        x0: [B, C, H, W] clean images
        t: [B] timesteps
        params: diffusion parameters
        device: torch device

    Returns:
        noisy_x: [B, C, H, W] noisy images
        noise: [B, C, H, W] added noise
    """
    noise = torch.randn_like(x0)

    sqrt_alpha_cumprod_t = params['sqrt_alphas_cumprod'][t][:, None, None, None]
    sqrt_one_minus_alpha_cumprod_t = params['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]

    noisy_x = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    return noisy_x, noise


@torch.no_grad()
def sample(model, params, image_size, batch_size, timesteps, device):
    """
    Generate samples using reverse diffusion process.

    Start from random noise and iteratively denoise.

    Args:
        model: trained UNet
        params: diffusion parameters
        image_size: (C, H, W)
        batch_size: number of samples
        timesteps: number of diffusion steps
        device: torch device

    Returns:
        [batch_size, C, H, W] generated images
    """
    model.eval()

    # Start from random noise
    x = torch.randn(batch_size, *image_size, device=device)

    for i in tqdm(reversed(range(timesteps)), desc='Sampling', total=timesteps):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, t)

        # Get parameters for this timestep
        alpha = params['alphas'][t][:, None, None, None]
        alpha_cumprod = params['alphas_cumprod'][t][:, None, None, None]
        beta = params['betas'][t][:, None, None, None]

        # Compute mean of reverse distribution
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # Reverse diffusion step
        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

    return x


# ============================================================================
# Training
# ============================================================================

def train_ddpm(epochs=10, batch_size=128, timesteps=1000, device='cuda'):
    """
    Train DDPM on MNIST dataset.

    Args:
        epochs: number of training epochs
        batch_size: batch size
        timesteps: number of diffusion steps (T)
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model and diffusion parameters
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    betas = linear_beta_schedule(timesteps).to(device)
    params = get_diffusion_params(betas)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size_actual = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size_actual,), device=device)

            # Forward diffusion (add noise)
            noisy_images, noise = forward_diffusion(images, t, params, device)

            # Predict noise
            predicted_noise = model(noisy_images, t)

            # MSE loss between predicted and actual noise
            loss = F.mse_loss(predicted_noise, noise)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            samples = sample(model, params, (1, 28, 28), 16, timesteps, device)
            samples = (samples + 1) / 2  # Denormalize to [0, 1]

            # Visualize
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
                ax.axis('off')
            plt.suptitle(f'Generated Samples - Epoch {epoch+1}')
            plt.tight_layout()
            plt.savefig(f'ddpm_samples_epoch_{epoch+1}.png')
            plt.close()

    print("Training completed!")
    return model, params


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Train model
    model, params = train_ddpm(epochs=10, batch_size=128, timesteps=1000, device='cuda')

    # Generate final samples
    print("\nGenerating final samples...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = sample(model, params, (1, 28, 28), 64, 1000, device)
    samples = (samples + 1) / 2

    # Visualize final samples
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('DDPM Generated Samples (Final)')
    plt.tight_layout()
    plt.savefig('ddpm_final_samples.png')
    plt.show()
