"""
17. Simple Diffusion Model (DDPM) Implementation

A minimal implementation of Denoising Diffusion Probabilistic Models
for MNIST digit generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math

print("=" * 60)
print("Simple Diffusion Model (DDPM) Implementation")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# 1. Noise Schedule
# ============================================
print("\n[1] Noise Schedule")
print("-" * 40)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (better performance)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_index_from_list(vals, t, x_shape):
    """Extract values from schedule at timestep t for each sample in batch"""
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionSchedule:
    """Manages all diffusion schedule parameters"""
    def __init__(self, timesteps=1000, beta_schedule='linear', device='cpu'):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # Calculations for diffusion q(x_t | x_0) and posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)

        # Posterior variance
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: sample x_t from q(x_t | x_0)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# Test schedule
schedule = DiffusionSchedule(timesteps=1000, device=device)
print(f"Timesteps: {schedule.timesteps}")
print(f"Beta range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
print(f"Alpha_bar range: [{schedule.alphas_cumprod[-1]:.6f}, {schedule.alphas_cumprod[0]:.6f}]")


# ============================================
# 2. U-Net Architecture
# ============================================
print("\n[2] U-Net Architecture")
print("-" * 40)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic convolutional block with time embedding"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[..., None, None]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    """Simple U-Net for noise prediction"""
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, base_channels=64):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Initial conv
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([
            Block(base_channels, base_channels * 2, time_dim),
            Block(base_channels * 2, base_channels * 4, time_dim),
        ])

        # Bottleneck
        self.bot1 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        self.bot2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)

        # Upsampling
        self.ups = nn.ModuleList([
            Block(base_channels * 4, base_channels * 2, time_dim, up=True),
            Block(base_channels * 2, base_channels, time_dim, up=True),
        ])

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        # Downsample
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        # Bottleneck
        x = F.relu(self.bot1(x))
        x = F.relu(self.bot2(x))

        # Upsample with skip connections
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)


# Test U-Net
unet = SimpleUNet(in_channels=1, out_channels=1)
x = torch.randn(4, 1, 28, 28)
t = torch.randint(0, 1000, (4,))
out = unet(x, t)
print(f"U-Net input: {x.shape}")
print(f"U-Net output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")


# ============================================
# 3. Training
# ============================================
print("\n[3] Training Loop")
print("-" * 40)


def train_diffusion(model, schedule, dataloader, epochs=5, lr=1e-3):
    """Train diffusion model"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            # Random timesteps
            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()

            # Add noise
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)

            # Predict noise
            noise_pred = model(x_t, t)

            # Loss
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    return losses


# ============================================
# 4. Sampling (Reverse Process)
# ============================================
print("\n[4] Sampling (Reverse Process)")
print("-" * 40)


@torch.no_grad()
def sample_ddpm(model, schedule, shape, device, show_progress=True):
    """DDPM sampling: generate images from pure noise"""
    model.eval()

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        betas_t = get_index_from_list(schedule.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            schedule.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(
            schedule.sqrt_recip_alphas, t, x.shape
        )

        # Predict noise
        noise_pred = model(x, t)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        # Add noise (except for t=0)
        if i > 0:
            posterior_variance_t = get_index_from_list(
                schedule.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x = model_mean

        if show_progress and i % 100 == 0:
            print(f"  Sampling step {schedule.timesteps - i}/{schedule.timesteps}")

    return x


@torch.no_grad()
def sample_ddim(model, schedule, shape, device, num_steps=50, eta=0.0):
    """DDIM sampling: faster with fewer steps"""
    model.eval()

    # Create step sequence
    step_size = schedule.timesteps // num_steps
    timesteps = list(range(0, schedule.timesteps, step_size))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        alpha_cumprod_t = schedule.alphas_cumprod[t]

        if i < len(timesteps) - 1:
            alpha_cumprod_prev = schedule.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=device)

        # Predict noise
        noise_pred = model(x, t_tensor)

        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Compute variance
        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )

        # Direction
        pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * noise_pred

        # Next x
        noise = torch.randn_like(x) if eta > 0 and i < len(timesteps) - 1 else 0
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir + sigma * noise

    return x


# ============================================
# 5. Visualize Diffusion Process
# ============================================
print("\n[5] Visualize Forward Process")
print("-" * 40)


def visualize_forward_process(schedule, image, timesteps_to_show):
    """Show image at different noise levels"""
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))

    for idx, t in enumerate(timesteps_to_show):
        t_tensor = torch.tensor([t])
        noisy = schedule.q_sample(image.unsqueeze(0), t_tensor)

        axes[idx].imshow(noisy[0, 0].cpu(), cmap='gray')
        axes[idx].set_title(f't = {t}')
        axes[idx].axis('off')

    plt.suptitle('Forward Diffusion Process')
    plt.tight_layout()
    plt.savefig('diffusion_forward.png', dpi=150)
    plt.close()
    print("Forward process visualization saved to diffusion_forward.png")


# ============================================
# 6. Training Example
# ============================================
print("\n[6] Training on MNIST")
print("-" * 40)

# Hyperparameters
timesteps = 1000
batch_size = 64
epochs = 5  # Increase for better results
lr = 1e-3

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # [0, 1] -> [-1, 1]
])

print("Loading MNIST dataset...")
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

# Schedule
schedule = DiffusionSchedule(timesteps=timesteps, beta_schedule='linear', device=device)

# Model
model = SimpleUNet(in_channels=1, out_channels=1).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Visualize forward process
sample_img, _ = train_data[0]
visualize_forward_process(schedule, sample_img, [0, 100, 300, 500, 700, 900, 999])

# Train
print("\nTraining diffusion model...")
losses = train_diffusion(model, schedule, train_loader, epochs=epochs, lr=lr)

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Model Training Loss')
plt.savefig('diffusion_loss.png', dpi=150)
plt.close()
print("Loss curve saved to diffusion_loss.png")

# Sample
print("\nGenerating samples with DDPM...")
samples = sample_ddpm(model, schedule, (16, 1, 28, 28), device)
samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
samples = samples.clamp(0, 1)

grid = vutils.make_grid(samples.cpu(), nrow=4, normalize=False, padding=2)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
plt.axis('off')
plt.title('DDPM Generated Samples')
plt.savefig('diffusion_samples.png', dpi=150)
plt.close()
print("Generated samples saved to diffusion_samples.png")

# DDIM sampling (faster)
print("\nGenerating samples with DDIM (50 steps)...")
samples_ddim = sample_ddim(model, schedule, (16, 1, 28, 28), device, num_steps=50, eta=0.0)
samples_ddim = (samples_ddim + 1) / 2
samples_ddim = samples_ddim.clamp(0, 1)

grid_ddim = vutils.make_grid(samples_ddim.cpu(), nrow=4, normalize=False, padding=2)
plt.figure(figsize=(8, 8))
plt.imshow(grid_ddim.permute(1, 2, 0).squeeze(), cmap='gray')
plt.axis('off')
plt.title('DDIM Generated Samples (50 steps)')
plt.savefig('diffusion_samples_ddim.png', dpi=150)
plt.close()
print("DDIM samples saved to diffusion_samples_ddim.png")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Diffusion Model Summary")
print("=" * 60)

summary = """
Key Concepts:
1. Forward Process: Gradually add noise to data
   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

2. Reverse Process: Learn to denoise step by step
   Model predicts noise epsilon at each step

3. Training: Simple MSE loss on noise prediction
   L = ||epsilon - epsilon_theta(x_t, t)||^2

4. Sampling:
   - DDPM: 1000 steps, stochastic
   - DDIM: 50-100 steps, deterministic

Noise Schedules:
- Linear: Simple, widely used
- Cosine: Better quality for small images

Key Parameters:
- timesteps: Number of diffusion steps (1000)
- beta_start, beta_end: Noise schedule bounds
- U-Net: Time-conditioned denoising network

Output Files:
- diffusion_forward.png: Forward process visualization
- diffusion_loss.png: Training loss curve
- diffusion_samples.png: DDPM generated samples
- diffusion_samples_ddim.png: DDIM generated samples
"""
print(summary)
print("=" * 60)
