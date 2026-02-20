"""
16. Variational Autoencoder (VAE) Implementation

VAE implementation for MNIST digit generation with latent space visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Variational Autoencoder (VAE) Implementation")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# 1. VAE Components
# ============================================
print("\n[1] VAE Architecture")
print("-" * 40)


class VAEEncoder(nn.Module):
    """VAE Encoder: Image -> mu, log_var"""
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """VAE Decoder: z -> Image"""
    def __init__(self, latent_dim=20, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        return self.deconv_layers(h)


class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def generate(self, num_samples, device):
        """Generate new samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def reconstruct(self, x):
        """Reconstruct input images"""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        return x_recon


# Test VAE
vae = VAE(in_channels=1, latent_dim=20)
x = torch.randn(4, 1, 28, 28)
x_recon, mu, log_var = vae(x)
print(f"Input: {x.shape}")
print(f"Reconstruction: {x_recon.shape}")
print(f"Mu: {mu.shape}, Log_var: {log_var.shape}")


# ============================================
# 2. Loss Functions
# ============================================
print("\n[2] VAE Loss Functions")
print("-" * 40)


def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    """VAE ELBO Loss: Reconstruction + KL Divergence

    Args:
        x: Original images
        x_recon: Reconstructed images
        mu: Latent mean
        log_var: Latent log variance
        beta: KL weight (beta > 1 for beta-VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def vae_loss_mse(x, x_recon, mu, log_var, beta=1.0):
    """VAE Loss with MSE reconstruction"""
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


print("Loss functions defined")


# ============================================
# 3. Training Loop
# ============================================
print("\n[3] Training Loop")
print("-" * 40)


def train_vae(model, dataloader, epochs=10, lr=1e-3, beta=1.0):
    """Train VAE on MNIST"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'total': [], 'recon': [], 'kl': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward
            x_recon, mu, log_var = model(data)

            # Loss
            loss, recon, kl = vae_loss(data, x_recon, mu, log_var, beta)
            loss = loss / data.size(0)  # Normalize by batch

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item() / data.size(0)
            total_kl += kl.item() / data.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        history['total'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['kl'].append(avg_kl)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    return history


# ============================================
# 4. Latent Space Visualization
# ============================================
print("\n[4] Latent Space Visualization")
print("-" * 40)


def visualize_latent_space(model, dataloader, device):
    """Visualize 2D latent space with class colors"""
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu())
            labels.append(label)

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Only plot first 2 dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('VAE Latent Space (First 2 Dimensions)')
    plt.savefig('vae_latent_space.png', dpi=150)
    plt.close()
    print("Latent space visualization saved to vae_latent_space.png")


def generate_manifold(model, n=20, latent_dim=2, device='cpu', digit_size=28):
    """Generate manifold of digits from 2D latent space"""
    model.eval()

    # Create grid in latent space
    grid_x = torch.linspace(-3, 3, n)
    grid_y = torch.linspace(-3, 3, n)

    figure = np.zeros((digit_size * n, digit_size * n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = xi
                z[0, 1] = yi

                x_decoded = model.decoder(z)
                digit = x_decoded[0, 0].cpu().numpy()

                figure[i * digit_size:(i + 1) * digit_size,
                       j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title('VAE Manifold')
    plt.savefig('vae_manifold.png', dpi=150)
    plt.close()
    print("Manifold saved to vae_manifold.png")


def explore_latent_dimension(model, dim_idx, range_vals, fixed_z, device):
    """Explore effect of changing one latent dimension"""
    model.eval()
    images = []

    with torch.no_grad():
        for val in range_vals:
            z = fixed_z.clone()
            z[0, dim_idx] = val
            img = model.decoder(z.to(device))
            images.append(img.cpu())

    return torch.cat(images, dim=0)


# ============================================
# 5. Reconstruction Visualization
# ============================================
print("\n[5] Reconstruction Visualization")
print("-" * 40)


def visualize_reconstructions(model, dataloader, num_samples=8, device='cpu'):
    """Show original vs reconstructed images"""
    model.eval()

    # Get batch
    data, _ = next(iter(dataloader))
    data = data[:num_samples].to(device)

    with torch.no_grad():
        recon, _, _ = model(data)

    # Create comparison grid
    comparison = torch.cat([data, recon], dim=0)
    grid = vutils.make_grid(comparison.cpu(), nrow=num_samples, normalize=True, padding=2)

    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Original (top) vs Reconstructed (bottom)')
    plt.savefig('vae_reconstruction.png', dpi=150)
    plt.close()
    print("Reconstruction comparison saved to vae_reconstruction.png")


# ============================================
# 6. Beta-VAE
# ============================================
print("\n[6] Beta-VAE for Disentanglement")
print("-" * 40)


class BetaVAE(VAE):
    """Beta-VAE with higher KL weight for disentanglement"""
    def __init__(self, in_channels=1, latent_dim=10, beta=4.0):
        super().__init__(in_channels, latent_dim)
        self.beta = beta

    def loss(self, x, x_recon, mu, log_var):
        return vae_loss(x, x_recon, mu, log_var, self.beta)


print(f"Beta-VAE class defined with configurable beta parameter")


# ============================================
# 7. Conditional VAE (CVAE)
# ============================================
print("\n[7] Conditional VAE")
print("-" * 40)


class CVAEEncoder(nn.Module):
    """CVAE Encoder with label conditioning"""
    def __init__(self, in_channels=1, latent_dim=20, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x, y):
        # Concat one-hot label as additional channels
        y_expanded = y.view(-1, self.num_classes, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, y_expanded], dim=1)

        h = self.conv_layers(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """CVAE Decoder with label conditioning"""
    def __init__(self, latent_dim=20, out_channels=1, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.fc = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        z_cond = torch.cat([z, y], dim=1)
        h = self.fc(z_cond)
        h = h.view(-1, 64, 7, 7)
        return self.deconv_layers(h)


class CVAE(nn.Module):
    """Conditional VAE for digit generation"""
    def __init__(self, in_channels=1, latent_dim=20, num_classes=10):
        super().__init__()
        self.encoder = CVAEEncoder(in_channels, latent_dim, num_classes)
        self.decoder = CVAEDecoder(latent_dim, in_channels, num_classes)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, label):
        y = F.one_hot(label, self.num_classes).float()
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z, y)
        return x_recon, mu, log_var

    def generate(self, label, num_samples, device):
        """Generate specific digit"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = F.one_hot(label.expand(num_samples), self.num_classes).float().to(device)
        return self.decoder(z, y)


print("CVAE class defined for conditional generation")


# ============================================
# 8. Training Example
# ============================================
print("\n[8] Training VAE on MNIST")
print("-" * 40)

# Hyperparameters
latent_dim = 20
batch_size = 128
epochs = 10
lr = 1e-3

# Data
transform = transforms.ToTensor()
print("Loading MNIST dataset...")
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
vae = VAE(in_channels=1, latent_dim=latent_dim).to(device)
print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Train
print("\nTraining VAE...")
history = train_vae(vae, train_loader, epochs=epochs, lr=lr, beta=1.0)

# Visualizations
print("\nGenerating visualizations...")

# 1. Reconstruction
visualize_reconstructions(vae, test_loader, num_samples=10, device=device)

# 2. Generated samples
vae.eval()
with torch.no_grad():
    samples = vae.generate(64, device)
    grid = vutils.make_grid(samples.cpu(), nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Generated Samples')
    plt.savefig('vae_samples.png', dpi=150)
    plt.close()
print("Generated samples saved to vae_samples.png")

# 3. Latent space (for 2D visualization, use latent_dim=2)
vae_2d = VAE(in_channels=1, latent_dim=2).to(device)
print("\nTraining 2D VAE for visualization...")
_ = train_vae(vae_2d, train_loader, epochs=5, lr=lr)
visualize_latent_space(vae_2d, test_loader, device)
generate_manifold(vae_2d, n=20, latent_dim=2, device=device)

# 4. Loss curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['total'], label='Total')
plt.plot(history['recon'], label='Reconstruction')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['kl'], label='KL Divergence', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('KL Divergence')

plt.tight_layout()
plt.savefig('vae_loss.png', dpi=150)
plt.close()
print("Loss curves saved to vae_loss.png")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("VAE Summary")
print("=" * 60)

summary = """
Key Concepts:
1. VAE: Probabilistic latent variable model
2. Reparameterization: z = mu + sigma * epsilon
3. ELBO: Reconstruction + KL Divergence
4. Beta-VAE: beta > 1 for disentanglement
5. CVAE: Conditional generation

Loss Function:
L = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
  = BCE(x, x_recon) + beta * (-0.5 * sum(1 + log(var) - mu^2 - var))

Latent Space Properties:
- Continuous and structured
- Can interpolate between samples
- Each dimension captures a factor of variation

Output Files:
- vae_samples.png: Generated samples
- vae_reconstruction.png: Original vs reconstructed
- vae_latent_space.png: 2D latent space visualization
- vae_manifold.png: Digit manifold
- vae_loss.png: Training loss curves
"""
print(summary)
print("=" * 60)
