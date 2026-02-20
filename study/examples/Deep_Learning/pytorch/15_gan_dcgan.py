"""
15. GAN and DCGAN Implementation

GAN (Generative Adversarial Networks) and DCGAN (Deep Convolutional GAN)
implementation for MNIST digit generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 60)
print("GAN and DCGAN Implementation")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# 1. Basic GAN (Fully Connected)
# ============================================
print("\n[1] Basic GAN Architecture")
print("-" * 40)


class Generator(nn.Module):
    """Basic Generator using fully connected layers"""
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    """Basic Discriminator using fully connected layers"""
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_size = int(np.prod(img_shape))

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


# Test basic GAN
G = Generator(latent_dim=100)
D = Discriminator()
z = torch.randn(4, 100)
fake_imgs = G(z)
validity = D(fake_imgs)
print(f"Generator output shape: {fake_imgs.shape}")
print(f"Discriminator output shape: {validity.shape}")


# ============================================
# 2. DCGAN Architecture
# ============================================
print("\n[2] DCGAN Architecture")
print("-" * 40)


class DCGANGenerator(nn.Module):
    """DCGAN Generator with transposed convolutions

    Architecture for 64x64 output:
    z (latent_dim,) -> (ngf*8, 4, 4) -> (ngf*4, 8, 8) -> (ngf*2, 16, 16)
    -> (ngf, 32, 32) -> (nc, 64, 64)
    """
    def __init__(self, latent_dim=100, ngf=64, nc=1):
        super().__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # Input: z (latent_dim,) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 32, 32) -> (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator with strided convolutions

    Architecture for 64x64 input:
    (nc, 64, 64) -> (ndf, 32, 32) -> (ndf*2, 16, 16) -> (ndf*4, 8, 8)
    -> (ndf*8, 4, 4) -> (1,)
    """
    def __init__(self, nc=1, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            # (nc, 64, 64) -> (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 32, 32) -> (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 16, 16) -> (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 8, 8) -> (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 4, 4) -> (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        return self.main(img).view(-1, 1)


# Test DCGAN
dc_G = DCGANGenerator(latent_dim=100, ngf=64, nc=1)
dc_D = DCGANDiscriminator(nc=1, ndf=64)
z = torch.randn(4, 100)
fake_imgs = dc_G(z)
validity = dc_D(fake_imgs)
print(f"DCGAN Generator output: {fake_imgs.shape}")
print(f"DCGAN Discriminator output: {validity.shape}")


# ============================================
# 3. Loss Functions
# ============================================
print("\n[3] GAN Loss Functions")
print("-" * 40)


def bce_loss(output, target):
    """Binary Cross Entropy Loss (vanilla GAN)"""
    return F.binary_cross_entropy(output, target)


def wasserstein_loss(output, is_real):
    """Wasserstein Loss (WGAN)

    D tries to maximize: E[D(real)] - E[D(fake)]
    G tries to maximize: E[D(fake)]
    """
    if is_real:
        return -torch.mean(output)
    else:
        return torch.mean(output)


def hinge_loss(output, is_real):
    """Hinge Loss

    D: max(0, 1 - D(real)) + max(0, 1 + D(fake))
    G: -E[D(fake)]
    """
    if is_real:
        return torch.mean(F.relu(1.0 - output))
    else:
        return torch.mean(F.relu(1.0 + output))


# ============================================
# 4. Gradient Penalty (WGAN-GP)
# ============================================
print("\n[4] Gradient Penalty (WGAN-GP)")
print("-" * 40)


def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_imgs.size(0)

    # Random interpolation between real and fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated.requires_grad_(True)

    # Get discriminator output
    d_interpolated = discriminator(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


print("Gradient penalty function defined")


# ============================================
# 5. Training Loop
# ============================================
print("\n[5] Training Loop")
print("-" * 40)


def train_gan(generator, discriminator, dataloader, epochs=5, latent_dim=100, lr=0.0002):
    """Train basic GAN on MNIST"""
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            optimizer_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return g_losses, d_losses


# ============================================
# 6. Sample Generation and Visualization
# ============================================
print("\n[6] Sample Generation")
print("-" * 40)


def generate_samples(generator, num_samples=64, latent_dim=100):
    """Generate samples from trained generator"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        samples = generator(z)
    return samples


def save_samples(samples, filename='generated_samples.png', nrow=8):
    """Save generated samples as image grid"""
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    grid = vutils.make_grid(samples.cpu(), nrow=nrow, normalize=False, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved to {filename}")


def latent_interpolation(generator, z1, z2, steps=10):
    """Interpolate between two latent vectors"""
    generator.eval()
    images = []
    with torch.no_grad():
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z.unsqueeze(0).to(device))
            images.append(img)
    return torch.cat(images, dim=0)


def spherical_interpolation(z1, z2, alpha):
    """Spherical linear interpolation (slerp)"""
    z1_norm = z1 / z1.norm()
    z2_norm = z2 / z2.norm()
    omega = torch.acos((z1_norm * z2_norm).sum())
    so = torch.sin(omega)
    return (torch.sin((1 - alpha) * omega) / so) * z1 + (torch.sin(alpha * omega) / so) * z2


# ============================================
# 7. Training Example
# ============================================
print("\n[7] Training Example (Basic GAN on MNIST)")
print("-" * 40)

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 5  # Increase for better results

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

print("Loading MNIST dataset...")
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

# Models
G = Generator(latent_dim=latent_dim, img_shape=(1, 28, 28)).to(device)
D = Discriminator(img_shape=(1, 28, 28)).to(device)

print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

# Train
print("\nTraining...")
g_losses, d_losses = train_gan(G, D, train_loader, epochs=epochs, latent_dim=latent_dim)

# Generate samples
print("\nGenerating samples...")
samples = generate_samples(G, num_samples=64, latent_dim=latent_dim)
save_samples(samples, 'gan_mnist_samples.png')

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator', alpha=0.7)
plt.plot(d_losses, label='Discriminator', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.savefig('gan_loss.png')
plt.close()
print("Loss plot saved to gan_loss.png")


# ============================================
# 8. Latent Space Exploration
# ============================================
print("\n[8] Latent Space Exploration")
print("-" * 40)

# Interpolation
z1 = torch.randn(latent_dim)
z2 = torch.randn(latent_dim)
interp_imgs = latent_interpolation(G, z1, z2, steps=10)
save_samples(interp_imgs, 'gan_interpolation.png', nrow=10)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("GAN/DCGAN Summary")
print("=" * 60)

summary = """
Key Concepts:
1. GAN: Generator vs Discriminator adversarial training
2. DCGAN: Convolutional architecture with BatchNorm, LeakyReLU
3. Loss: BCE (vanilla), Wasserstein, Hinge
4. WGAN-GP: Gradient penalty for stable training

Training Tips:
- Adam with beta1=0.5
- Learning rate: 0.0001 ~ 0.0002
- Label smoothing for stability
- Monitor D/G balance

Common Issues:
- Mode collapse: G produces limited variety
- Training instability: D too strong
- Vanishing gradients: Use WGAN/WGAN-GP

Output Files:
- gan_mnist_samples.png: Generated samples
- gan_loss.png: Training loss curves
- gan_interpolation.png: Latent space interpolation
"""
print(summary)
print("=" * 60)
