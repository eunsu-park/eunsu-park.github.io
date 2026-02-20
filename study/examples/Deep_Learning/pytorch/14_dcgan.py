"""
DCGAN (Deep Convolutional GAN) Implementation

This script implements DCGAN following "Unsupervised Representation Learning
with Deep Convolutional Generative Adversarial Networks" (Radford et al., 2015).

Key architecture guidelines:
- Replace pooling with strided convolutions (discriminator) and transposed convolutions (generator)
- Use batch normalization in both generator and discriminator
- Remove fully connected hidden layers
- Use ReLU in generator (except output: Tanh), LeakyReLU in discriminator
- Proper weight initialization

References:
- Radford et al. (2015): https://arxiv.org/abs/1511.06434
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================================
# Weight Initialization
# ============================================================================

def weights_init(m):
    """
    Custom weight initialization as described in DCGAN paper.

    - Conv/ConvTranspose layers: mean=0, std=0.02
    - BatchNorm layers: mean=1, std=0.02
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================================
# Generator
# ============================================================================

class Generator(nn.Module):
    """
    DCGAN Generator: transforms latent vector z to image.

    Architecture:
    - Input: [batch_size, nz, 1, 1] latent vector
    - 4 transposed convolution blocks with BatchNorm and ReLU
    - Output: [batch_size, nc, 64, 64] image with Tanh activation

    Args:
        nz: size of latent vector (input noise dimension)
        ngf: number of generator filters in first layer
        nc: number of output channels (1 for grayscale, 3 for RGB)
    """
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input: [batch, nz, 1, 1]
            # Output: [batch, ngf*8, 4, 4]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # [batch, ngf*8, 4, 4] -> [batch, ngf*4, 8, 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # [batch, ngf*4, 8, 8] -> [batch, ngf*2, 16, 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # [batch, ngf*2, 16, 16] -> [batch, ngf, 32, 32]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [batch, ngf, 32, 32] -> [batch, nc, 64, 64]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: [batch_size, nz, 1, 1] latent vectors
        Returns:
            [batch_size, nc, 64, 64] generated images
        """
        return self.main(z)


# ============================================================================
# Discriminator
# ============================================================================

class Discriminator(nn.Module):
    """
    DCGAN Discriminator: classifies images as real or fake.

    Architecture:
    - Input: [batch_size, nc, 64, 64] image
    - 4 strided convolution blocks with BatchNorm and LeakyReLU
    - Output: [batch_size, 1, 1, 1] probability (via Sigmoid)

    Args:
        nc: number of input channels (1 for grayscale, 3 for RGB)
        ndf: number of discriminator filters in first layer
    """
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: [batch, nc, 64, 64]
            # Output: [batch, ndf, 32, 32]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf, 32, 32] -> [batch, ndf*2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*2, 16, 16] -> [batch, ndf*4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*4, 8, 8] -> [batch, ndf*8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*8, 4, 4] -> [batch, 1, 1, 1]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, nc, 64, 64] images
        Returns:
            [batch_size, 1, 1, 1] probability of being real
        """
        return self.main(x)


# ============================================================================
# Training
# ============================================================================

def train_dcgan(epochs=25, batch_size=128, nz=100, lr=0.0002, beta1=0.5, device='cuda'):
    """
    Train DCGAN on MNIST dataset.

    Args:
        epochs: number of training epochs
        batch_size: batch size
        nz: size of latent vector
        lr: learning rate
        beta1: beta1 parameter for Adam optimizer
        device: 'cuda' or 'cpu'

    Returns:
        generator: trained Generator model
        discriminator: trained Discriminator model
        losses: dict with generator and discriminator losses
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Data preparation
    # Resize MNIST to 64x64 and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize models
    netG = Generator(nz=nz, ngf=64, nc=1).to(device)
    netD = Discriminator(nc=1, ndf=64).to(device)

    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    print("Generator architecture:")
    print(netG)
    print("\nDiscriminator architecture:")
    print(netD)

    # Loss function and optimizers
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Labels for real and fake images
    real_label = 1.0
    fake_label = 0.0

    # Lists to track losses
    G_losses = []
    D_losses = []

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)

            # ================================================================
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ================================================================
            netD.zero_grad()

            # Train with real images
            label = torch.full((batch_size_actual,), real_label, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake images
            noise = torch.randn(batch_size_actual, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Total discriminator loss
            errD = errD_real + errD_fake
            optimizerD.step()

            # ================================================================
            # (2) Update Generator: maximize log(D(G(z)))
            # ================================================================
            netG.zero_grad()
            label.fill_(real_label)  # Fake images should be classified as real
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Save losses
            if i % 50 == 0:
                G_losses.append(errG.item())
                D_losses.append(errD.item())

        # Print statistics
        print(f'[{epoch+1}/{epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Generate and save images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()

            # Create image grid
            grid = make_grid(fake_samples, nrow=8, normalize=True)

            # Visualize
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.title(f'Generated Images - Epoch {epoch+1}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'dcgan_samples_epoch_{epoch+1}.png')
            plt.close()

    print("\nTraining completed!")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Losses')
    plt.savefig('dcgan_losses.png')
    plt.close()

    return netG, netD, {'G_losses': G_losses, 'D_losses': D_losses}


# ============================================================================
# Image Generation
# ============================================================================

@torch.no_grad()
def generate_images(generator, num_images=64, nz=100, device='cuda'):
    """
    Generate images using trained generator.

    Args:
        generator: trained Generator model
        num_images: number of images to generate
        nz: size of latent vector
        device: 'cuda' or 'cpu'

    Returns:
        [num_images, nc, 64, 64] generated images
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    generator.eval()

    # Sample random noise
    noise = torch.randn(num_images, nz, 1, 1, device=device)

    # Generate images
    fake_images = generator(noise)

    return fake_images.cpu()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Train DCGAN
    netG, netD, losses = train_dcgan(
        epochs=25,
        batch_size=128,
        nz=100,
        lr=0.0002,
        beta1=0.5,
        device='cuda'
    )

    # Generate final samples
    print("\nGenerating final samples...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = generate_images(netG, num_images=64, nz=100, device=device)

    # Visualize final samples
    grid = make_grid(samples, nrow=8, normalize=True)

    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title('DCGAN Generated Samples (Final)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('dcgan_final_samples.png')
    plt.show()

    # Optional: Save models
    torch.save(netG.state_dict(), 'dcgan_generator.pth')
    torch.save(netD.state_dict(), 'dcgan_discriminator.pth')
    print("\nModels saved to dcgan_generator.pth and dcgan_discriminator.pth")
