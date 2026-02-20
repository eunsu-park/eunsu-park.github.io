"""
Stable Diffusion Fine-tuning with LoRA (Low-Rank Adaptation)

This script demonstrates how to fine-tune a diffusion model using LoRA,
a parameter-efficient fine-tuning technique that adds trainable low-rank
matrices to attention layers while keeping the base model frozen.

Key Concepts:
- LoRA (Low-Rank Adaptation): Inject trainable rank decomposition matrices
- Diffusion Models: Iterative denoising process for image generation
- Parameter-Efficient Fine-tuning: Train only a small subset of parameters
- UNet Architecture: Core denoising network in diffusion models
- Text-to-Image: Conditioning image generation with text prompts

Requirements:
    pip install torch torchvision diffusers transformers accelerate safetensors
"""

import argparse
import os
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    print("Please install required packages:")
    print("pip install diffusers transformers accelerate")
    exit(1)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer that can be injected into linear layers.

    LoRA decomposes weight updates into two low-rank matrices:
    W' = W + BA, where B is (out_features, rank) and A is (rank, in_features)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: (BA) * x

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA output scaled by alpha/rank
        """
        # Compute low-rank transformation
        result = F.linear(x, self.lora_A)  # (..., rank)
        result = F.linear(result, self.lora_B)  # (..., out_features)
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with optional LoRA adaptation."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation + LoRA adaptation
        return self.linear(x) + self.lora(x)


def inject_lora_to_attention(unet: UNet2DConditionModel, rank: int = 4,
                             alpha: float = 1.0) -> nn.Module:
    """
    Inject LoRA layers into the attention modules of UNet.

    In Stable Diffusion, attention modules have Q, K, V projection layers
    and an output projection layer. We add LoRA to all of these.

    Args:
        unet: UNet2DConditionModel from diffusers
        rank: Rank of LoRA matrices
        alpha: Scaling factor for LoRA

    Returns:
        Modified UNet with LoRA layers
    """
    # Freeze all parameters first
    for param in unet.parameters():
        param.requires_grad = False

    lora_count = 0

    # Iterate through all attention modules
    for name, module in unet.named_modules():
        if "attn" in name and isinstance(module, nn.Linear):
            # Determine parent module and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]

            parent = unet
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            # Replace Linear with LinearWithLoRA
            original_linear = getattr(parent, attr_name)
            setattr(parent, attr_name, LinearWithLoRA(original_linear, rank, alpha))
            lora_count += 1

    print(f"Injected LoRA into {lora_count} linear layers")
    return unet


def get_lora_parameters(model: nn.Module) -> list:
    """Extract only LoRA parameters for training."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def save_lora_weights(model: nn.Module, save_path: str):
    """Save only LoRA weights (not the full model)."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_state_dict[name] = param.cpu()

    torch.save(lora_state_dict, save_path)
    print(f"LoRA weights saved to {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")


def load_lora_weights(model: nn.Module, load_path: str):
    """Load LoRA weights into the model."""
    lora_state_dict = torch.load(load_path)

    # Load weights
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"LoRA weights loaded from {load_path}")


class DummyDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.

    In practice, you would use:
    - Custom image-caption pairs
    - Datasets from HuggingFace (e.g., lambdalabs/pokemon-blip-captions)
    - Your own domain-specific data
    """

    def __init__(self, size: int = 100, img_size: int = 512):
        self.size = size
        self.img_size = img_size

        # Dummy prompts
        self.prompts = [
            "a beautiful landscape with mountains",
            "a cute cat sitting on a chair",
            "abstract digital art",
            "a futuristic city at night",
            "a serene beach at sunset"
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Generate random image (in practice, load real images)
        image = torch.randn(3, self.img_size, self.img_size)

        # Random prompt
        prompt = self.prompts[idx % len(self.prompts)]

        return image, prompt


def train_step(unet: nn.Module, noise_scheduler: DDPMScheduler,
               text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer,
               optimizer: torch.optim.Optimizer, batch: Tuple,
               device: torch.device) -> float:
    """
    Single training step for diffusion model fine-tuning.

    The training objective is to predict the noise added to the image.
    """
    images, prompts = batch
    images = images.to(device)

    # Encode text prompts
    text_inputs = tokenizer(
        prompts, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    # Sample random timesteps
    batch_size = images.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    ).long()

    # Add noise to images
    noise = torch.randn_like(images)
    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

    # Predict noise with UNet
    model_output = unet(noisy_images, timesteps, text_embeddings).sample

    # Compute loss (MSE between predicted and actual noise)
    loss = F.mse_loss(model_output, noise)

    # Backward pass (only LoRA parameters are updated)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def generate_image(pipeline: StableDiffusionPipeline, prompt: str,
                  save_path: str, num_inference_steps: int = 50):
    """Generate image with the fine-tuned model."""
    with torch.no_grad():
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5
        ).images[0]

    image.save(save_path)
    print(f"Generated image saved to {save_path}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained Stable Diffusion model
    print("\nLoading pretrained Stable Diffusion model...")
    print("Note: First run will download ~4GB of model weights")

    model_id = "runwayml/stable-diffusion-v1-5"

    # Load components separately for fine-tuning
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze text encoder (we only fine-tune UNet)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    # Inject LoRA into UNet attention layers
    print("\nInjecting LoRA layers...")
    unet = inject_lora_to_attention(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    unet.to(device)

    # Print parameter statistics
    total_params = count_parameters(unet, trainable_only=False)
    trainable_params = count_parameters(unet, trainable_only=True)
    print(f"\nUNet Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable (LoRA): {trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Setup training
    dataset = DummyDataset(size=args.num_samples, img_size=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    lora_params = get_lora_parameters(unet)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    # Training loop
    print("\n" + "="*60)
    print("Starting LoRA Fine-tuning")
    print("="*60)
    print("Note: This is a demonstration with dummy data.")
    print("For real fine-tuning, replace DummyDataset with actual image-caption pairs.")

    unet.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(
                unet, noise_scheduler, text_encoder,
                tokenizer, optimizer, batch, device
            )
            epoch_loss += loss

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Save LoRA weights
    os.makedirs(args.output_dir, exist_ok=True)
    lora_path = os.path.join(args.output_dir, "lora_weights.pt")
    save_lora_weights(unet, lora_path)

    # Generate sample images with fine-tuned model
    print("\n" + "="*60)
    print("Generating Sample Images")
    print("="*60)

    # Create pipeline with fine-tuned UNet
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None  # Disable for faster inference
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Test prompts
    test_prompts = [
        "a beautiful landscape with mountains",
        "a cute cat sitting on a chair",
        "abstract digital art with vibrant colors"
    ]

    for idx, prompt in enumerate(test_prompts):
        save_path = os.path.join(args.output_dir, f"generated_{idx+1}.png")
        print(f"\nGenerating: '{prompt}'")
        generate_image(pipeline, prompt, save_path, num_inference_steps=30)

    # Demonstrate loading LoRA weights
    print("\n" + "="*60)
    print("Demonstrating LoRA Weight Loading")
    print("="*60)

    # Create a fresh UNet and load LoRA weights
    unet_fresh = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet_fresh = inject_lora_to_attention(unet_fresh, rank=args.lora_rank, alpha=args.lora_alpha)
    load_lora_weights(unet_fresh, lora_path)

    print("\nFine-tuning complete!")
    print(f"LoRA weights saved to: {lora_path}")
    print(f"Generated images saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (use 1 for limited VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of training samples (dummy data)")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="Rank of LoRA matrices (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                       help="LoRA scaling factor")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./lora_output",
                       help="Directory to save LoRA weights and generated images")

    args = parser.parse_args()

    # Validate CUDA availability for Stable Diffusion
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Stable Diffusion is extremely slow on CPU.")
        print("This script is designed for GPU usage. Proceed at your own risk.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    main(args)
