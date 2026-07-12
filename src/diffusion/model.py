"""
Action-Conditioned Diffusion Model for GameNGen
Based on Stable Diffusion v1.4 with modifications for action conditioning
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel


class ActionEmbedding(nn.Module):
    """Embed discrete actions into continuous vectors"""

    def __init__(self, num_actions: int, embedding_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        # Learnable action embeddings
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (batch_size, seq_len) - discrete action indices
        Returns:
            (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(actions)


class NoiseAugmentationEmbedding(nn.Module):
    """Embedding for noise augmentation levels"""

    def __init__(self, num_buckets: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim

        # Learnable noise level embeddings
        self.embedding = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, noise_levels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_levels: (batch_size,) - discrete noise level indices [0, num_buckets)
        Returns:
            (batch_size, embedding_dim)
        """
        return self.embedding(noise_levels)


class ActionConditionedDiffusionModel(nn.Module):
    """
    Action-conditioned diffusion model based on Stable Diffusion v1.4

    Modifications from original SD:
    1. Action conditioning via cross-attention (replaces text conditioning)
    2. Frame history conditioning via latent concatenation
    3. Noise augmentation embeddings
    4. Optimized for real-time inference (4-step DDIM)
    """

    def __init__(
        self,
        pretrained_model_name: str = "CompVis/stable-diffusion-v1-4",
        num_actions: int = 3,
        action_embedding_dim: int = 128,
        context_length: int = 32,
        num_noise_buckets: int = 10,
        max_noise_level: float = 0.7,
        cfg_drop_prob: float = 0.1,
        prediction_type: str = "v_prediction",
        decoder_checkpoint: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.action_embedding_dim = action_embedding_dim
        self.context_length = context_length
        self.num_noise_buckets = num_noise_buckets
        self.max_noise_level = max_noise_level
        self.cfg_drop_prob = cfg_drop_prob
        self.prediction_type = prediction_type
        self.device = device
        self.dtype = dtype

        print(f"Loading Stable Diffusion v1.4 from {pretrained_model_name}...")

        # Load pretrained components
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, subfolder="vae", torch_dtype=dtype
        ).to(device)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet", torch_dtype=dtype
        ).to(device)

        # Initialize scheduler for training and inference
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler"
        )
        self.noise_scheduler.register_to_config(prediction_type=prediction_type)

        # Freeze VAE encoder during training (we'll fine-tune decoder separately)
        for param in self.vae.parameters():
            param.requires_grad = False

        if decoder_checkpoint:
            checkpoint = torch.load(decoder_checkpoint, map_location=device, weights_only=False)
            if checkpoint.get("base_model") not in (None, pretrained_model_name):
                raise ValueError("decoder checkpoint was trained from a different base model")
            self.vae.decoder.load_state_dict(checkpoint["decoder"])

        # Action embedding layer
        self.action_embedding = ActionEmbedding(num_actions, action_embedding_dim).to(
            device
        )

        # Noise augmentation embedding
        self.noise_aug_embedding = NoiseAugmentationEmbedding(
            num_noise_buckets, action_embedding_dim
        ).to(device)

        # Project action embeddings to cross-attention dimension
        # SD's cross-attention expects dimension 768
        self.action_proj = nn.Linear(action_embedding_dim, 768).to(device)

        # Modify U-Net input channels to accept concatenated context frames
        # Original: 4 channels (latent)
        # New: 4 + 4*context_length channels (current + context frames)
        original_in_channels = self.unet.config.in_channels
        new_in_channels = original_in_channels * (1 + context_length)

        # Replace first conv layer to accept more channels
        old_conv = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(
            new_in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dtype=dtype,
        ).to(device)

        # Initialize new conv weights: copy original weights for first 4 channels,
        # initialize rest with small random values
        with torch.no_grad():
            # Copy weights for the first 4 channels (original latent input)
            self.unet.conv_in.weight[:, :original_in_channels] = old_conv.weight
            # Initialize weights for context frames with small values
            nn.init.kaiming_normal_(
                self.unet.conv_in.weight[:, original_in_channels:],
                mode="fan_out",
                nonlinearity="relu",
            )
            self.unet.conv_in.weight[:, original_in_channels:] *= 0.01  # Scale down

            if old_conv.bias is not None:
                self.unet.conv_in.bias.copy_(old_conv.bias)

        print("Model initialization complete!")
        print(
            f"  VAE: {sum(p.numel() for p in self.vae.parameters()):,} parameters (frozen)"
        )
        print(f"  U-Net: {sum(p.numel() for p in self.unet.parameters()):,} parameters")
        print(
            f"  Action Embedding: {sum(p.numel() for p in self.action_embedding.parameters()):,} parameters"
        )
        print(
            f"  Total trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters"
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames to latent space using VAE

        Args:
            frames: (batch_size, num_frames, channels, height, width) or
                   (batch_size, channels, height, width)
        Returns:
            latents: encoded latents
        """
        # Handle both single frames and sequences
        if frames.dim() == 5:
            batch_size, num_frames, c, h, w = frames.shape
            frames = frames.reshape(batch_size * num_frames, c, h, w)
            need_reshape = True
        else:
            batch_size, c, h, w = frames.shape
            need_reshape = False

        # Normalize to [-1, 1] if needed
        frames = frames.to(device=self.device, dtype=self.dtype)
        if frames.max() > 1.0:
            frames = frames / 127.5 - 1.0

        # Encode
        with torch.no_grad():
            latents = self.vae.encode(frames).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Reshape back if needed
        if need_reshape:
            latent_c, latent_h, latent_w = latents.shape[1:]
            latents = latents.reshape(
                batch_size, num_frames, latent_c, latent_h, latent_w
            )

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image space using VAE

        Args:
            latents: (batch_size, channels, height, width)
        Returns:
            images: (batch_size, 3, height, width) in range [0, 255]
        """
        # Unscale latents
        latents = latents.to(device=self.device, dtype=self.dtype)
        latents = latents / self.vae.config.scaling_factor

        # Decode
        with torch.no_grad():
            images = self.vae.decode(latents).sample

        # Convert from [-1, 1] to [0, 255]
        images = (images + 1.0) * 127.5
        images = images.clamp(0, 255)

        return images

    def add_noise_augmentation(
        self, latents: torch.Tensor, noise_level: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise augmentation to latents (for training stability)

        Args:
            latents: (batch_size, num_frames, channels, height, width)
            noise_level: (batch_size,) - noise levels in [0, max_noise_level]
                        If None, sample uniformly
        Returns:
            noised_latents: latents with added noise
            noise_level_ids: discrete noise level bucket IDs
        """
        batch_size = latents.shape[0]

        # Sample noise level if not provided
        if noise_level is None:
            noise_level = (
                torch.rand(batch_size, device=self.device) * self.max_noise_level
            )

        # Convert to discrete buckets
        noise_level_ids = (
            noise_level * self.num_noise_buckets / self.max_noise_level
        ).long()
        noise_level_ids = noise_level_ids.clamp(0, self.num_noise_buckets - 1)

        # Add Gaussian noise
        noise = torch.randn_like(latents)
        noised_latents = latents + noise_level.view(-1, 1, 1, 1, 1) * noise

        return noised_latents, noise_level_ids

    def forward(
        self,
        target_frame: torch.Tensor,
        context_frames: torch.Tensor,
        actions: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass for training

        Args:
            target_frame: (batch_size, 3, H, W) - frame to predict
            context_frames: (batch_size, context_length, 3, H, W) - past frames
            actions: (batch_size, context_length) - past actions
            noise_level: (batch_size,) - optional noise augmentation level
            timesteps: (batch_size,) - diffusion timesteps, if None will sample
            return_dict: whether to return dict

        Returns:
            loss or dict with loss and predictions
        """
        batch_size = target_frame.shape[0]

        # Encode target frame to latents
        target_latents = self.encode_frames(target_frame)

        # Encode context frames
        context_latents = self.encode_frames(context_frames)  # (B, T, C, H, W)

        # Apply noise augmentation to context frames
        if noise_level is not None or self.training:
            context_latents, noise_level_ids = self.add_noise_augmentation(
                context_latents, noise_level
            )
        else:
            noise_level_ids = torch.zeros(
                batch_size, device=self.device, dtype=torch.long
            )

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Train the observation-unconditional branch required for CFG.  Action
        # tokens deliberately remain conditioned, matching the paper setup.
        if self.training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(batch_size, device=self.device) < self.cfg_drop_prob
            context_latents = context_latents.masked_fill(
                drop_mask.view(-1, 1, 1, 1, 1), 0
            )
        else:
            drop_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        # Concatenate noisy target with context frames
        # Flatten context: (B, T, C, H, W) -> (B, T*C, H, W)
        context_flat = context_latents.reshape(
            batch_size,
            self.context_length * context_latents.shape[2],
            context_latents.shape[3],
            context_latents.shape[4],
        )

        # Concatenate in channel dimension
        unet_input = torch.cat([noisy_latents, context_flat], dim=1)

        # Embed actions
        action_embeds = self.action_embedding(actions)  # (B, T, D)

        # Add noise level embedding to action embeddings
        noise_embeds = self.noise_aug_embedding(noise_level_ids)  # (B, D)
        noise_embeds = noise_embeds.unsqueeze(1)  # (B, 1, D)

        # Concatenate noise embedding with action embeddings
        action_embeds = torch.cat([noise_embeds, action_embeds], dim=1)  # (B, T+1, D)

        # Project to cross-attention dimension
        encoder_hidden_states = self.action_proj(action_embeds)  # (B, T+1, 768)

        # U-Net prediction
        model_output = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        ).sample

        # Compute loss (velocity parameterization)
        # v = alpha_t * noise - sigma_t * x_0
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=target_latents.device, dtype=target_latents.dtype
        )
        alpha_t = alphas_cumprod[timesteps].sqrt()
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt()

        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)

        v_target = alpha_t * noise - sigma_t * target_latents
        v_pred = model_output

        loss = F.mse_loss(v_pred, v_target)

        if return_dict:
            return {
                "loss": loss,
                "v_pred": v_pred,
                "v_target": v_target,
                "context_dropped": drop_mask,
            }

        return loss

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        actions: torch.Tensor,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.5,
        return_latents: bool = False,
    ) -> torch.Tensor:
        """
        Generate next frame given context

        Args:
            context_frames: (batch_size, context_length, 3, H, W)
            actions: (batch_size, context_length)
            num_inference_steps: number of DDIM steps
            guidance_scale: CFG scale
            return_latents: whether to return latents instead of images

        Returns:
            generated_frame: (batch_size, 3, H, W)
        """
        batch_size = context_frames.shape[0]

        # Encode context frames (no noise augmentation during inference)
        context_latents = self.encode_frames(context_frames)

        # Flatten context
        context_flat = context_latents.reshape(
            batch_size,
            self.context_length * context_latents.shape[2],
            context_latents.shape[3],
            context_latents.shape[4],
        )

        # Embed actions
        action_embeds = self.action_embedding(actions)

        # Add noise level embedding (0 during inference)
        noise_level_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        noise_embeds = self.noise_aug_embedding(noise_level_ids).unsqueeze(1)

        action_embeds = torch.cat([noise_embeds, action_embeds], dim=1)
        encoder_hidden_states = self.action_proj(action_embeds)

        # Initialize latents with noise
        latents = torch.randn(
            batch_size,
            4,
            context_latents.shape[3],
            context_latents.shape[4],
            device=self.device,
            dtype=self.dtype,
        )

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # DDIM sampling loop
        for t in timesteps:
            # Concatenate with context
            unet_input = torch.cat([latents, context_flat], dim=1)

            # Predict noise
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample

            # CFG (if guidance_scale > 1.0)
            if guidance_scale > 1.0:
                # Unconditional prediction (with dropped context)
                unet_input_uncond = torch.cat(
                    [latents, torch.zeros_like(context_flat)], dim=1
                )

                noise_pred_uncond = self.unet(
                    unet_input_uncond, t, encoder_hidden_states=encoder_hidden_states
                ).sample

                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # Compute previous sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        if return_latents:
            return latents

        # Decode to image
        images = self.decode_latents(latents)

        return images

    def save_pretrained(self, save_path: str):
        """Save model checkpoint"""
        import os

        os.makedirs(save_path, exist_ok=True)

        torch.save(
            {
                "unet": self.unet.state_dict(),
                "action_embedding": self.action_embedding.state_dict(),
                "noise_aug_embedding": self.noise_aug_embedding.state_dict(),
                "action_proj": self.action_proj.state_dict(),
                "config": {
                    "num_actions": self.num_actions,
                    "action_embedding_dim": self.action_embedding_dim,
                    "context_length": self.context_length,
                    "num_noise_buckets": self.num_noise_buckets,
                    "max_noise_level": self.max_noise_level,
                    "cfg_drop_prob": self.cfg_drop_prob,
                    "prediction_type": self.prediction_type,
                },
            },
            os.path.join(save_path, "model.pt"),
        )

        print(f"Model saved to {save_path}")

    def load_pretrained(self, load_path: str):
        """Load model checkpoint"""
        import os

        checkpoint = torch.load(
            os.path.join(load_path, "model.pt"), map_location=self.device
        )

        self.unet.load_state_dict(checkpoint["unet"])
        self.action_embedding.load_state_dict(checkpoint["action_embedding"])
        self.noise_aug_embedding.load_state_dict(checkpoint["noise_aug_embedding"])
        self.action_proj.load_state_dict(checkpoint["action_proj"])

        print(f"Model loaded from {load_path}")


if __name__ == "__main__":
    # Test model creation
    print("Testing ActionConditionedDiffusionModel...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionConditionedDiffusionModel(
        num_actions=3,
        context_length=32,
        device=device,
        dtype=torch.float32,  # Use float32 for testing
    )

    print("\nModel created successfully!")
    print(f"Device: {device}")

    # Test forward pass
    batch_size = 2
    target_frame = torch.randint(
        0, 255, (batch_size, 3, 256, 512), dtype=torch.float32
    ).to(device)
    context_frames = torch.randint(
        0, 255, (batch_size, 32, 3, 256, 512), dtype=torch.float32
    ).to(device)
    actions = torch.randint(0, 3, (batch_size, 32)).to(device)

    print("\nTesting forward pass...")
    loss = model(target_frame, context_frames, actions)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting generation...")
    generated = model.generate(context_frames, actions, num_inference_steps=4)
    print(f"Generated shape: {generated.shape}")

    print("\nAll tests passed!")
