"""
Text-Conditioned Game Generation
Extension beyond paper: Generate game content from text descriptions

Example: "Generate a dark, scary DOOM level with lots of monsters"
"""

from typing import List

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class TextConditionedGameNGen(nn.Module):
    """
    Text-conditioned extension of GameNGen

    Allows conditioning game generation on text prompts:
    - "Generate a dark scary level"
    - "Create outdoor environment"
    - "Make enemies appear"
    - etc.

    Uses CLIP text encoder to convert text to embeddings
    """

    def __init__(
        self,
        base_model,  # ActionConditionedDiffusionModel
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_embedding_dim: int = 512,
        device: str = "cuda",
    ):
        super().__init__()

        self.base_model = base_model
        self.device = device

        # Load CLIP for text encoding
        print(f"Loading CLIP model: {clip_model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)

        # Freeze CLIP (we're just using it for encoding)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Project CLIP embeddings to match action embeddings
        self.text_proj = nn.Linear(text_embedding_dim, base_model.action_embedding_dim).to(device)

        print("✓ Text conditioning initialized")

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings

        Args:
            text_prompts: List of text descriptions

        Returns:
            text_embeds: (batch_size, embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(
            self.device
        )

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_embeds = outputs.pooler_output  # (batch, 512)

        # Project
        text_embeds = self.text_proj(text_embeds)  # (batch, 128)

        return text_embeds

    def generate_with_text(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        text_prompt: str,
        num_inference_steps: int = 4,
        text_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate next frame conditioned on text

        Args:
            context_frames: (batch, context_length, 3, H, W)
            context_actions: (batch, context_length)
            text_prompt: Text description
            num_inference_steps: DDIM steps
            text_guidance_scale: How much to weight text vs actions

        Returns:
            generated_frame: (batch, 3, H, W)
        """
        batch_size = context_frames.shape[0]

        # Encode text
        text_embeds = self.encode_text([text_prompt] * batch_size)  # (batch, 128)

        # Get action embeddings from base model
        action_embeds = self.base_model.action_embedding(
            context_actions
        )  # (batch, T, 128)

        # Combine text and action embeddings
        # Add text as additional context token
        text_embeds = text_embeds.unsqueeze(1)  # (batch, 1, 128)

        # Concatenate: [text_embed, action_embeds]
        combined_embeds = torch.cat(
            [text_embeds, action_embeds], dim=1
        )  # (batch, T+1, 128)

        # Add noise level embedding
        noise_level_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        noise_embeds = self.base_model.noise_aug_embedding(noise_level_ids).unsqueeze(1)

        # Full conditioning: [noise, text, actions]
        combined_embeds = torch.cat([noise_embeds, combined_embeds], dim=1)

        # Project to cross-attention dimension
        encoder_hidden_states = self.base_model.action_proj(combined_embeds)

        # Encode context frames
        context_latents = self.base_model.encode_frames(context_frames)

        # Flatten context
        context_flat = context_latents.reshape(
            batch_size,
            self.base_model.context_length * context_latents.shape[2],
            context_latents.shape[3],
            context_latents.shape[4],
        )

        # Initialize latents
        latents = torch.randn(
            batch_size,
            4,
            context_latents.shape[3],
            context_latents.shape[4],
            device=self.device,
            dtype=self.base_model.dtype,
        )

        # DDIM sampling
        self.base_model.noise_scheduler.set_timesteps(
            num_inference_steps, device=self.device
        )

        for t in self.base_model.noise_scheduler.timesteps:
            unet_input = torch.cat([latents, context_flat], dim=1)

            noise_pred = self.base_model.unet(
                unet_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample

            latents = self.base_model.noise_scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # Decode
        images = self.base_model.decode_latents(latents)

        return images


# Example usage and prompts
EXAMPLE_PROMPTS = {
    "scary": "dark scary level with shadows and monsters",
    "outdoor": "bright outdoor environment with open spaces",
    "combat": "intense combat situation with many enemies",
    "peaceful": "calm peaceful area with no threats",
    "industrial": "industrial metal corridors and machinery",
}


if __name__ == "__main__":
    print("Text-Conditioned GameNGen Example")
    print("\nThis extends GameNGen with text conditioning using CLIP.")
    print("\nExample prompts:")
    for key, prompt in EXAMPLE_PROMPTS.items():
        print(f"  - {prompt}")

    print("\nTo use:")
    print("  1. Load your trained GameNGen model")
    print("  2. Wrap it: text_model = TextConditionedGameNGen(base_model)")
    print(
        "  3. Generate: frame = text_model.generate_with_text(context, actions, 'dark scary level')"
    )

    print("\nNote: Requires training the text_proj layer on your data")
    print("      with text annotations for game states.")
