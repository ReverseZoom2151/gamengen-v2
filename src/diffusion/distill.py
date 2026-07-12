"""
Model Distillation for 1-Step Inference
Paper Appendix A.6: Distill 4-step model to 1-step for 50 FPS

"We experimented with distilling our model, following (Yin et al., 2024; Wang et al., 2023)
in the single-step setting. Distillation does help substantially..."
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))



class DistillationTrainer:
    """
    Distill multi-step diffusion model to single-step model

    Paper Appendix A.6:
    "During distillation training we use 3 U-Nets, all initialized with a
    GameNGen model: generator, teacher, and fake-score model."
    """

    def __init__(
        self,
        teacher_checkpoint: str,
        config: dict,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        print("=" * 60)
        print("Model Distillation for 1-Step Inference")
        print("=" * 60)

        # Create three models
        print("\nCreating models...")
        print("  1. Teacher (frozen, 4-step model)")
        print("  2. Generator (student, 1-step model)")
        print("  3. Fake-score model (predicts generator output)")

        # Teacher model (frozen)
        self.teacher = self._create_model()
        self._load_checkpoint(self.teacher, teacher_checkpoint)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        print("  ✓ Teacher loaded and frozen")

        # Generator model (student)
        self.generator = self._create_model()
        self._load_checkpoint(self.generator, teacher_checkpoint)
        self.generator.train()

        print("  ✓ Generator initialized from teacher")

        # Fake-score model
        self.fake_score_model = self._create_model()
        self._load_checkpoint(self.fake_score_model, teacher_checkpoint)
        self.fake_score_model.train()

        print("  ✓ Fake-score model initialized")

        # Optimizers
        distill_config = config["distillation"]

        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=distill_config["learning_rate"]
        )

        self.fake_score_optimizer = optim.Adam(
            self.fake_score_model.parameters(), lr=distill_config["learning_rate"]
        )

        print("  ✓ Optimizers created")

    def _create_model(self):
        """Create a model instance"""
        from src.diffusion.model import ActionConditionedDiffusionModel

        return ActionConditionedDiffusionModel(
            pretrained_model_name=self.config["diffusion"]["pretrained_model"],
            num_actions=self.config["environment"].get("num_actions", 3),
            action_embedding_dim=self.config["diffusion"]["action_embedding_dim"],
            context_length=self.config["diffusion"]["context_length"],
            num_noise_buckets=self.config["diffusion"]["noise_augmentation"][
                "num_buckets"
            ],
            max_noise_level=self.config["diffusion"]["noise_augmentation"][
                "max_noise_level"
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _load_checkpoint(self, model, checkpoint_path):
        """Load checkpoint into model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.unet.load_state_dict(checkpoint["unet"])
        model.action_embedding.load_state_dict(checkpoint["action_embedding"])
        model.noise_aug_embedding.load_state_dict(checkpoint["noise_aug_embedding"])
        model.action_proj.load_state_dict(checkpoint["action_proj"])

    @torch.no_grad()
    def get_teacher_prediction(
        self,
        target_latents: torch.Tensor,
        context_frames: torch.Tensor,
        actions: torch.Tensor,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Get teacher's 4-step prediction
        This is the target for distillation
        """
        # Encode context
        context_latents = self.teacher.encode_frames(context_frames)

        # Add small noise to target (varying amounts)
        noise_level = torch.rand(target_latents.shape[0], device=self.device) * 0.3
        noise = torch.randn_like(target_latents)
        noised_latents = target_latents + noise_level.view(-1, 1, 1, 1) * noise

        # Generate with teacher (4 steps)
        # Use custom generation with specific initial noise
        # For distillation, we want to predict the clean latent from noised version
        self.teacher.noise_scheduler.set_timesteps(4, device=self.device)

        # Simplified: just return target with small noise
        # In practice, would run full teacher generation
        return target_latents

    def get_generator_prediction(
        self,
        noised_latents: torch.Tensor,
        context_frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get generator's 1-step prediction
        """
        # Encode context
        context_latents = self.generator.encode_frames(context_frames)

        # Flatten context
        batch_size = context_latents.shape[0]
        context_flat = context_latents.reshape(
            batch_size,
            self.config["diffusion"]["context_length"] * context_latents.shape[2],
            context_latents.shape[3],
            context_latents.shape[4],
        )

        # Embed actions
        action_embeds = self.generator.action_embedding(actions)
        noise_level_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        noise_embeds = self.generator.noise_aug_embedding(noise_level_ids).unsqueeze(1)
        action_embeds = torch.cat([noise_embeds, action_embeds], dim=1)
        encoder_hidden_states = self.generator.action_proj(action_embeds)

        # Concatenate with context
        unet_input = torch.cat([noised_latents, context_flat], dim=1)

        # Single-step prediction (timestep = 0, direct prediction)
        timestep = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        # U-Net prediction
        predicted_latents = self.generator.unet(
            unet_input, timestep, encoder_hidden_states=encoder_hidden_states
        ).sample

        return predicted_latents

    def distillation_step(self, batch: dict) -> dict:
        """
        Single distillation training step

        Paper Appendix A.6:
        "To train the generator, we use the teacher and the fake-score model to
        predict the noise added to an input image - ε_real and ε_fake.
        We optimize the weights of the generator to minimize the generator
        gradient value at each pixel weighted by ε_real − ε_fake."
        """
        target_frame = batch["target_frame"].to(self.device)
        context_frames = batch["context_frames"].to(self.device)
        context_actions = batch["context_actions"].to(self.device)

        # Encode target to latents
        target_latents = self.teacher.encode_frames(target_frame)

        # Add noise
        noise = torch.randn_like(target_latents)
        noise_level = torch.rand(target_latents.shape[0], device=self.device) * 0.5
        noised_latents = target_latents + noise_level.view(-1, 1, 1, 1) * noise

        # 1. Train fake-score model
        self.fake_score_optimizer.zero_grad()

        # Fake-score predicts generator output
        with torch.no_grad():
            generator_output = self.get_generator_prediction(
                noised_latents, context_frames, context_actions
            )

        fake_score_pred = self.get_generator_prediction(
            noised_latents, context_frames, context_actions
        )

        # Standard diffusion loss for fake-score model
        fake_score_loss = nn.functional.mse_loss(fake_score_pred, target_latents)

        fake_score_loss.backward()
        self.fake_score_optimizer.step()

        # 2. Train generator
        self.generator_optimizer.zero_grad()

        # Get teacher prediction (real epsilon)
        with torch.no_grad():
            teacher_pred = self.get_teacher_prediction(
                target_latents, context_frames, context_actions
            )

        # Get fake-score prediction
        with torch.no_grad():
            fake_score_pred = self.get_generator_prediction(
                noised_latents, context_frames, context_actions
            )

        # Generator prediction
        generator_pred = self.get_generator_prediction(
            noised_latents, context_frames, context_actions
        )

        # Distillation loss: weighted by difference
        # ε_real - ε_fake gives us the gradient direction
        weight = (teacher_pred - fake_score_pred).abs()
        generator_loss = (weight * (generator_pred - teacher_pred) ** 2).mean()

        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.generator_optimizer.step()

        return {
            "generator_loss": generator_loss.item(),
            "fake_score_loss": fake_score_loss.item(),
        }

    def save_generator(self, save_path: str):
        """Save distilled generator"""
        torch.save(
            {
                "unet": self.generator.unet.state_dict(),
                "action_embedding": self.generator.action_embedding.state_dict(),
                "noise_aug_embedding": self.generator.noise_aug_embedding.state_dict(),
                "action_proj": self.generator.action_proj.state_dict(),
                "config": self.config,
                "distilled": True,
                "num_inference_steps": 1,
            },
            save_path,
        )

        print(f"Distilled model saved to {save_path}")


def distill_model(config: dict):
    """Main distillation training loop.

    The historical implementation returned clean target latents as a fake
    teacher target and invoked the generator in place of the fake-score model.
    That cannot produce a valid Appendix A.6 distilled model, so this command
    is deliberately quarantined until the three-network objective and rollout
    fixtures are implemented and validated.
    """
    raise NotImplementedError(
        "one-step distillation is not validated; do not use this command for "
        "training or performance claims"
    )

    device = config.get("device", "cuda")
    distill_config = config["distillation"]

    # Create trainer
    trainer = DistillationTrainer(
        teacher_checkpoint=distill_config["teacher_checkpoint"],
        config=config,
        device=device,
    )

    # Create dataloader
    print("\nCreating dataloader...")

    dataloader = create_dataloader(
        data_dir=config["data_dir"],
        batch_size=32,
        context_length=config["diffusion"]["context_length"],
        resolution=(
            config["environment"]["resolution"]["height"],
            config["environment"]["resolution"]["width"],
        ),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )

    print(f"Dataloader created: {len(dataloader)} batches")

    # Training loop
    num_steps = distill_config["num_steps"]
    save_freq = 1000

    output_dir = Path(config["checkpoint_dir"]) / "distilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["log_dir"]) / "distillation"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)

    print("\n" + "=" * 60)
    print(f"Starting distillation training")
    print(f"Steps: {num_steps}")
    print("=" * 60 + "\n")

    dataloader_iter = iter(dataloader)
    pbar = tqdm(total=num_steps, desc="Distillation")

    for step in range(num_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Training step
        losses = trainer.distillation_step(batch)

        pbar.update(1)
        pbar.set_postfix(
            {
                "gen_loss": f"{losses['generator_loss']:.4f}",
                "fake_loss": f"{losses['fake_score_loss']:.4f}",
            }
        )

        # Logging
        if step % 10 == 0:
            writer.add_scalar("distill/generator_loss", losses["generator_loss"], step)
            writer.add_scalar(
                "distill/fake_score_loss", losses["fake_score_loss"], step
            )

        # Save checkpoint
        if (step + 1) % save_freq == 0 or step == num_steps - 1:
            checkpoint_path = output_dir / f"distilled_step_{step + 1}.pt"
            trainer.save_generator(checkpoint_path)
            print(f"\n✓ Saved checkpoint: {checkpoint_path}\n")

    pbar.close()
    writer.close()

    # Save final model
    final_path = output_dir / "distilled_final.pt"
    trainer.save_generator(final_path)

    print("\n" + "=" * 60)
    print("Distillation complete!")
    print(f"Final model: {final_path}")
    print("This model can run at 1-step (50 FPS vs 20 FPS)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Distill GameNGen to 1-step model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier3_full_doom.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--teacher", type=str, required=True, help="Path to trained teacher checkpoint"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override number of distillation steps"
    )

    args = parser.parse_args()

    # Load config
    from src.config import load_config

    config = load_config(args.config)

    # Set teacher checkpoint
    if "distillation" not in config:
        config["distillation"] = {}

    config["distillation"]["teacher_checkpoint"] = args.teacher

    if args.steps:
        config["distillation"]["num_steps"] = args.steps

    # Set default values if not in config
    if "num_steps" not in config["distillation"]:
        config["distillation"]["num_steps"] = 50000

    if "learning_rate" not in config["distillation"]:
        config["distillation"]["learning_rate"] = 1e-5

    # Distill
    distill_model(config)


if __name__ == "__main__":
    main()
