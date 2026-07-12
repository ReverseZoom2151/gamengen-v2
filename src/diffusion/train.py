"""
Training script for GameNGen diffusion model
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.diffusion.dataset import create_dataloader
from src.diffusion.model import ActionConditionedDiffusionModel
from src.config import load_config, validate_config


def compute_metrics(pred_frames: torch.Tensor, target_frames: torch.Tensor) -> dict:
    """
    Compute evaluation metrics (PSNR, MSE)

    Args:
        pred_frames: (batch_size, 3, H, W) in [0, 255]
        target_frames: (batch_size, 3, H, W) in [0, 255]

    Returns:
        dict with metrics
    """
    mse = torch.mean((pred_frames - target_frames) ** 2).item()

    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    else:
        psnr = float("inf")

    return {
        "mse": mse,
        "psnr": psnr,
    }


def train_step(
    model: ActionConditionedDiffusionModel,
    batch: dict,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool = True,
) -> dict:
    """
    Single training step

    Args:
        model: Diffusion model
        batch: Training batch
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        use_amp: Whether to use automatic mixed precision

    Returns:
        dict with loss
    """
    target_frame = batch["target_frame"].to(model.device)
    context_frames = batch["context_frames"].to(model.device)
    context_actions = batch["context_actions"].to(model.device)

    optimizer.zero_grad()

    # Forward pass with mixed precision
    if use_amp:
        with autocast():
            loss = model(target_frame, context_frames, context_actions)
    else:
        loss = model(target_frame, context_frames, context_actions)

    # Backward pass
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {"loss": loss.item()}


@torch.no_grad()
def evaluate(
    model: ActionConditionedDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
) -> dict:
    """
    Evaluate model on validation set

    Args:
        model: Diffusion model
        dataloader: Validation dataloader
        num_batches: Number of batches to evaluate

    Returns:
        dict with metrics
    """
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    num_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        target_frame = batch["target_frame"].to(model.device)
        context_frames = batch["context_frames"].to(model.device)
        context_actions = batch["context_actions"].to(model.device)

        # Compute loss
        loss = model(target_frame, context_frames, context_actions)
        total_loss += loss.item() * target_frame.shape[0]

        # Generate frames
        generated = model.generate(
            context_frames, context_actions, num_inference_steps=4
        )

        # Compute metrics
        metrics = compute_metrics(generated, target_frame)
        total_psnr += metrics["psnr"] * target_frame.shape[0]

        num_samples += target_frame.shape[0]

    model.train()

    return {
        "loss": total_loss / num_samples,
        "psnr": total_psnr / num_samples,
    }


def train(config: dict):
    """Main training loop"""

    # Setup
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir / config["experiment_name"])

    print("=" * 60)
    print("GameNGen Diffusion Model Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")

    # Create dataloaders
    print("\nCreating dataloaders...")

    train_dataloader = create_dataloader(
        data_dir=config["data_dir"],
        batch_size=config["diffusion"]["batch_size"],
        context_length=config["diffusion"]["context_length"],
        resolution=(
            config["environment"]["resolution"]["height"],
            config["environment"]["resolution"]["width"],
        ),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )

    print(f"Training batches: {len(train_dataloader)}")

    # Create model
    print("\nCreating model...")

    use_amp = config.get("mixed_precision", True) and device == "cuda"
    dtype = torch.float16 if use_amp else torch.float32

    model = ActionConditionedDiffusionModel(
        pretrained_model_name=config["diffusion"]["pretrained_model"],
        num_actions=config["environment"].get("num_actions", 3),
        action_embedding_dim=config["diffusion"]["action_embedding_dim"],
        context_length=config["diffusion"]["context_length"],
        num_noise_buckets=config["diffusion"]["noise_augmentation"]["num_buckets"],
        max_noise_level=config["diffusion"]["noise_augmentation"]["max_noise_level"],
        cfg_drop_prob=config["diffusion"].get("cfg_drop_prob", 0.1),
        device=device,
        dtype=dtype,
    )

    # Optimizer
    print("\nSetting up optimizer...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["diffusion"]["learning_rate"],
        weight_decay=config["diffusion"]["weight_decay"],
        betas=(config["diffusion"]["adam_beta1"], config["diffusion"]["adam_beta2"]),
        eps=config["diffusion"]["adam_epsilon"],
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=use_amp)

    # Resume from checkpoint if exists
    start_step = 0
    checkpoint_path = output_dir / "latest_checkpoint.pt"

    if checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.unet.load_state_dict(checkpoint["unet"])
        model.action_embedding.load_state_dict(checkpoint["action_embedding"])
        model.noise_aug_embedding.load_state_dict(checkpoint["noise_aug_embedding"])
        model.action_proj.load_state_dict(checkpoint["action_proj"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]

        print(f"Resumed from step {start_step}")

    # Training parameters
    num_train_steps = config["diffusion"]["num_train_steps"]
    save_every = config["diffusion"]["save_every_n_steps"]
    eval_every = config["diffusion"]["eval_every_n_steps"]
    log_every = config["logging"]["log_interval"]

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training from step {start_step}")
    print(f"Total steps: {num_train_steps}")
    print(f"Batch size: {config['diffusion']['batch_size']}")
    print(f"Mixed precision: {use_amp}")
    print("=" * 60 + "\n")

    model.train()
    global_step = start_step
    running_loss = 0.0
    start_time = time.time()

    dataloader_iter = iter(train_dataloader)

    pbar = tqdm(initial=start_step, total=num_train_steps, desc="Training")

    while global_step < num_train_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Reset dataloader
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)

        # Training step
        step_output = train_step(model, batch, optimizer, scaler, use_amp=use_amp)

        loss = step_output["loss"]
        running_loss += loss

        global_step += 1
        pbar.update(1)

        # Logging
        if global_step % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            steps_per_sec = log_every / elapsed

            pbar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "steps/s": f"{steps_per_sec:.2f}"}
            )

            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)

            running_loss = 0.0
            start_time = time.time()

        # Evaluation
        if global_step % eval_every == 0:
            print("\n" + "=" * 60)
            print(f"Evaluating at step {global_step}")

            eval_metrics = evaluate(model, train_dataloader, num_batches=10)

            print(f"Evaluation metrics:")
            print(f"  Loss: {eval_metrics['loss']:.4f}")
            print(f"  PSNR: {eval_metrics['psnr']:.2f}")
            print("=" * 60 + "\n")

            writer.add_scalar("eval/loss", eval_metrics["loss"], global_step)
            writer.add_scalar("eval/psnr", eval_metrics["psnr"], global_step)

        # Save checkpoint
        if global_step % save_every == 0 or global_step == num_train_steps:
            checkpoint_file = output_dir / f"checkpoint_step_{global_step}.pt"

            torch.save(
                {
                    "step": global_step,
                    "unet": model.unet.state_dict(),
                    "action_embedding": model.action_embedding.state_dict(),
                    "noise_aug_embedding": model.noise_aug_embedding.state_dict(),
                    "action_proj": model.action_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                checkpoint_file,
            )

            # Also save as latest
            latest_file = output_dir / "latest_checkpoint.pt"
            torch.save(
                {
                    "step": global_step,
                    "unet": model.unet.state_dict(),
                    "action_embedding": model.action_embedding.state_dict(),
                    "noise_aug_embedding": model.noise_aug_embedding.state_dict(),
                    "action_proj": model.action_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                latest_file,
            )

            print(f"\n✓ Saved checkpoint: {checkpoint_file}\n")

            # Keep only last N checkpoints
            keep_last_n = config["diffusion"]["keep_last_n_checkpoints"]
            if keep_last_n > 0:
                checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
                if len(checkpoints) > keep_last_n:
                    for ckpt in checkpoints[:-keep_last_n]:
                        ckpt.unlink()

    pbar.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final checkpoint saved to: {output_dir}")
    print("=" * 60)

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train GameNGen diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Override data directory"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.data:
        config["data_dir"] = args.data

    if args.steps:
        config["diffusion"]["num_train_steps"] = args.steps

    if args.batch_size:
        config["diffusion"]["batch_size"] = args.batch_size

    if args.device:
        config["device"] = args.device

    validate_config(config)

    # Train
    train(config)


if __name__ == "__main__":
    main()
