"""Training entry point for the action-conditioned diffusion model."""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config, validate_config
from src.diffusion.dataset import create_dataloaders
from src.diffusion.optimizers import create_optimizer
from src.utils.training import (
    atomic_torch_save,
    build_run_manifest,
    capture_rng_state,
    checkpoint_step,
    restore_rng_state,
    seed_everything,
)


def compute_metrics(pred_frames: torch.Tensor, target_frames: torch.Tensor) -> dict:
    mse = torch.mean((pred_frames - target_frames) ** 2).item()
    return {"mse": mse, "psnr": float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)}


def make_scheduler(optimizer: torch.optim.Optimizer, name: str, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Create a scheduler whose step count is optimizer steps, not microbatches."""
    name = name.lower()
    if name not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unsupported lr_scheduler: {name}")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if name == "constant":
            return 1.0
        if name == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def _move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["target_frame"].to(device, non_blocking=True),
        batch["context_frames"].to(device, non_blocking=True),
        batch["context_actions"].to(device, non_blocking=True),
    )


def train_microbatch(model, batch: dict, scaler: GradScaler, accumulation_steps: int, use_amp: bool) -> torch.Tensor:
    """Backpropagate a normalized microbatch loss; the caller owns optimizer stepping."""
    target, context_frames, context_actions = _move_batch(batch, model.device)
    with autocast(enabled=use_amp):
        loss = model(target, context_frames, context_actions)
    scaler.scale(loss / accumulation_steps).backward()
    return loss.detach()


@torch.no_grad()
def evaluate(model, dataloader, num_batches: int = 10) -> dict:
    model.eval()
    total_loss = total_psnr = 0.0
    num_samples = 0
    for index, batch in enumerate(dataloader):
        if index >= num_batches:
            break
        target, context_frames, context_actions = _move_batch(batch, model.device)
        loss = model(target, context_frames, context_actions)
        generated = model.generate(context_frames, context_actions, num_inference_steps=4)
        metrics = compute_metrics(generated, target)
        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += metrics["psnr"] * batch_size
        num_samples += batch_size
    model.train()
    if num_samples == 0:
        raise ValueError("validation dataloader yielded no samples")
    return {"loss": total_loss / num_samples, "psnr": total_psnr / num_samples}


def _checkpoint_payload(model, optimizer, scheduler, scaler, step: int, config: dict) -> dict:
    return {
        "format_version": 2,
        "step": step,
        "model": {
            "unet": model.unet.state_dict(),
            "action_embedding": model.action_embedding.state_dict(),
            "noise_aug_embedding": model.noise_aug_embedding.state_dict(),
            "action_proj": model.action_proj.state_dict(),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "run_manifest": build_run_manifest(config, config.get("data_dir")),
        "config": config,
    }


def _restore_checkpoint(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model", checkpoint)  # v1 compatibility
    for name in ("unet", "action_embedding", "noise_aug_embedding", "action_proj"):
        getattr(model, name).load_state_dict(model_state[name])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def _next_batch(iterator: Iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def train(config: dict) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
        from src.diffusion.model import ActionConditionedDiffusionModel
    except ImportError as error:
        raise RuntimeError(
            "diffusion training requires TensorBoard; install the diffusion extra"
        ) from error
    seed_everything(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir, log_dir = Path(config["checkpoint_dir"]), Path(config["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    diffusion = config["diffusion"]
    writer = SummaryWriter(log_dir / config["experiment_name"])
    try:
        train_loader, validation_loader = create_dataloaders(
            data_dir=config["data_dir"], batch_size=diffusion["batch_size"], context_length=diffusion["context_length"],
            resolution=(config["environment"]["resolution"]["height"], config["environment"]["resolution"]["width"]),
            num_workers=config.get("num_workers", 4), validation_fraction=diffusion.get("validation_fraction", 0.05), seed=int(config.get("seed", 0)),
            split_manifest_path=str(output_dir / "validation_split.json"),
        )
        use_amp = bool(config.get("mixed_precision", True) and device.type == "cuda")
        model = ActionConditionedDiffusionModel(
            pretrained_model_name=diffusion["pretrained_model"], num_actions=config["environment"]["num_actions"],
            action_embedding_dim=diffusion["action_embedding_dim"], context_length=diffusion["context_length"],
            num_noise_buckets=diffusion["noise_augmentation"]["num_buckets"], max_noise_level=diffusion["noise_augmentation"]["max_noise_level"],
            cfg_drop_prob=diffusion.get("cfg_drop_prob", 0.1), device=str(device), dtype=torch.float16 if use_amp else torch.float32,
        )
        optimizer = create_optimizer(diffusion.get("optimizer", "AdamW"), model.parameters(), diffusion)
        scheduler = make_scheduler(optimizer, diffusion.get("lr_scheduler", "constant"), diffusion.get("warmup_steps", 0), diffusion["num_train_steps"])
        scaler = GradScaler(enabled=use_amp)
        latest = output_dir / "latest_checkpoint.pt"
        global_step = _restore_checkpoint(latest, model, optimizer, scheduler, scaler, device) if latest.exists() else 0
        accumulation = max(1, int(diffusion.get("gradient_accumulation_steps", 1)))
        gradient_clip = float(diffusion.get("gradient_clip", 1.0))
        save_every, eval_every = diffusion["save_every_n_steps"], diffusion["eval_every_n_steps"]
        log_every = config["logging"]["log_interval"]
        iterator, running_loss, start_time = iter(train_loader), 0.0, time.time()
        progress = tqdm(initial=global_step, total=diffusion["num_train_steps"], desc="Training")
        while global_step < diffusion["num_train_steps"]:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accumulation):
                batch, iterator = _next_batch(iterator, train_loader)
                step_loss += train_microbatch(model, batch, scaler, accumulation, use_amp).item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running_loss += step_loss / accumulation
            progress.update(1)
            if global_step % log_every == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                average = running_loss / log_every
                progress.set_postfix(loss=f"{average:.4f}", steps_per_sec=f"{log_every / elapsed:.2f}")
                writer.add_scalar("train/loss", average, global_step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                running_loss, start_time = 0.0, time.time()
            if global_step % eval_every == 0:
                metrics = evaluate(model, validation_loader)
                writer.add_scalar("validation/loss", metrics["loss"], global_step)
                writer.add_scalar("validation/psnr", metrics["psnr"], global_step)
            if global_step % save_every == 0 or global_step == diffusion["num_train_steps"]:
                payload = _checkpoint_payload(model, optimizer, scheduler, scaler, global_step, config)
                checkpoint_file = output_dir / f"checkpoint_step_{global_step}.pt"
                atomic_torch_save(payload, checkpoint_file)
                atomic_torch_save(payload, latest)
                keep = int(diffusion.get("keep_last_n_checkpoints", 0))
                checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"), key=checkpoint_step)
                for stale in checkpoints[:-keep] if keep > 0 else []:
                    stale.unlink()
        progress.close()
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GameNGen diffusion model")
    parser.add_argument("--config", default="configs/tier1_chrome_dino.yaml")
    parser.add_argument("--data")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data: config["data_dir"] = args.data
    if args.steps: config["diffusion"]["num_train_steps"] = args.steps
    if args.batch_size: config["diffusion"]["batch_size"] = args.batch_size
    if args.device: config["device"] = args.device
    train(validate_config(config))


if __name__ == "__main__":
    main()
