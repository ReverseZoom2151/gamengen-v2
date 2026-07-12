"""
Real-time inference for GameNGen
Auto-regressive generation of gameplay
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import imageio
import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.diffusion.model import ActionConditionedDiffusionModel
from src.environment.chrome_dino_env import SimpleDinoEnv
from src.config import load_config


class GameNGenPlayer:
    """Real-time game player using GameNGen"""

    def __init__(
        self,
        model: ActionConditionedDiffusionModel,
        context_length: int = 32,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.5,
        target_fps: int = 20,
    ):
        self.model = model
        self.context_length = context_length
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.target_fps = target_fps

        # Frame buffer for context
        self.frame_buffer = deque(maxlen=context_length)
        self.action_buffer = deque(maxlen=context_length)

        # Timing
        self.frame_times = []
        self.last_frame_time = None

    def reset(self, initial_frames: List[np.ndarray], initial_actions: List[int]):
        """
        Reset player with initial context

        Args:
            initial_frames: List of initial frames (context_length frames)
            initial_actions: List of initial actions (context_length actions)
        """
        assert len(initial_frames) == self.context_length
        assert len(initial_actions) == self.context_length

        self.frame_buffer.clear()
        self.action_buffer.clear()

        for frame, action in zip(initial_frames, initial_actions):
            self.frame_buffer.append(frame)
            self.action_buffer.append(action)

        self.frame_times = []
        self.last_frame_time = time.time()

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to tensor"""
        # Ensure correct shape and type
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        # Convert to tensor (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)

        return frame_tensor

    def _postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable frame"""
        # (C, H, W) -> (H, W, C)
        frame = frame_tensor.permute(1, 2, 0).cpu().numpy()

        # Clip to [0, 255] and convert to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def generate_next_frame(self, action: int) -> np.ndarray:
        """
        Generate next frame given action

        Args:
            action: Action to take

        Returns:
            Generated frame as numpy array (H, W, 3) uint8
        """
        start_time = time.time()

        # Prepare context
        context_frames = [self._preprocess_frame(f) for f in self.frame_buffer]
        context_frames = torch.stack(context_frames).unsqueeze(0)  # (1, T, C, H, W)

        # The newest action belongs to the final context frame and produces the
        # frame being sampled.  Replacing the final action before sampling keeps
        # controls responsive without changing the model's fixed token length.
        context_action_values = list(self.action_buffer)
        context_action_values[-1] = action
        context_actions = torch.tensor(context_action_values, dtype=torch.long).unsqueeze(0)

        # Move to device
        context_frames = context_frames.to(self.model.device)
        context_actions = context_actions.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                context_frames,
                context_actions,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )  # (1, 3, H, W)

        # Convert to numpy
        generated_frame = self._postprocess_frame(generated[0])

        # Update buffers
        self.frame_buffer.append(generated_frame.astype(np.float32))
        self.action_buffer.append(action)

        # Track timing
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)

        return generated_frame

    def get_fps(self) -> float:
        """Get average FPS over last N frames"""
        if len(self.frame_times) == 0:
            return 0.0

        avg_time = np.mean(self.frame_times[-30:])  # Last 30 frames
        return 1.0 / avg_time if avg_time > 0 else 0.0


def play_interactive(
    model: ActionConditionedDiffusionModel,
    config: dict,
    save_video: Optional[str] = None,
):
    """
    Interactive gameplay mode

    Args:
        model: Trained diffusion model
        config: Configuration dict
        save_video: Optional path to save video
    """
    print("\n" + "=" * 60)
    print("GameNGen - Interactive Gameplay")
    print("=" * 60)

    # Create environment for initial frames
    print("Initializing environment...")
    env = SimpleDinoEnv(
        width=config["environment"]["resolution"]["width"],
        height=config["environment"]["resolution"]["height"],
    )

    # Create player
    player = GameNGenPlayer(
        model=model,
        context_length=config["diffusion"]["context_length"],
        num_inference_steps=config["inference"]["num_sampling_steps"],
        guidance_scale=config["inference"]["cfg_scale"],
        target_fps=config["inference"]["fps"],
    )

    # Collect initial context from real environment
    print("Collecting initial context...")
    obs, info = env.reset()

    initial_frames = []
    initial_actions = []

    for i in range(config["diffusion"]["context_length"]):
        action = env.action_space.sample()  # Random initial actions
        next_obs, reward, done, truncated, info = env.step(action)

        initial_frames.append(obs.astype(np.float32))
        initial_actions.append(action)

        obs = next_obs

        if done:
            obs, info = env.reset()

    # Reset player with initial context
    player.reset(initial_frames, initial_actions)

    print("\nStarting gameplay!")
    print("Controls:")
    print("  SPACE: Jump")
    print("  DOWN: Duck")
    print("  Q: Quit")
    print("=" * 60 + "\n")

    # Setup video recording
    frames_recorded = []

    # Main loop
    frame_count = 0
    action = 0  # No action initially

    try:
        while True:
            # Generate next frame
            frame = player.generate_next_frame(action)

            # Record frame
            if save_video:
                frames_recorded.append(frame)

            # Display
            # Convert to BGR for OpenCV
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add FPS counter
            fps = player.get_fps()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Add frame counter
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("GameNGen", display_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                action = 1  # Jump
            elif key == 82:  # Up arrow
                action = 1  # Jump
            elif key == 84:  # Down arrow
                action = 2  # Duck
            else:
                action = 0  # No action

            frame_count += 1

            # Progress update
            if frame_count % 100 == 0:
                print(f"Frame {frame_count} | FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        cv2.destroyAllWindows()
        env.close()

    print(f"\nTotal frames generated: {frame_count}")

    # Save video
    if save_video and len(frames_recorded) > 0:
        print(f"\nSaving video to {save_video}...")
        imageio.mimsave(save_video, frames_recorded, fps=config["inference"]["fps"])
        print(f"✓ Video saved ({len(frames_recorded)} frames)")

    print("\n" + "=" * 60)
    print("Gameplay session ended")
    print("=" * 60)


def evaluate_on_dataset(
    model: ActionConditionedDiffusionModel,
    config: dict,
    num_trajectories: int = 10,
):
    """
    Evaluate model on recorded trajectories

    Args:
        model: Trained diffusion model
        config: Configuration dict
        num_trajectories: Number of trajectories to evaluate
    """
    from src.diffusion.dataset import create_dataloader

    print("\n" + "=" * 60)
    print("GameNGen - Evaluation Mode")
    print("=" * 60)

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=config["data_dir"],
        batch_size=1,
        context_length=config["diffusion"]["context_length"],
        resolution=(
            config["environment"]["resolution"]["height"],
            config["environment"]["resolution"]["width"],
        ),
        num_workers=0,
        shuffle=True,
        max_trajectories=num_trajectories,
    )

    print(f"Evaluating on {len(dataloader)} trajectories")

    # Evaluate
    total_psnr = 0.0
    total_time = 0.0

    for i, batch in enumerate(dataloader):
        if i >= num_trajectories:
            break

        context_frames = batch["context_frames"].to(model.device)
        context_actions = batch["context_actions"].to(model.device)
        target_frame = batch["target_frame"].to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                context_frames,
                context_actions,
                num_inference_steps=config["inference"]["num_sampling_steps"],
                guidance_scale=config["inference"]["cfg_scale"],
            )
        elapsed = time.time() - start_time

        # Compute PSNR
        mse = torch.mean((generated - target_frame) ** 2).item()
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse > 0 else float("inf")

        total_psnr += psnr
        total_time += elapsed

        print(
            f"Trajectory {i+1}/{num_trajectories} | PSNR: {psnr:.2f} | Time: {elapsed*1000:.1f}ms"
        )

    avg_psnr = total_psnr / num_trajectories
    avg_time = total_time / num_trajectories
    avg_fps = 1.0 / avg_time

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average time per frame: {avg_time*1000:.1f} ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GameNGen inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "evaluate"],
        help="Inference mode",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Path to save video (interactive mode)",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10,
        help="Number of trajectories (evaluate mode)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Create model
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = ActionConditionedDiffusionModel(
        pretrained_model_name=config["diffusion"]["pretrained_model"],
        num_actions=config["environment"].get("num_actions", 3),
        action_embedding_dim=config["diffusion"]["action_embedding_dim"],
        context_length=config["diffusion"]["context_length"],
        num_noise_buckets=config["diffusion"]["noise_augmentation"]["num_buckets"],
        max_noise_level=config["diffusion"]["noise_augmentation"]["max_noise_level"],
        decoder_checkpoint=config.get("inference", {}).get("decoder_checkpoint"),
        device=device,
        dtype=torch.float32,  # Use float32 for inference
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = checkpoint.get("model", checkpoint)

    model.unet.load_state_dict(model_state["unet"])
    model.action_embedding.load_state_dict(model_state["action_embedding"], strict=False)
    model.noise_aug_embedding.load_state_dict(model_state["noise_aug_embedding"])
    model.action_proj.load_state_dict(model_state["action_proj"])

    model.eval()

    print("✓ Model loaded successfully")

    # Run inference
    if args.mode == "interactive":
        play_interactive(model, config, args.save_video)
    elif args.mode == "evaluate":
        evaluate_on_dataset(model, config, args.num_trajectories)


if __name__ == "__main__":
    main()
