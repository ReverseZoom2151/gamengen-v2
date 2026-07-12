"""
Fréchet Video Distance (FVD) Implementation
Proper implementation using I3D (Inflated 3D ConvNet) for feature extraction
Based on paper Section 5.1: "We measure the FVD"
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm


class I3D(nn.Module):
    """
    Inflated 3D ConvNet for video feature extraction

    Simplified implementation for FVD computation.
    Full implementation would use: https://github.com/piergiaj/pytorch-i3d

    For production use: pip install pytorch-i3d
    """

    def __init__(self, num_classes=400, in_channels=3):
        super().__init__()

        # Simplified I3D architecture
        # In practice, load pretrained I3D weights
        self.conv3d_1a = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool3d_2a = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.conv3d_2b = nn.Conv3d(64, 64, kernel_size=1)
        self.conv3d_2c = nn.Conv3d(64, 192, kernel_size=3, padding=1)
        self.maxpool3d_3a = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # More layers would be here in full implementation

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, frames, height, width)
        Returns:
            features: (batch, feature_dim)
        """
        x = self.conv3d_1a(x)
        x = torch.relu(x)
        x = self.maxpool3d_2a(x)

        x = self.conv3d_2b(x)
        x = torch.relu(x)
        x = self.conv3d_2c(x)
        x = torch.relu(x)
        x = self.maxpool3d_3a(x)

        # Extract features before final classification
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)

        return features


class FVDCalculator:
    """
    Fréchet Video Distance Calculator

    Paper Section 5.1:
    "We measure the FVD (Unterthiner et al., 2019) computed over a random
    holdout of 512 trajectories, measuring the distance between the predicted
    and ground truth trajectory distributions"

    FVD measures distribution similarity between real and generated videos
    using features from I3D network pretrained on Kinetics.
    """

    def __init__(self, device: str = "cuda", weights_path: str | None = None):
        """
        Args:
            device: Device for computation
            use_pretrained: Whether to use pretrained I3D (requires pytorch-i3d package)
        """
        self.device = device

        try:
            # Try to use proper I3D implementation if available
            from pytorch_i3d import InceptionI3d  # type: ignore[import]

            self.i3d = InceptionI3d(400, in_channels=3).to(device)

            if not weights_path:
                raise ValueError("FVD requires an explicit pretrained I3D weights_path")
            self.i3d.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

            self.using_full_i3d = True

        except ImportError as error:
            raise RuntimeError("paper-comparable FVD requires pytorch-i3d and pretrained weights") from error

        self.i3d.eval()

    def preprocess_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Preprocess videos for I3D

        Args:
            videos: (batch, frames, height, width, channels) in [0, 255]
        Returns:
            preprocessed: (batch, channels, frames, height, width) in [-1, 1]
        """
        # Permute to (batch, channels, frames, height, width)
        if videos.shape[-1] == 3:  # (B, T, H, W, C)
            videos = videos.permute(0, 4, 1, 2, 3)

        # Normalize to [-1, 1]
        videos = videos.float() / 127.5 - 1.0

        # Resize to 224x224 if needed (I3D expects this)
        if videos.shape[-2:] != (224, 224):
            batch, channels, frames, h, w = videos.shape
            videos = videos.view(-1, channels, h, w)  # Flatten batch and frames
            videos = torch.nn.functional.interpolate(
                videos, size=(224, 224), mode="bilinear", align_corners=False
            )
            videos = videos.view(batch, channels, frames, 224, 224)

        return videos

    def extract_features(self, videos: torch.Tensor) -> np.ndarray:
        """
        Extract I3D features from videos

        Args:
            videos: (batch, frames, height, width, channels) in [0, 255]
        Returns:
            features: (batch, feature_dim) numpy array
        """
        videos = self.preprocess_videos(videos).to(self.device)

        with torch.no_grad():
            features = self.i3d(videos)

        return features.cpu().numpy()

    def compute_fvd(
        self, real_videos: torch.Tensor, fake_videos: torch.Tensor, batch_size: int = 16
    ) -> float:
        """
        Compute Fréchet Video Distance

        Args:
            real_videos: (N, T, H, W, C) real videos
            fake_videos: (N, T, H, W, C) generated videos
            batch_size: Batch size for feature extraction

        Returns:
            fvd: Fréchet Video Distance score
        """
        if len(real_videos) != len(fake_videos) or len(real_videos) < 2:
            raise ValueError("FVD requires equally sized real/fake sets with at least two videos")
        # Extract features in batches
        real_features = []
        fake_features = []

        for i in range(0, len(real_videos), batch_size):
            real_batch = real_videos[i : i + batch_size]
            fake_batch = fake_videos[i : i + batch_size]

            real_features.append(self.extract_features(real_batch))
            fake_features.append(self.extract_features(fake_batch))

        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)

        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Compute Fréchet distance
        diff = mu_real - mu_fake

        # Product might be complex with imaginary parts
        covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fvd = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)

        return float(fvd)


def evaluate_fvd_on_trajectories(
    model,
    dataloader,
    num_trajectories: int = 512,
    trajectory_length: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Evaluate FVD on gameplay trajectories

    Paper Section 5.1:
    "For 16 frames our model obtains an FVD of 114.02.
     For 32 frames our model obtains an FVD of 186.23."

    Args:
        model: Trained diffusion model
        dataloader: Data loader with real trajectories
        num_trajectories: Number of trajectories to evaluate
        trajectory_length: Length of each trajectory (16 or 32 frames)
        device: Device for computation

    Returns:
        dict with FVD scores
    """
    raise NotImplementedError(
        "This helper cannot produce valid FVD from one-window dataset batches. "
        "Supply separately recorded real and autoregressively generated trajectories "
        "to FVDCalculator.compute_fvd instead."
    )


if __name__ == "__main__":
    # Test FVD calculator
    print("Testing FVD Calculator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        fvd_calc = FVDCalculator(device=device)

        # Create dummy videos
        real_videos = torch.randint(0, 255, (10, 16, 224, 224, 3), dtype=torch.float32)
        fake_videos = torch.randint(0, 255, (10, 16, 224, 224, 3), dtype=torch.float32)

        # Compute FVD
        fvd = fvd_calc.compute_fvd(real_videos, fake_videos)

        print(f"FVD: {fvd:.2f}")
        print("Note: This is on random data, so FVD will be high")
        print("\nFor accurate FVD, install: pip install pytorch-i3d")

    except Exception as e:
        print(f"FVD test failed: {e}")
        import traceback

        traceback.print_exc()
