"""
Image-Based Game Modding
Extension beyond paper: Edit game content by providing example images

Based on paper's Appendix A.4: "Out-of-Distribution Sampling"
"We replicate the same frame for the entirety of the history buffer"
"The model often consistently integrates the added characters into the new location"
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from src.diffusion.conditioning import condition_current_action


class ImageBasedModding:
    """
    Image-based game modification system

    Allows:
    - Inserting characters into new locations
    - Changing level layouts
    - Adding new objects
    - Style transfer to game

    Paper Appendix A.4 shows this works with simple image editing.
    """

    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: Trained ActionConditionedDiffusionModel
            device: Device for computation
        """
        self.model = model
        self.device = device

    def load_and_preprocess_image(
        self, image_path: str, target_size: tuple = (256, 512)
    ) -> np.ndarray:
        """
        Load and preprocess image

        Args:
            image_path: Path to image file
            target_size: (height, width)

        Returns:
            frame: (H, W, 3) numpy array in [0, 255]
        """
        img = Image.open(image_path).convert("RGB")

        # Resize to target size
        img = img.resize((target_size[1], target_size[0]))  # PIL uses (W, H)

        # Convert to numpy
        frame = np.array(img, dtype=np.float32)

        return frame

    def paste_object(
        self,
        base_frame: np.ndarray,
        object_frame: np.ndarray,
        position: tuple,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Paste object from one frame into another

        Args:
            base_frame: Base frame (H, W, 3)
            object_frame: Frame containing object to paste
            position: (y, x) position to paste at
            mask: Optional binary mask for object

        Returns:
            modified_frame: Frame with object pasted
        """
        result = base_frame.copy()
        y, x = position
        if y < 0 or x < 0:
            raise ValueError("paste position must be non-negative")

        if mask is None:
            # Simple paste
            h, w = object_frame.shape[:2]
            # Ensure we don't go out of bounds
            y_end = min(y + h, result.shape[0])
            x_end = min(x + w, result.shape[1])

            result[y:y_end, x:x_end] = object_frame[: y_end - y, : x_end - x]
        else:
            # Masked paste (for non-rectangular objects)
            h, w = object_frame.shape[:2]
            if y + h > result.shape[0] or x + w > result.shape[1]:
                raise ValueError("masked paste must fit within base_frame")
            for c in range(3):  # RGB channels
                result[y : y + h, x : x + w, c] = np.where(
                    mask, object_frame[:, :, c], result[y : y + h, x : x + w, c]
                )

        return result

    def generate_from_edited_frame(
        self,
        edited_frame: np.ndarray,
        actions: List[int],
        num_frames_to_generate: int = 100,
    ) -> List[np.ndarray]:
        """
        Generate gameplay from manually edited frame

        Paper Appendix A.4 methodology:
        "We replicate the same frame for the entirety of the history buffer"

        Args:
            edited_frame: Manually edited starting frame
            actions: Sequence of actions to execute
            num_frames_to_generate: Number of frames to generate

        Returns:
            List of generated frames
        """
        # Convert to tensor
        frame_tensor = torch.from_numpy(edited_frame).permute(2, 0, 1)  # (3, H, W)
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Replicate for entire context
        context_frames = frame_tensor.unsqueeze(1).repeat(
            1, self.model.context_length, 1, 1, 1
        )
        # (1, context_length, 3, H, W)

        # Use "no action" for initial context
        context_actions = torch.zeros(
            1, self.model.context_length, dtype=torch.long, device=self.device
        )

        generated_frames = []

        print(f"Generating {num_frames_to_generate} frames from edited image...")

        for i in range(num_frames_to_generate):
            action = actions[i] if i < len(actions) else 0
            # The current action produces the next sampled frame.
            context_actions = condition_current_action(context_actions, action)

            # Generate next frame
            with torch.no_grad():
                generated = self.model.generate(
                    context_frames,
                    context_actions,
                    num_inference_steps=4,
                    guidance_scale=1.5,
                )

            # Convert to numpy
            frame_np = generated[0].permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

            generated_frames.append(frame_np)

            # Update context
            generated = generated.unsqueeze(1)  # (1, 1, 3, H, W)
            context_frames = torch.cat([context_frames[:, 1:], generated], dim=1)

            action_tensor = torch.tensor([[action]], device=self.device)
            context_actions = torch.cat([context_actions[:, 1:], action_tensor], dim=1)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{num_frames_to_generate} frames")

        return generated_frames

    def insert_character(
        self,
        base_frame_path: str,
        character_frame_path: str,
        position: tuple,
        actions: List[int],
        output_video_path: str = "modded_gameplay.mp4",
        num_frames: int = 100,
    ):
        """
        Insert character from one level into another and generate gameplay

        Paper Appendix A.4: "inserting a monster from an advanced level into an early one"

        Args:
            base_frame_path: Base level frame
            character_frame_path: Frame with character to insert
            position: (y, x) where to insert
            actions: Actions to execute
            output_video_path: Output video
            num_frames: Frames to generate
        """
        print("=" * 60)
        print("Image-Based Character Insertion")
        print("=" * 60)

        # Load images
        base = self.load_and_preprocess_image(base_frame_path)
        character = self.load_and_preprocess_image(character_frame_path)

        # Extract character (simple approach - would use segmentation in production)
        # For now, paste a region
        modified_frame = self.paste_object(base, character[:100, :100], position)

        print(f"Modified frame created")

        # Generate gameplay
        generated_frames = self.generate_from_edited_frame(
            modified_frame, actions, num_frames
        )

        # Save as video
        import imageio

        imageio.mimsave(output_video_path, generated_frames, fps=20)

        print(f"\n✓ Saved modded gameplay to {output_video_path}")
        print("=" * 60)

        return generated_frames

    def change_level_layout(
        self,
        base_frame_path: str,
        layout_elements: List[dict],  # List of {image_path, position}
        actions: List[int],
        output_video_path: str = "modified_level.mp4",
        num_frames: int = 100,
    ):
        """
        Modify level layout by adding walls, doors, pools, etc.

        Paper Appendix A.4: "inserting features such as walls, doors, or pools"

        Args:
            base_frame_path: Base level frame
            layout_elements: List of elements to add
            actions: Actions sequence
            output_video_path: Output video
            num_frames: Frames to generate
        """
        print("=" * 60)
        print("Level Layout Modification")
        print("=" * 60)

        # Load base frame
        modified_frame = self.load_and_preprocess_image(base_frame_path)

        # Add each layout element
        for element in layout_elements:
            element_img = self.load_and_preprocess_image(element["image_path"])
            modified_frame = self.paste_object(
                modified_frame, element_img, element["position"]
            )

        print(f"Added {len(layout_elements)} layout elements")

        # Generate gameplay
        generated_frames = self.generate_from_edited_frame(
            modified_frame, actions, num_frames
        )

        # Save
        import imageio

        imageio.mimsave(output_video_path, generated_frames, fps=20)

        print(f"\n✓ Saved modified level to {output_video_path}")
        print("=" * 60)

        return generated_frames


if __name__ == "__main__":
    print("Image-Based Modding System")
    print("\nThis allows you to modify games by editing frames!")
    print("\nCapabilities:")
    print("  - Insert characters into new locations")
    print("  - Modify level layouts")
    print("  - Add new objects")
    print("  - Change visual style")

    print("\nBased on paper's Appendix A.4")
    print("\nUsage:")
    print("  modding = ImageBasedModding(model)")
    print("  modding.insert_character('level.png', 'monster.png', (100, 200), actions)")

    print("\nRequires:")
    print("  - Trained GameNGen model")
    print("  - Source images to modify/paste")
    print("  - Action sequence for generation")
