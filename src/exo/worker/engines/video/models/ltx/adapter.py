from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from PIL import Image

from exo.worker.engines.video.models.base import VideoModelAdapter

if TYPE_CHECKING:
    from exo.worker.engines.video.config import VideoModelConfig


class LtxModelAdapter(VideoModelAdapter):
    """Model adapter for LTX-Video-2 models.

    LTX-2 is a DiT-based video generation model with:
    - Gemma 3 text encoder (or T5 for some variants)
    - Asymmetric dual-stream transformer (video + audio, we handle video only)
    - 3D spatiotemporal VAE decoder
    - Flow matching scheduler
    """

    def __init__(
        self,
        config: VideoModelConfig,
        model_id: str,
        local_path: Path,
    ) -> None:
        super().__init__(config, model_id, local_path)
        self._text_encoder: nn.Module | None = None
        self._transformer: nn.Module | None = None
        self._vae: nn.Module | None = None
        self._scheduler_config: dict[str, object] | None = None

    def load(self) -> None:
        """Load LTX-2 model components from local path.

        TODO: Implement full model loading once mlx-video or a dedicated
        MLX implementation for LTX-2 is integrated.
        """
        logger.info(f"Loading LTX-2 model from {self._local_path}")

        # Load scheduler config if available
        scheduler_path = self._local_path / "scheduler" / "scheduler_config.json"
        if scheduler_path.exists():
            with open(scheduler_path) as f:
                self._scheduler_config = json.load(f)

        # TODO: Load text encoder, transformer, and VAE weights
        # This requires integration with mlx-video or a custom MLX implementation
        # of the LTX-2 architecture.
        #
        # For now, this adapter provides the interface scaffolding.
        # The actual weight loading will be implemented when we integrate
        # with the mlx-video package or port the LTX-2 PyTorch code to MLX.
        logger.warning(
            "LTX-2 adapter: model loading is scaffolded. "
            "Full implementation requires mlx-video integration."
        )

    def encode_prompt(self, prompt: str, negative_prompt: str | None = None) -> mx.array:
        """Encode text prompt using the text encoder (Gemma 3 or T5)."""
        if self._text_encoder is None:
            # Return placeholder embeddings for scaffolding
            logger.warning("Text encoder not loaded, returning placeholder embeddings")
            return mx.zeros((1, 128, 2048))

        # TODO: Implement actual prompt encoding
        raise NotImplementedError("LTX-2 prompt encoding not yet implemented")

    def create_latents(
        self,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> mx.array:
        """Create initial noise latents for video diffusion.

        LTX-2 uses 4x16x16 spatiotemporal compression, so latent dimensions
        are: (batch, channels, temporal, height//16, width//16)
        """
        mx.random.seed(seed)
        # LTX-2 latent space: 128 channels, compressed spatiotemporally
        temporal_len = max(1, num_frames // 4)
        latent_h = height // 16
        latent_w = width // 16
        return mx.random.normal((1, 128, temporal_len, latent_h, latent_w))

    def denoise_step(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        timestep: float,
        guidance_scale: float | None = None,
    ) -> mx.array:
        """Run a single denoising step through the transformer."""
        if self._transformer is None:
            # Placeholder: return slightly less noisy latents
            return latents * 0.99

        # TODO: Implement actual denoising step
        raise NotImplementedError("LTX-2 denoising not yet implemented")

    def decode_latents(self, latents: mx.array) -> list[Image.Image]:
        """Decode latents to video frames using the 3D VAE decoder."""
        if self._vae is None:
            # Return placeholder frames for scaffolding
            logger.warning("VAE not loaded, returning placeholder frames")
            num_frames = latents.shape[2] * 4 if latents.ndim >= 3 else 24
            return [
                Image.new("RGB", (512, 512), color=(64, 64, 64))
                for _ in range(num_frames)
            ]

        # TODO: Implement actual VAE decoding
        raise NotImplementedError("LTX-2 VAE decoding not yet implemented")

    def get_timesteps(self, num_steps: int) -> list[float]:
        """Return flow matching timestep schedule.

        LTX-2 uses a linear flow matching schedule from 1.0 to 0.0.
        """
        return [1.0 - i / num_steps for i in range(num_steps)]

    def slice_transformer_blocks(self, start_layer: int, end_layer: int) -> None:
        """Slice transformer blocks for distributed pipeline parallelism."""
        logger.info(f"Slicing LTX-2 transformer: layers [{start_layer}, {end_layer})")
        # TODO: Implement layer slicing once transformer is loaded
