from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    from PIL import Image

    from exo.worker.engines.video.config import VideoModelConfig


class VideoModelAdapter(ABC):
    """Abstract base class for video generation model adapters."""

    def __init__(
        self,
        config: VideoModelConfig,
        model_id: str,
        local_path: Path,
    ) -> None:
        self._config = config
        self._model_id = model_id
        self._local_path = local_path

    @property
    def config(self) -> VideoModelConfig:
        return self._config

    @abstractmethod
    def load(self) -> None:
        """Load model weights from local_path."""
        ...

    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: str | None = None) -> mx.array:
        """Encode text prompt into embeddings."""
        ...

    @abstractmethod
    def create_latents(
        self,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> mx.array:
        """Create initial noise latents for video generation."""
        ...

    @abstractmethod
    def denoise_step(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        timestep: float,
        guidance_scale: float | None = None,
    ) -> mx.array:
        """Run a single denoising step."""
        ...

    @abstractmethod
    def decode_latents(self, latents: mx.array) -> list[Image.Image]:
        """Decode latents to a list of PIL Image frames."""
        ...

    @abstractmethod
    def get_timesteps(self, num_steps: int) -> list[float]:
        """Return the denoising timestep schedule."""
        ...

    def slice_transformer_blocks(self, start_layer: int, end_layer: int) -> None:  # noqa: B027
        """Remove transformer blocks outside the assigned range for distributed inference."""
