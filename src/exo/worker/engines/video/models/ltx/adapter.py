from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from loguru import logger
from PIL import Image

from exo.worker.engines.video.models.base import VideoModelAdapter

if TYPE_CHECKING:
    from exo.worker.engines.video.config import VideoModelConfig


class LtxModelAdapter(VideoModelAdapter):
    """Model adapter for LTX-Video-2 using mlx-video.

    Delegates to mlx_video.models.ltx_2.generate.generate_video() which handles
    the full pipeline: text encoding, denoising, VAE decoding.
    """

    def __init__(
        self,
        config: VideoModelConfig,
        model_id: str,
        local_path: Path,
    ) -> None:
        super().__init__(config, model_id, local_path)
        self._loaded = False

    def load(self) -> None:
        """Verify model files exist. Actual loading is deferred to generate_video()."""
        logger.info(f"Verifying LTX-2 model at {self._local_path}")
        model_file = self._local_path / "ltx-2-19b-distilled.safetensors"
        text_encoder_dir = self._local_path / "text_encoder"
        if not model_file.exists() and not list(self._local_path.glob("*.safetensors")):
            raise FileNotFoundError(f"No safetensors found at {self._local_path}")
        if not text_encoder_dir.exists():
            raise FileNotFoundError(f"Text encoder not found at {text_encoder_dir}")
        self._loaded = True
        logger.info("LTX-2 model files verified")

    def encode_prompt(self, prompt: str, negative_prompt: str | None = None) -> mx.array:
        """Not used — mlx-video handles encoding internally."""
        return mx.zeros((1, 1, 1))

    def create_latents(
        self,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> mx.array:
        """Not used — mlx-video handles latent creation internally."""
        return mx.zeros((1,))

    def denoise_step(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        timestep: float,
        guidance_scale: float | None = None,
    ) -> mx.array:
        """Not used — mlx-video handles denoising internally."""
        return latents

    def decode_latents(self, latents: mx.array) -> list[Image.Image]:
        """Not used — mlx-video handles decoding internally."""
        return []

    def get_timesteps(self, num_steps: int) -> list[float]:
        """Not used — mlx-video handles scheduling internally."""
        return []

    def generate_full(
        self,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        seed: int,
        num_steps: int,
        guidance_scale: float | None,
        fps: int = 24,
    ) -> list[Image.Image]:
        """Generate video using mlx-video's full pipeline.

        This is the main entry point — it delegates to mlx_video's generate_video()
        which handles text encoding, denoising, and VAE decoding internally.
        """
        from mlx_video.models.ltx_2.generate import (
            PipelineType,  # pyright: ignore[reportMissingImports]
        )
        from mlx_video.models.ltx_2.generate import (
            generate_video as mlx_generate,  # pyright: ignore[reportMissingImports]
        )

        # Determine if this is a distilled or dev model
        is_distilled = "distilled" in self._model_id.lower()
        pipeline = PipelineType.DISTILLED if is_distilled else PipelineType.DEV

        # Use HuggingFace repo ID so mlx-video can resolve the model
        # in its expected directory structure (transformer/, vae/, etc.)
        model_repo = self._model_id  # e.g. "mlx-community/LTX-2-distilled-bf16"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output.mp4")

            logger.info(
                f"Running LTX-2 generation: {width}x{height}, {num_frames} frames, "
                f"pipeline={pipeline.value}, steps={num_steps}"
            )

            video_np = mlx_generate(
                model_repo=model_repo,
                text_encoder_repo=None,
                prompt=prompt,
                pipeline=pipeline,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                cfg_scale=guidance_scale or 4.0,
                seed=seed,
                fps=fps,
                output_path=output_path,
                verbose=True,
                audio=False,
            )

            # Convert numpy frames to PIL Images
            frames: list[Image.Image] = []
            for frame_np in video_np:
                frames.append(Image.fromarray(frame_np))  # pyright: ignore[reportUnknownArgumentType]

            logger.info(f"Generated {len(frames)} frames")
            return frames
