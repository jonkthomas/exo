from __future__ import annotations

import io
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    from exo.worker.engines.video.distributed_model import DistributedVideoModel

from exo.api.types import VideoGenerationTaskParams


@dataclass
class VideoGenerationResponse:
    """Final video generation result."""

    video_data: bytes  # Encoded video bytes (mp4)
    format: str = "mp4"
    num_frames: int = 0


def _frames_to_mp4_bytes(frames: list[Image.Image], fps: int = 24) -> bytes:
    """Encode a list of PIL Image frames into MP4 bytes.

    Uses a simple uncompressed approach. For production, consider using ffmpeg.
    """
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]
        import numpy as np

        arrays = [np.array(f) for f in frames]
        buf = io.BytesIO()
        iio.imwrite(buf, arrays, extension=".mp4", fps=fps)  # pyright: ignore[reportUnknownMemberType]
        return buf.getvalue()
    except ImportError:
        # Fallback: return frames as animated GIF if imageio not available
        logger.warning("imageio not available, falling back to GIF output")
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        return buf.getvalue()


def _resolve_dimensions(
    size: str, default_width: int = 768, default_height: int = 512
) -> tuple[int, int]:
    if size == "auto":
        return default_width, default_height
    parts = size.split("x")
    return int(parts[0]), int(parts[1])


def warmup_video_generator(model: DistributedVideoModel) -> None:
    """Run a minimal forward pass to warm up the model."""
    logger.info("Warming up video model...")
    # Minimal warmup: encode a short prompt
    prompt_embeds = model.adapter.encode_prompt("warmup")
    mx.eval(prompt_embeds)
    logger.info("Video model warmup complete")


def generate_video(
    model: DistributedVideoModel,
    task: VideoGenerationTaskParams,
    cancel_checker: Callable[[], bool] | None = None,
) -> Generator[VideoGenerationResponse, None, None]:
    """Generate video from a text prompt.

    Yields VideoGenerationResponse when complete.
    """
    quality = task.quality or "medium"
    num_steps = model.config.get_steps_for_quality(quality)
    if task.advanced_params and task.advanced_params.num_inference_steps:
        num_steps = task.advanced_params.num_inference_steps

    guidance_scale = model.config.guidance_scale
    if task.advanced_params and task.advanced_params.guidance:
        guidance_scale = task.advanced_params.guidance

    seed = 42
    if task.advanced_params and task.advanced_params.seed is not None:
        seed = task.advanced_params.seed

    fps = task.fps
    duration = task.duration_seconds
    num_frames = int(fps * duration)

    width, height = _resolve_dimensions(task.size)

    logger.info(
        f"Generating video: {num_frames} frames at {width}x{height}, "
        f"{num_steps} steps, guidance={guidance_scale}"
    )

    # 1. Encode prompt
    prompt_embeds = model.adapter.encode_prompt(
        task.prompt,
        negative_prompt=(
            task.advanced_params.negative_prompt
            if task.advanced_params
            else None
        ),
    )
    mx.eval(prompt_embeds)

    # 2. Create initial latents
    latents = model.adapter.create_latents(seed, num_frames, height, width)
    mx.eval(latents)

    # 3. Iterative denoising
    timesteps = model.adapter.get_timesteps(num_steps)
    for i, t in enumerate(timesteps):
        if cancel_checker and cancel_checker():
            logger.info("Video generation cancelled")
            return

        latents = model.adapter.denoise_step(
            latents, prompt_embeds, t, guidance_scale
        )
        mx.eval(latents)

        if (i + 1) % 5 == 0:
            logger.info(f"Denoising step {i + 1}/{num_steps}")

    # 4. Decode latents to frames
    logger.info("Decoding latents to video frames...")
    frames = model.adapter.decode_latents(latents)

    # 5. Encode frames to video
    logger.info(f"Encoding {len(frames)} frames to mp4...")
    video_bytes = _frames_to_mp4_bytes(frames, fps=fps)

    yield VideoGenerationResponse(
        video_data=video_bytes,
        format="mp4",
        num_frames=len(frames),
    )
