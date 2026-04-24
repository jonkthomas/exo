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
    """Encode a list of PIL Image frames into MP4 bytes."""
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]
        import numpy as np

        arrays = [np.array(f) for f in frames]
        buf = io.BytesIO()
        iio.imwrite(buf, arrays, extension=".mp4", fps=fps)  # pyright: ignore[reportUnknownMemberType]
        return buf.getvalue()
    except ImportError:
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
    size: str, default_width: int = 512, default_height: int = 512
) -> tuple[int, int]:
    if size == "auto":
        return default_width, default_height
    parts = size.split("x")
    return int(parts[0]), int(parts[1])


def warmup_video_generator(model: DistributedVideoModel) -> None:
    """Warmup is a no-op — mlx-video loads on first generation."""
    logger.info("Video model warmup (deferred to first generation)")


def generate_video(
    model: DistributedVideoModel,
    task: VideoGenerationTaskParams,
    cancel_checker: Callable[[], bool] | None = None,
) -> Generator[VideoGenerationResponse, None, None]:
    """Generate video from a text prompt.

    Uses mlx-video's generate_video() for the full pipeline if the adapter
    supports it (generate_full method), otherwise falls back to the step-by-step
    adapter interface.
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

    # LTX-2 requires num_frames = 1 + 8*k
    if num_frames % 8 != 1:
        num_frames = round((num_frames - 1) / 8) * 8 + 1
        if num_frames < 9:
            num_frames = 9

    width, height = _resolve_dimensions(task.size)

    # LTX-2 requires dimensions divisible by 64 (distilled) or 32 (dev)
    height = (height // 64) * 64
    width = (width // 64) * 64
    if height == 0:
        height = 512
    if width == 0:
        width = 512

    logger.info(
        f"Generating video: {num_frames} frames at {width}x{height}, "
        f"{num_steps} steps, guidance={guidance_scale}, seed={seed}"
    )

    # Use the full mlx-video pipeline if adapter supports it
    if hasattr(model.adapter, "generate_full"):
        frames = model.adapter.generate_full(
            prompt=task.prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            fps=fps,
        )
    else:
        # Fallback: step-by-step adapter interface
        prompt_embeds = model.adapter.encode_prompt(
            task.prompt,
            negative_prompt=(
                task.advanced_params.negative_prompt if task.advanced_params else None
            ),
        )
        mx.eval(prompt_embeds)

        latents = model.adapter.create_latents(seed, num_frames, height, width)
        mx.eval(latents)

        timesteps = model.adapter.get_timesteps(num_steps)
        for _i, t in enumerate(timesteps):
            if cancel_checker and cancel_checker():
                logger.info("Video generation cancelled")
                return
            latents = model.adapter.denoise_step(latents, prompt_embeds, t, guidance_scale)
            mx.eval(latents)

        frames = model.adapter.decode_latents(latents)

    # Encode frames to video
    logger.info(f"Encoding {len(frames)} frames to mp4...")
    video_bytes = _frames_to_mp4_bytes(frames, fps=fps)

    yield VideoGenerationResponse(
        video_data=video_bytes,
        format="mp4",
        num_frames=len(frames),
    )
