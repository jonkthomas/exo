from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from loguru import logger

from exo.shared.constants import EXO_MODELS_DIRS
from exo.worker.engines.video.models import (
    create_adapter_for_model,
    get_config_for_model,
)

if TYPE_CHECKING:
    from exo.shared.types.worker.instances import BoundInstance
    from exo.worker.engines.video.models.base import VideoModelAdapter


class DistributedVideoModel:
    """Wraps a video model adapter with distributed inference support."""

    def __init__(
        self,
        adapter: VideoModelAdapter,
        group: mx.distributed.Group | None = None,
    ) -> None:
        self.adapter = adapter
        self.group = group

    @property
    def config(self):
        return self.adapter.config


def _find_local_model_path(model_id: str) -> Path | None:
    """Look for the model in known download directories."""
    safe_name = model_id.replace("/", "--")
    for model_dir in EXO_MODELS_DIRS:
        candidate = Path(model_dir) / safe_name
        if candidate.exists():
            return candidate
    return None


def initialize_video_model(bound_instance: BoundInstance) -> DistributedVideoModel:
    """Factory to create and initialize a DistributedVideoModel from a BoundInstance."""
    model_id = bound_instance.instance.shard_assignments.model_id
    shard = bound_instance.bound_shard

    logger.info(f"Initializing video model: {model_id}")

    local_path = _find_local_model_path(model_id)
    if local_path is None:
        raise FileNotFoundError(
            f"Video model weights not found locally for {model_id}. "
            "Please download the model first."
        )

    config = get_config_for_model(model_id)
    adapter = create_adapter_for_model(config, model_id, local_path)

    # Slice transformer for distributed inference if needed
    if shard.world_size > 1:
        adapter.slice_transformer_blocks(shard.start_layer, shard.end_layer)

    adapter.load()

    logger.info(f"Video model {model_id} loaded successfully")
    return DistributedVideoModel(adapter=adapter)
