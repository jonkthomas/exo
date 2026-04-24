from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from exo.worker.engines.video.config import VideoModelConfig
from exo.worker.engines.video.models.ltx.config import (
    LTX_DEV_CONFIG,
    LTX_DISTILLED_CONFIG,
)

if TYPE_CHECKING:
    from pathlib import Path

    from exo.worker.engines.video.models.base import VideoModelAdapter


def _lazy_ltx_adapter() -> type[VideoModelAdapter]:
    from exo.worker.engines.video.models.ltx.adapter import LtxModelAdapter

    return LtxModelAdapter


_ADAPTER_REGISTRY: dict[str, Callable[[], type[VideoModelAdapter]]] = {
    "ltx": _lazy_ltx_adapter,
}

_CONFIG_REGISTRY: dict[str, VideoModelConfig] = {
    "ltx-2-distilled": LTX_DISTILLED_CONFIG,
    "ltx-2-dev": LTX_DEV_CONFIG,
}


def get_config_for_model(model_id: str) -> VideoModelConfig:
    model_id_lower = model_id.lower()
    for pattern, config in _CONFIG_REGISTRY.items():
        if pattern in model_id_lower:
            return config
    raise ValueError(f"No video model config found for model_id={model_id}")


def create_adapter_for_model(
    config: VideoModelConfig,
    model_id: str,
    local_path: Path,
) -> VideoModelAdapter:
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No video adapter registered for family={config.model_family}")
    adapter_cls = factory()
    logger.info(f"Creating video adapter {adapter_cls.__name__} for {model_id}")
    return adapter_cls(config=config, model_id=model_id, local_path=local_path)
