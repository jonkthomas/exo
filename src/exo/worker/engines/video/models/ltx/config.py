from exo.worker.engines.video.config import VideoModelConfig

LTX_DISTILLED_CONFIG = VideoModelConfig(
    model_family="ltx",
    num_transformer_layers=48,
    default_steps={"low": 8, "medium": 20, "high": 40},
    num_sync_steps=4,
    guidance_scale=7.5,
    default_fps=24,
    default_duration_seconds=5.0,
)

LTX_DEV_CONFIG = VideoModelConfig(
    model_family="ltx",
    num_transformer_layers=48,
    default_steps={"low": 20, "medium": 40, "high": 80},
    num_sync_steps=4,
    guidance_scale=7.5,
    default_fps=24,
    default_duration_seconds=5.0,
)
