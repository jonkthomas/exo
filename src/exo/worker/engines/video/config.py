from pydantic import BaseModel


class VideoModelConfig(BaseModel):
    model_family: str

    num_transformer_layers: int

    default_steps: dict[str, int]  # {"low": X, "medium": Y, "high": Z}
    num_sync_steps: int  # Number of sync steps for distributed inference

    guidance_scale: float | None = None  # None or <= 1.0 disables CFG

    default_fps: int = 24
    default_duration_seconds: float = 5.0

    def get_steps_for_quality(self, quality: str) -> int:
        return self.default_steps[quality]
