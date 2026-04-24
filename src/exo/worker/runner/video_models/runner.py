import base64
import math

import mlx.core as mx

from exo.api.types import VideoGenerationTaskParams
from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import ErrorChunk, VideoChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    VideoGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.utils_mlx import initialize_mlx
from exo.worker.engines.video import (
    DistributedVideoModel,
    generate_video,
    initialize_video_model,
    warmup_video_generator,
)
from exo.worker.runner.bootstrap import logger

# Max chunk size for video data (1MB base64 chunks)
MAX_VIDEO_CHUNK_SIZE = 1_000_000


def _is_primary_output_node(shard_metadata: ShardMetadata) -> bool:
    """Check if this node is the primary output node for video generation."""
    if isinstance(shard_metadata, PipelineShardMetadata):
        return shard_metadata.device_rank == shard_metadata.world_size - 1
    return True  # Single node is always primary


class Runner:
    """Video generation runner — mirrors the image runner state machine."""

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ) -> None:
        self.bound_instance = bound_instance
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver

        self.shard_metadata = bound_instance.bound_shard
        self.current_status: RunnerStatus = RunnerIdle()
        self.video_model: DistributedVideoModel | None = None
        self.group: mx.distributed.Group | None = None
        self._cancelled_tasks: set[TaskId] = set()

    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.bound_instance.bound_runner_id,
                runner_status=status,
            )
        )

    def acknowledge_task(self, task: Task) -> None:
        self.event_sender.send(
            TaskAcknowledged(
                task_id=task.task_id,
                runner_id=self.bound_instance.bound_runner_id,
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def _check_cancelled(self, task_id: TaskId) -> bool:
        """Check if a task has been cancelled."""
        for cancel_id in self.cancel_receiver.collect():
            self._cancelled_tasks.add(cancel_id)
        return task_id in self._cancelled_tasks or CANCEL_ALL_TASKS in self._cancelled_tasks

    def _run_video_task(
        self,
        task: Task,
        task_params: VideoGenerationTaskParams,
        command_id: CommandId,
    ) -> None:
        assert self.video_model
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)

        def cancel_checker() -> bool:
            return self._check_cancelled(task.task_id)

        try:
            video_index = 0
            for response in generate_video(
                model=self.video_model,
                task=task_params,
                cancel_checker=cancel_checker,
            ):
                if _is_primary_output_node(self.shard_metadata):
                    # Encode video data as base64 and chunk it
                    encoded_data = base64.b64encode(response.video_data).decode("utf-8")
                    total_chunks = max(1, math.ceil(len(encoded_data) / MAX_VIDEO_CHUNK_SIZE))

                    for chunk_idx in range(total_chunks):
                        start = chunk_idx * MAX_VIDEO_CHUNK_SIZE
                        end = min(start + MAX_VIDEO_CHUNK_SIZE, len(encoded_data))
                        chunk_data = encoded_data[start:end]

                        is_last_chunk = chunk_idx == total_chunks - 1
                        self.event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=VideoChunk(
                                    model=ModelId(task_params.model),
                                    data=chunk_data,
                                    chunk_index=chunk_idx,
                                    total_chunks=total_chunks,
                                    video_index=video_index,
                                    format=response.format,
                                    finish_reason="stop" if is_last_chunk else None,
                                ),
                            )
                        )
                    video_index += 1

        except Exception as e:
            logger.opt(exception=e).warning(f"Video generation failed: {e}")
            if _is_primary_output_node(self.shard_metadata):
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=ErrorChunk(
                            model=ModelId(task_params.model),
                            error_message=str(e),
                        ),
                    )
                )

        self.update_status(RunnerReady())

    def handle_task(self, task: Task) -> None:
        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.group = initialize_mlx(self.bound_instance)
                self.update_status(RunnerConnected())

            case LoadModel() if (
                isinstance(self.current_status, RunnerConnected)
                and self.group is not None
            ) or (
                isinstance(self.current_status, RunnerIdle) and self.group is None
            ):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                assert ModelTask.TextToVideo in self.shard_metadata.model_card.tasks
                self.video_model = initialize_video_model(self.bound_instance)
                self.update_status(RunnerLoaded())

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                if self.video_model:
                    warmup_video_generator(self.video_model)
                self.update_status(RunnerReady())

            case VideoGeneration(
                task_params=task_params, command_id=command_id
            ) if isinstance(self.current_status, RunnerReady):
                self._run_video_task(task, task_params, command_id)

            case Shutdown():
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.update_status(RunnerShutdown())

            case _:
                logger.warning(
                    f"Video runner ignoring task {type(task).__name__} "
                    f"in state {type(self.current_status).__name__}"
                )

    def main(self) -> None:
        logger.info("Video runner starting")
        self.update_status(RunnerIdle())

        with self.task_receiver as tasks:
            for task in tasks:
                self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                was_cancelled = (task.task_id in self._cancelled_tasks) or (
                    CANCEL_ALL_TASKS in self._cancelled_tasks
                )
                if not was_cancelled:
                    self.send_task_status(task, TaskStatus.Complete)
                self.update_status(self.current_status)

                if isinstance(self.current_status, RunnerShutdown):
                    break

        logger.info("Video runner shutting down")
