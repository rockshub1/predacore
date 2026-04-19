"""
Lane Queue — Session-serialized task execution.

Inspired by OpenClaw's reliability design: each user/session gets a
dedicated "lane" where tasks execute serially (FIFO). This prevents
race conditions and ensures multi-step operations are atomic.

Usage:
    queue = LaneQueue()
    result = await queue.submit("user_123", my_coroutine, arg1, arg2)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class LaneTask:
    """A single task queued in a lane."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: str | None = None
    coroutine_fn: Callable | None = field(default=None, repr=False)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    timeout: float = 300.0  # 5 minute default
    _future: asyncio.Future | None = field(default=None, repr=False)


@dataclass
class Lane:
    """A serial execution lane for one session/user."""

    session_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_task: LaneTask | None = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    _worker_task: asyncio.Task | None = field(default=None, repr=False)


class LaneQueue:
    """
    Session-serialized task execution queue.

    Each session gets its own Lane. Tasks within a Lane execute
    serially (FIFO). Different Lanes execute concurrently.

    Design choices:
    - Serial-by-default prevents state corruption in multi-step operations
    - Lock per lane ensures atomic tool loops
    - Timeout + monitoring prevents hung tasks from blocking lanes
    - Lazy lane creation: lanes are created on first task
    - Per-lane queue size limits prevent OOM under load
    - Dead letter queue captures failed tasks for later inspection
    """

    # Maximum tasks that can be queued in a single lane before rejection.
    MAX_QUEUE_SIZE: int = 100

    # Maximum failed tasks retained in the dead letter queue.
    MAX_DEAD_LETTERS: int = 500

    def __init__(self, max_lanes: int = 100, default_timeout: float = 300.0):
        self._lanes: dict[str, Lane] = {}
        self._max_lanes = max_lanes
        self._default_timeout = default_timeout
        self._shutdown = False
        self._dead_letters: list[LaneTask] = []
        logger.info(
            "LaneQueue initialized (max_lanes=%d, timeout=%.0fs)",
            max_lanes,
            default_timeout,
        )

    def _get_or_create_lane(self, session_id: str) -> Lane:
        """Get existing lane or create a new one."""
        if session_id not in self._lanes:
            if len(self._lanes) >= self._max_lanes:
                # Evict oldest inactive lane
                oldest = min(self._lanes.values(), key=lambda l: l.last_active)
                if oldest.current_task is None:
                    logger.info("Evicting inactive lane: %s", oldest.session_id)
                    # Cancel the orphaned worker task before removing
                    if (
                        oldest._worker_task is not None
                        and not oldest._worker_task.done()
                    ):
                        oldest._worker_task.cancel()
                    del self._lanes[oldest.session_id]
                else:
                    logger.warning(
                        "All lanes busy, cannot create new lane for %s", session_id
                    )
                    raise RuntimeError(
                        f"All {self._max_lanes} lanes are busy. Try again shortly."
                    )

            lane = Lane(session_id=session_id)
            self._lanes[session_id] = lane
            # Start worker for this lane
            lane._worker_task = asyncio.create_task(
                self._lane_worker(lane), name=f"lane-{session_id[:8]}"
            )
            logger.debug("Created lane for session %s", session_id)

        return self._lanes[session_id]

    async def submit(
        self,
        session_id: str,
        coro_fn: Callable[..., Coroutine],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a task to a session's lane. Blocks until the task completes.

        Tasks in the same session execute serially. Tasks in different
        sessions execute concurrently.

        Args:
            session_id: The session/user key for lane selection
            coro_fn: An async callable to execute
            *args: Positional arguments for the callable
            timeout: Override default timeout (seconds)
            **kwargs: Keyword arguments for the callable

        Returns:
            The result of the coroutine

        Raises:
            TimeoutError: If task exceeds timeout
            Exception: Any exception raised by the coroutine
        """
        if self._shutdown:
            raise RuntimeError("LaneQueue is shutting down")

        lane = self._get_or_create_lane(session_id)

        # Enforce per-lane queue size limit to prevent OOM
        if lane.queue.qsize() >= self.MAX_QUEUE_SIZE:
            raise RuntimeError(
                f"Lane '{session_id[:8]}' queue full "
                f"({self.MAX_QUEUE_SIZE} pending tasks). Try again later."
            )

        task = LaneTask(
            session_id=session_id,
            coroutine_fn=coro_fn,
            args=args,
            kwargs=kwargs,
            timeout=timeout or self._default_timeout,
        )

        # Use a Future to communicate result back
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        task._future = future
        await lane.queue.put((task, future))
        lane.total_tasks += 1
        lane.last_active = time.time()

        logger.debug(
            "Task %s queued in lane %s (queue_size=%d)",
            task.task_id,
            session_id[:8],
            lane.queue.qsize(),
        )

        # Wait for result
        return await future

    async def _lane_worker(self, lane: Lane) -> None:
        """Worker loop for a single lane — processes tasks serially."""
        logger.debug("Lane worker started for %s", lane.session_id[:8])

        while not self._shutdown:
            try:
                # Wait for next task
                task, future = await asyncio.wait_for(
                    lane.queue.get(),
                    timeout=60.0,  # Check shutdown every 60s
                )
            except asyncio.TimeoutError:
                # No tasks for 60s, check if we should keep the lane alive
                if lane.queue.empty() and time.time() - lane.last_active > 600:
                    logger.debug(
                        "Lane %s idle for 10min, shutting down worker",
                        lane.session_id[:8],
                    )
                    break
                continue

            # Execute task with lane locked (atomic)
            async with lane.lock:
                lane.current_task = task
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()

                try:
                    result = await asyncio.wait_for(
                        task.coroutine_fn(*task.args, **task.kwargs),
                        timeout=task.timeout or self._default_timeout,
                    )
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    lane.completed_tasks += 1
                    if not future.done():
                        future.set_result(result)

                except asyncio.TimeoutError:
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"Task timed out after {task.timeout or self._default_timeout}s"
                    lane.failed_tasks += 1
                    if not future.done():
                        future.set_exception(TimeoutError(task.error))
                    logger.error("Task %s timed out in lane %s", task.task_id, lane.session_id[:8])

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    lane.failed_tasks += 1
                    if not future.done():
                        future.set_exception(e)
                    logger.error(
                        "Task %s failed in lane %s: %s",
                        task.task_id,
                        lane.session_id[:8],
                        e,
                    )

                finally:
                    task.completed_at = time.time()
                    lane.current_task = None
                    lane.last_active = time.time()
                    lane.queue.task_done()

        # Cleanup
        if lane.session_id in self._lanes:
            del self._lanes[lane.session_id]
        logger.debug("Lane worker stopped for %s", lane.session_id[:8])

    def cancel_current(self, session_id: str) -> bool:
        """Cancel the currently running task in a session's lane.

        Returns True if a task was cancelled, False if no task running.
        """
        lane = self._lanes.get(session_id)
        if lane is None or lane.current_task is None:
            return False

        task = lane.current_task
        logger.info(
            "Cancelling task %s in lane %s", task.task_id, session_id[:8]
        )
        task.status = TaskStatus.CANCELLED

        # Cancel the worker's current asyncio task to interrupt execution
        if lane._worker_task and not lane._worker_task.done():
            lane._worker_task.cancel()
            # Recreate the worker so the lane keeps processing future tasks
            lane._worker_task = asyncio.create_task(
                self._lane_worker(lane), name=f"lane-{session_id[:8]}"
            )

        # Resolve any pending future so callers are not left hanging
        if hasattr(task, '_future') and task._future and not task._future.done():
            task._future.set_exception(asyncio.CancelledError())

        return True

    def get_lane_stats(self) -> dict[str, Any]:
        """Get stats for all active lanes."""
        return {
            "active_lanes": len(self._lanes),
            "lanes": {
                sid: {
                    "queue_size": lane.queue.qsize(),
                    "total_tasks": lane.total_tasks,
                    "completed": lane.completed_tasks,
                    "failed": lane.failed_tasks,
                    "has_current_task": lane.current_task is not None,
                    "idle_seconds": time.time() - lane.last_active,
                }
                for sid, lane in self._lanes.items()
            },
        }

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all lane workers and resolve any pending futures.

        Idle workers are parked in ``queue.get()`` with a long inner
        timeout and won't self-exit on the shutdown flag, so we cancel
        them directly. Before cancelling, resolve every caller's future
        so no one is left hanging on a task that will never complete.
        """
        logger.info("LaneQueue shutting down (%d active lanes)...", len(self._lanes))
        self._shutdown = True

        # Resolve futures for anything in-flight or queued so callers of
        # submit() don't hang on results that will never arrive.
        for lane in self._lanes.values():
            # 1. Drain queued tasks that never got to run.
            while not lane.queue.empty():
                try:
                    _, future = lane.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not future.done():
                    future.set_exception(
                        asyncio.CancelledError("LaneQueue shutdown")
                    )
                lane.queue.task_done()

            # 2. Resolve the currently-running task's future.
            if lane.current_task is not None:
                fut = lane.current_task._future
                if fut is not None and not fut.done():
                    fut.set_exception(
                        asyncio.CancelledError("LaneQueue shutdown")
                    )

        # 3. Cancel all worker tasks immediately. Idle workers are parked
        #    in `queue.get()` and won't observe `self._shutdown` until a
        #    task is submitted, which defeats the point of a fast shutdown.
        tasks = [
            lane._worker_task
            for lane in self._lanes.values()
            if lane._worker_task and not lane._worker_task.done()
        ]

        for t in tasks:
            t.cancel()

        if tasks:
            # Wait for cancellation to propagate. Suppress CancelledError
            # from the gathered tasks so shutdown itself doesn't raise.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LaneQueue.shutdown: %d workers did not cancel within %.1fs",
                    len(tasks),
                    timeout,
                )

        self._lanes.clear()
        logger.info("LaneQueue shutdown complete")
