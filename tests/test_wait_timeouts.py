from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

import cohere.utils as utils


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay

    async def async_sleep(self, delay: float) -> None:
        self.sleep(delay)


def setup_wait(monkeypatch: Any, async_mode: bool, states: Any) -> tuple[Any, ...]:
    clock = Clock()
    timer = SimpleNamespace(monotonic=clock.monotonic, time=clock.monotonic, sleep=clock.sleep)
    monkeypatch.setattr(utils, "time", timer)
    monkeypatch.setattr(utils, "asyncio", SimpleNamespace(sleep=clock.async_sleep))
    get_job = AsyncMock(side_effect=states) if async_mode else Mock(side_effect=states)
    monkeypatch.setattr(utils, "async_get_job" if async_mode else "get_job", get_job)
    return clock, timer, get_job


async def wait_for(async_mode: bool, **kwargs: Any) -> Any:
    awaitable: Any = object()
    if async_mode:
        return await utils.async_wait(object(), awaitable, **kwargs)
    return utils.wait(object(), awaitable, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_timeout_clamps_sleep_and_stops_polling(monkeypatch: Any, async_mode: bool) -> None:
    pending = SimpleNamespace(status="processing")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, lambda *args: pending)

    with pytest.raises(TimeoutError, match="wait timed out after 2 seconds"):
        await wait_for(async_mode, timeout=2, interval=10)

    assert clock.sleeps == [2]
    assert get_job.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_wall_clock_adjustment_does_not_expire_wait(monkeypatch: Any, async_mode: bool) -> None:
    pending = SimpleNamespace(status="processing")
    complete = SimpleNamespace(status="complete")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, [pending, complete])
    wall_times = iter([100.0, 10000.0])
    timer.time = lambda: next(wall_times)

    assert await wait_for(async_mode, timeout=5, interval=1) is complete
    assert clock.sleeps == [1]
    assert get_job.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_initial_success_does_not_sleep(monkeypatch: Any, async_mode: bool) -> None:
    complete = SimpleNamespace(status="complete")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, [complete])

    assert await wait_for(async_mode, timeout=0) is complete
    assert clock.sleeps == []
    assert get_job.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_success_before_deadline_is_preserved(monkeypatch: Any, async_mode: bool) -> None:
    pending = SimpleNamespace(status="processing")
    complete = SimpleNamespace(status="complete")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, [pending, complete])

    assert await wait_for(async_mode, timeout=5, interval=1) is complete
    assert clock.sleeps == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_failed_jobs_keep_the_failure_reason(monkeypatch: Any, async_mode: bool) -> None:
    pending = SimpleNamespace(status="processing")
    failed = SimpleNamespace(status="failed")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, [pending, failed])
    reason = Mock(return_value="job failed")
    monkeypatch.setattr(utils, "get_failure_reason", reason)

    with pytest.raises(Exception, match="job failed"):
        await wait_for(async_mode, timeout=5, interval=1)

    reason.assert_called_once_with(failed)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_wait_without_timeout_preserves_interval(monkeypatch: Any, async_mode: bool) -> None:
    pending = SimpleNamespace(status="processing")
    complete = SimpleNamespace(status="complete")
    clock, timer, get_job = setup_wait(monkeypatch, async_mode, [pending, complete])

    assert await wait_for(async_mode, timeout=None, interval=7) is complete
    assert clock.sleeps == [7]
