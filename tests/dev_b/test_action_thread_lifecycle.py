"""Lifecycle tests for application.action_thread.ActionThread.

These tests exercise only thread startup, queue polling, timeout
handling, graceful shutdown, and exception isolation. No dependency's
real behavior (mouse injection, keyboard injection, OS interaction) is
exercised; every constructor dependency is a mock.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from application.action_thread import ActionThread


class _FakeQueue:
    """A minimal stand-in for queue.Queue with scriptable get() behavior.

    Attributes:
        get_calls: List of the timeout values passed to each get() call.
    """

    def __init__(self, items: list[Any] | None = None) -> None:
        """Initializes the fake queue.

        Args:
            items: Optional list of items to return in order, one per
                call to get(). Once exhausted, get() raises
                queue.Empty on every subsequent call.
        """
        self._items = list(items) if items else []
        self._lock = threading.Lock()
        self.get_calls: list[float] = []

    def get(self, timeout: float | None = None) -> Any:
        """Returns the next scripted item or raises queue.Empty.

        Args:
            timeout: Ignored except for being recorded in get_calls.

        Returns:
            The next item from the scripted list.

        Raises:
            queue.Empty: If no scripted items remain.
        """
        with self._lock:
            self.get_calls.append(timeout)
            if not self._items:
                raise queue.Empty
            return self._items.pop(0)


class _RaisingThenEmptyQueue:
    """A fake queue whose get() raises once, then always raises Empty."""

    def __init__(self, exception: BaseException) -> None:
        """Initializes the fake queue.

        Args:
            exception: The exception instance to raise on the first
                call to get().
        """
        self._exception = exception
        self._raised = False
        self._lock = threading.Lock()

    def get(self, timeout: float | None = None) -> Any:
        """Raises the scripted exception once, then queue.Empty forever.

        Args:
            timeout: Ignored.

        Raises:
            BaseException: The scripted exception, on the first call.
            queue.Empty: On every subsequent call.
        """
        with self._lock:
            if not self._raised:
                self._raised = True
                raise self._exception
            raise queue.Empty


@pytest.fixture()
def dependencies() -> dict[str, Any]:
    """Returns a dict of mocked constructor dependencies for ActionThread."""
    return {
        "settings_manager": MagicMock(name="settings_manager"),
        "coordinate_mapper": MagicMock(name="coordinate_mapper"),
        "mouse_injector": MagicMock(name="mouse_injector"),
        "keyboard_injector": MagicMock(name="keyboard_injector"),
        "event_logger": MagicMock(name="event_logger"),
    }


def _make_thread(
    dependencies: dict[str, Any],
    result_queue: Any,
    stop_event: threading.Event,
    queue_timeout: float = 0.05,
) -> ActionThread:
    """Constructs an ActionThread wired to the given queue and dependencies."""
    return ActionThread(
        result_queue=result_queue,
        stop_event=stop_event,
        settings_manager=dependencies["settings_manager"],
        coordinate_mapper=dependencies["coordinate_mapper"],
        mouse_injector=dependencies["mouse_injector"],
        keyboard_injector=dependencies["keyboard_injector"],
        event_logger=dependencies["event_logger"],
        app_signals=None,
        queue_timeout=queue_timeout,
    )


def test_start(dependencies: dict[str, Any]) -> None:
    """Starting the thread transitions is_running() to True."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(), stop_event)

    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while not thread.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert thread.is_running() is True
    finally:
        thread.stop()


def test_stop(dependencies: dict[str, Any]) -> None:
    """stop() signals shutdown and waits for the thread to finish."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(), stop_event)

    thread.start()
    time.sleep(0.05)
    thread.stop()

    assert stop_event.is_set() is True
    assert thread.is_running() is False
    assert thread.isFinished() is True


def test_stop_twice(dependencies: dict[str, Any]) -> None:
    """Calling stop() twice does not raise and remains safe."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(), stop_event)

    thread.start()
    time.sleep(0.05)
    thread.stop()
    thread.stop()

    assert thread.is_running() is False
    assert thread.isFinished() is True


def test_stop_without_start(dependencies: dict[str, Any]) -> None:
    """Calling stop() on a thread that was never started does not raise."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(), stop_event)

    thread.stop()

    assert stop_event.is_set() is True
    assert thread.is_running() is False


def test_empty_queue_does_not_crash(dependencies: dict[str, Any]) -> None:
    """A queue that only ever raises queue.Empty does not crash the thread."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(items=[]), stop_event)

    thread.start()
    time.sleep(0.2)
    assert thread.is_running() is True
    thread.stop()

    assert thread.isFinished() is True


def test_timeout_passed_to_queue_get(dependencies: dict[str, Any]) -> None:
    """The configured queue_timeout is passed to every result_queue.get() call."""
    stop_event = threading.Event()
    fake_queue = _FakeQueue(items=[])
    thread = _make_thread(dependencies, fake_queue, stop_event, queue_timeout=0.07)

    thread.start()
    time.sleep(0.25)
    thread.stop()

    assert len(fake_queue.get_calls) > 0
    assert all(call_timeout == 0.07 for call_timeout in fake_queue.get_calls)


def test_graceful_shutdown(dependencies: dict[str, Any]) -> None:
    """The thread terminates promptly and cleanly after stop() is called."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(items=[]), stop_event)

    thread.start()
    time.sleep(0.05)

    start_time = time.monotonic()
    thread.stop(timeout_ms=2000)
    elapsed = time.monotonic() - start_time

    assert thread.isFinished() is True
    assert elapsed < 2.0


def test_exception_inside_loop_does_not_crash_thread(
    dependencies: dict[str, Any]
) -> None:
    """An unexpected exception from result_queue.get() is logged and survived."""
    stop_event = threading.Event()
    faulty_queue = _RaisingThenEmptyQueue(RuntimeError("simulated failure"))
    thread = _make_thread(dependencies, faulty_queue, stop_event)

    thread.start()
    time.sleep(0.2)
    assert thread.is_running() is True

    thread.stop()
    assert thread.isFinished() is True


def test_exception_inside_handle_result_does_not_crash_thread(
    dependencies: dict[str, Any]
) -> None:
    """An unexpected exception while handling a result does not crash the thread."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(items=[{"bad": "result"}]), stop_event)

    original_handle_result = thread._handle_result

    def _raising_handle_result(result: Any) -> None:
        raise RuntimeError("simulated handler failure")

    thread._handle_result = _raising_handle_result  # type: ignore[method-assign]

    thread.start()
    time.sleep(0.15)
    assert thread.is_running() is True

    thread.stop()
    assert thread.isFinished() is True
    assert original_handle_result is not None


def test_thread_finishes(dependencies: dict[str, Any]) -> None:
    """After stop(), the underlying QThread reports finished and not running."""
    stop_event = threading.Event()
    thread = _make_thread(dependencies, _FakeQueue(items=[]), stop_event)

    thread.start()
    time.sleep(0.05)
    thread.stop()

    assert thread.isFinished() is True
    assert thread.isRunning() is False
    assert thread.is_running() is False
