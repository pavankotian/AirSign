"""Unit tests for application.db.event_logger.EventLogger."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from application.db.event_logger import EventLogger


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Returns a path to a session database within a temporary directory."""
    return tmp_path / "airsign_session.db"


@pytest.fixture()
def logger_instance(db_path: Path) -> EventLogger:
    """Returns an EventLogger instance, closed automatically after the test."""
    instance = EventLogger(db_path=db_path)
    yield instance
    instance.close()


def _sample_event(
    timestamp: float = 1.0,
    gesture: str = "PINCH",
    fsm_state: str = "PINCH_START",
    fsm_event: str | None = "PINCH_DETECTED",
    action: str | None = "left_click",
    confidence: float = 0.95,
    latency_ms: float = 12.5,
) -> dict[str, object]:
    """Builds a dictionary of keyword arguments for log_event()."""
    return {
        "timestamp": timestamp,
        "gesture": gesture,
        "fsm_state": fsm_state,
        "fsm_event": fsm_event,
        "action": action,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


def test_database_creation(db_path: Path) -> None:
    """EventLogger creates the database file if it does not already exist."""
    assert not db_path.exists()
    logger_obj = EventLogger(db_path=db_path)
    try:
        assert db_path.exists()
    finally:
        logger_obj.close()


def test_schema_creation(db_path: Path) -> None:
    """EventLogger creates the events table with the expected columns."""
    logger_obj = EventLogger(db_path=db_path)
    try:
        connection = sqlite3.connect(str(db_path))
        try:
            cursor = connection.execute("PRAGMA table_info(events)")
            columns = {row[1] for row in cursor.fetchall()}
        finally:
            connection.close()
    finally:
        logger_obj.close()

    expected_columns = {
        "id",
        "timestamp",
        "gesture",
        "fsm_state",
        "fsm_event",
        "action",
        "confidence",
        "latency_ms",
    }
    assert expected_columns.issubset(columns)


def test_insert_event(logger_instance: EventLogger) -> None:
    """log_event() inserts exactly one row that can be retrieved."""
    logger_instance.log_event(**_sample_event())

    events = logger_instance.get_all_events()
    assert len(events) == 1
    assert events[0]["gesture"] == "PINCH"
    assert events[0]["fsm_state"] == "PINCH_START"
    assert events[0]["fsm_event"] == "PINCH_DETECTED"
    assert events[0]["action"] == "left_click"
    assert events[0]["confidence"] == pytest.approx(0.95)
    assert events[0]["latency_ms"] == pytest.approx(12.5)


def test_multiple_inserts(logger_instance: EventLogger) -> None:
    """Multiple log_event() calls each produce a distinct stored row."""
    for index in range(5):
        logger_instance.log_event(
            **_sample_event(timestamp=float(index), gesture="FIST")
        )

    events = logger_instance.get_all_events()
    assert len(events) == 5
    assert [event["timestamp"] for event in events] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_get_all_events_returns_dicts(logger_instance: EventLogger) -> None:
    """get_all_events() returns a list of plain dictionaries, not tuples."""
    logger_instance.log_event(**_sample_event())

    events = logger_instance.get_all_events()
    assert isinstance(events, list)
    assert all(isinstance(event, dict) for event in events)


def test_get_all_events_empty_database(logger_instance: EventLogger) -> None:
    """get_all_events() returns an empty list when no events exist."""
    assert logger_instance.get_all_events() == []


def test_get_event_count(logger_instance: EventLogger) -> None:
    """get_event_count() reflects the number of rows inserted."""
    assert logger_instance.get_event_count() == 0

    for index in range(3):
        logger_instance.log_event(**_sample_event(timestamp=float(index)))

    assert logger_instance.get_event_count() == 3


def test_session_summary(logger_instance: EventLogger) -> None:
    """get_session_summary() computes correct aggregate statistics."""
    logger_instance.log_event(
        **_sample_event(gesture="PINCH", confidence=0.8, latency_ms=10.0)
    )
    logger_instance.log_event(
        **_sample_event(gesture="PINCH", confidence=0.9, latency_ms=20.0)
    )
    logger_instance.log_event(
        **_sample_event(gesture="FIST", confidence=1.0, latency_ms=30.0)
    )

    summary = logger_instance.get_session_summary()

    assert summary["total_events"] == 3
    assert summary["average_confidence"] == pytest.approx(0.9)
    assert summary["average_latency_ms"] == pytest.approx(20.0)
    assert summary["gesture_counts"] == {"PINCH": 2, "FIST": 1}


def test_session_summary_empty_database(logger_instance: EventLogger) -> None:
    """get_session_summary() handles an empty database without error."""
    summary = logger_instance.get_session_summary()

    assert summary["total_events"] == 0
    assert summary["average_confidence"] == pytest.approx(0.0)
    assert summary["average_latency_ms"] == pytest.approx(0.0)
    assert summary["gesture_counts"] == {}


def test_gesture_counts_group_by(logger_instance: EventLogger) -> None:
    """gesture_counts correctly tallies events per distinct gesture label."""
    gestures = ["PINCH", "PINCH", "FIST", "OPEN_PALM", "FIST", "FIST"]
    for gesture in gestures:
        logger_instance.log_event(**_sample_event(gesture=gesture))

    summary = logger_instance.get_session_summary()
    assert summary["gesture_counts"] == {
        "PINCH": 2,
        "FIST": 3,
        "OPEN_PALM": 1,
    }


def test_clear_session(logger_instance: EventLogger) -> None:
    """clear_session() removes all rows without dropping the table."""
    for index in range(4):
        logger_instance.log_event(**_sample_event(timestamp=float(index)))
    assert logger_instance.get_event_count() == 4

    logger_instance.clear_session()

    assert logger_instance.get_event_count() == 0
    assert logger_instance.get_all_events() == []

    logger_instance.log_event(**_sample_event())
    assert logger_instance.get_event_count() == 1


def test_close(db_path: Path) -> None:
    """close() closes the underlying connection cleanly."""
    logger_obj = EventLogger(db_path=db_path)
    logger_obj.close()

    with pytest.raises(sqlite3.ProgrammingError):
        logger_obj.get_event_count()


def test_close_twice(db_path: Path) -> None:
    """close() may be called more than once without raising."""
    logger_obj = EventLogger(db_path=db_path)
    logger_obj.close()
    logger_obj.close()


def test_concurrent_logging(logger_instance: EventLogger) -> None:
    """Multiple threads may call log_event() concurrently without error."""
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    events_per_thread = 20
    thread_count = 8

    def writer(thread_index: int) -> None:
        try:
            for sample_index in range(events_per_thread):
                logger_instance.log_event(
                    **_sample_event(
                        timestamp=float(thread_index * 1000 + sample_index),
                        gesture="PINCH",
                    )
                )
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(thread_index,))
        for thread_index in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert logger_instance.get_event_count() == events_per_thread * thread_count


def test_invalid_database_path(tmp_path: Path) -> None:
    """Initialization raises an OSError when the path cannot be created."""
    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a directory", encoding="utf-8")
    invalid_path = blocking_file / "session.db"

    with pytest.raises(OSError):
        EventLogger(db_path=invalid_path)
