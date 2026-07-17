"""SQLite-backed gesture event logging for the AirSign Developer B track.

This module implements :class:`EventLogger`, the sole component
responsible for interacting with the SQLite session database. It
records every dispatched gesture event and exposes aggregate query
methods used by the UI action log and the PDF session report.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    gesture TEXT NOT NULL,
    fsm_state TEXT NOT NULL,
    fsm_event TEXT,
    action TEXT,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL
)
"""


class EventLogger:
    """Thread-safe SQLite logger for AirSign gesture events.

    This is the only class in the codebase permitted to open a
    connection to the session database. All access is serialized
    through a single reentrant lock so the logger may be shared safely
    across ``ActionThread``, the UI thread, and the PDF export thread.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = "airsign_session.db") -> None:
        """Opens (or creates) the session database and its schema.

        Args:
            db_path: Path to the SQLite database file. Created
                automatically, along with any missing parent
                directories, if it does not already exist.

        Raises:
            sqlite3.Error: If the database file cannot be opened or the
                schema cannot be created.
        """
        self._lock = threading.RLock()
        self.db_path: Path = Path(db_path)
        self._closed = False

        parent = self.db_path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        try:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        except sqlite3.Error:
            logger.exception(
                "Failed to open session database at %s", self.db_path
            )
            raise

        try:
            with self._lock:
                self._connection.execute(_SCHEMA_SQL)
                self._connection.commit()
        except sqlite3.Error:
            logger.exception(
                "Failed to create schema in session database at %s",
                self.db_path,
            )
            raise

        logger.info("Session database ready at %s", self.db_path)

    def log_event(
        self,
        timestamp: float,
        gesture: str,
        fsm_state: str,
        fsm_event: str | None,
        action: str | None,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Inserts one gesture event row and commits immediately.

        Args:
            timestamp: Event time as a Unix epoch float, typically from
                ``time.time()``.
            gesture: The classified gesture label active at this event.
            fsm_state: The FSM state active at this event.
            fsm_event: The FSM event that fired, or ``None`` if this
                row records a non-transition frame.
            action: The OS action dispatched for this event, or
                ``None`` if no action was bound or dispatched.
            confidence: Classifier confidence in the range 0.0-1.0.
            latency_ms: End-to-end pipeline latency in milliseconds for
                this frame.

        Raises:
            sqlite3.Error: If the insert fails for any reason. The
                error is logged before being re-raised.
        """
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO events (
                        timestamp, gesture, fsm_state, fsm_event,
                        action, confidence, latency_ms
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        gesture,
                        fsm_state,
                        fsm_event,
                        action,
                        confidence,
                        latency_ms,
                    ),
                )
                self._connection.commit()
            except sqlite3.Error:
                logger.exception(
                    "Failed to log event (gesture=%s, fsm_event=%s)",
                    gesture,
                    fsm_event,
                )
                raise

        logger.debug(
            "Logged event: gesture=%s fsm_state=%s fsm_event=%s action=%s",
            gesture,
            fsm_state,
            fsm_event,
            action,
        )

    def get_all_events(self) -> list[dict[str, Any]]:
        """Returns every logged event as a list of dictionaries.

        Returns:
            A list of event dictionaries ordered by ``id`` ascending,
            each containing the keys ``id``, ``timestamp``, ``gesture``,
            ``fsm_state``, ``fsm_event``, ``action``, ``confidence``,
            and ``latency_ms``. Returns an empty list if no events have
            been logged.

        Raises:
            sqlite3.Error: If the query fails for any reason. The error
                is logged before being re-raised.
        """
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "SELECT * FROM events ORDER BY id ASC"
                )
                rows = cursor.fetchall()
            except sqlite3.Error:
                logger.exception("Failed to fetch all events")
                raise

        return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        """Returns the total number of logged events.

        Returns:
            The number of rows in the events table, computed via
            ``COUNT(*)``.

        Raises:
            sqlite3.Error: If the query fails for any reason. The error
                is logged before being re-raised.
        """
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "SELECT COUNT(*) AS event_count FROM events"
                )
                row = cursor.fetchone()
            except sqlite3.Error:
                logger.exception("Failed to compute event count")
                raise

        return int(row["event_count"])

    def get_session_summary(self) -> dict[str, Any]:
        """Computes aggregate statistics for the current session.

        Returns:
            A dictionary with the keys:

            - ``total_events``: Total number of logged events.
            - ``average_confidence``: Mean confidence across all
              events, or ``0.0`` if no events exist.
            - ``average_latency_ms``: Mean latency in milliseconds
              across all events, or ``0.0`` if no events exist.
            - ``gesture_counts``: A mapping of gesture label to the
              number of events logged for that gesture, computed via
              ``GROUP BY``.

        Raises:
            sqlite3.Error: If any query fails for any reason. The error
                is logged before being re-raised.
        """
        with self._lock:
            try:
                totals_cursor = self._connection.execute(
                    """
                    SELECT
                        COUNT(*) AS total_events,
                        AVG(confidence) AS average_confidence,
                        AVG(latency_ms) AS average_latency_ms
                    FROM events
                    """
                )
                totals_row = totals_cursor.fetchone()

                gesture_cursor = self._connection.execute(
                    """
                    SELECT gesture, COUNT(*) AS gesture_count
                    FROM events
                    GROUP BY gesture
                    """
                )
                gesture_rows = gesture_cursor.fetchall()
            except sqlite3.Error:
                logger.exception("Failed to compute session summary")
                raise

        total_events = int(totals_row["total_events"])
        average_confidence = (
            float(totals_row["average_confidence"])
            if totals_row["average_confidence"] is not None
            else 0.0
        )
        average_latency_ms = (
            float(totals_row["average_latency_ms"])
            if totals_row["average_latency_ms"] is not None
            else 0.0
        )
        gesture_counts = {
            row["gesture"]: int(row["gesture_count"]) for row in gesture_rows
        }

        return {
            "total_events": total_events,
            "average_confidence": average_confidence,
            "average_latency_ms": average_latency_ms,
            "gesture_counts": gesture_counts,
        }

    def clear_session(self) -> None:
        """Deletes all event rows without dropping the table.

        Raises:
            sqlite3.Error: If the delete fails for any reason. The
                error is logged before being re-raised.
        """
        with self._lock:
            try:
                self._connection.execute("DELETE FROM events")
                self._connection.commit()
            except sqlite3.Error:
                logger.exception("Failed to clear session events")
                raise

        logger.info("Cleared all events from session database at %s", self.db_path)

    def close(self) -> None:
        """Closes the underlying database connection.

        Safe to call multiple times; subsequent calls after the first
        are no-ops.
        """
        with self._lock:
            if self._closed:
                logger.debug(
                    "close() called on already-closed EventLogger for %s",
                    self.db_path,
                )
                return

            try:
                self._connection.close()
            except sqlite3.Error:
                logger.exception(
                    "Error while closing session database at %s", self.db_path
                )
                raise
            finally:
                self._closed = True

        logger.info("Closed session database at %s", self.db_path)
