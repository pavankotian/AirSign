"""PDF session report generation for the AirSign Developer B track.

This module is a pure PDF-rendering utility. It accepts already-prepared,
caller-supplied structured data and renders it into a professional PDF
report using ReportLab. It never queries a database, reads
configuration, computes statistics, performs gesture analysis, touches
``ActionThread`` or any UI code, or otherwise modifies application
state -- every value that appears in the generated PDF is exactly the
value the caller passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

logger = logging.getLogger(__name__)


@dataclass
class PDFReportData:
    """Structured input for a single PDF session report.

    Every field is already-prepared, display-ready data supplied by
    the caller; this module performs no computation on any of it.

    Attributes:
        title: Report title, rendered at the top of the document.
        generated_at: Report generation timestamp, already formatted
            as a display string by the caller.
        session_summary: Ordered ``(label, value)`` pairs rendered
            under the "Session Summary" section (for example,
            ``("Session Duration", "12m 34s")``). Omit the section
            entirely by passing an empty list.
        statistics: Ordered ``(label, value)`` pairs rendered under
            the "Statistics" section (for example,
            ``("Average Confidence", "0.91")``,
            ``("Average Latency", "34.2 ms")``,
            ``("Total Events", "128")``). Omit the section entirely by
            passing an empty list.
        gesture_counts: Ordered ``(gesture_label, count)`` pairs
            rendered under the "Gesture Summary" section. Omit the
            section entirely by passing an empty list.
        event_table_headers: Column headers for the optional event
            table, or ``None`` to omit the table entirely.
        event_table_rows: Row data for the optional event table, each
            row a list of display strings aligned to
            ``event_table_headers``, or ``None`` to omit the table.
        notes: Optional free-text notes rendered at the end of the
            report, or ``None`` to omit the section.
    """

    title: str
    generated_at: str
    session_summary: list[tuple[str, str]] = field(default_factory=list)
    statistics: list[tuple[str, str]] = field(default_factory=list)
    gesture_counts: list[tuple[str, str]] = field(default_factory=list)
    event_table_headers: Optional[list[str]] = None
    event_table_rows: Optional[list[list[str]]] = None
    notes: Optional[str] = None


@dataclass
class PDFExportResult:
    """Outcome of a PDF export attempt.

    Attributes:
        success: Whether the PDF was generated and saved successfully.
        output_path: The path the PDF was written to, or ``None`` if
            generation failed before a path could be finalized.
        error_message: A human-readable description of the failure, or
            ``None`` if ``success`` is ``True``.
    """

    success: bool
    output_path: Optional[Path]
    error_message: Optional[str] = None


class PDFReportGenerator:
    """Renders a :class:`PDFReportData` instance into a PDF file.

    This class is stateless aside from its cached paragraph styles; a
    single instance may be reused to generate any number of reports.
    All document assembly is broken into small, single-purpose
    private helper methods, each responsible for exactly one section
    of the report.
    """

    def __init__(self) -> None:
        """Initializes the generator and its paragraph styles."""
        self._styles = getSampleStyleSheet()
        self._section_heading_style = ParagraphStyle(
            name="AirSignSectionHeading",
            parent=self._styles["Heading2"],
            spaceBefore=12,
            spaceAfter=6,
        )
        self._timestamp_style = ParagraphStyle(
            name="AirSignTimestamp",
            parent=self._styles["Normal"],
            textColor=colors.grey,
        )
        self._table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )

        logger.info("PDFReportGenerator initialized")

    def generate(
        self, data: PDFReportData, output_path: str | Path
    ) -> PDFExportResult:
        """Generates a PDF report from ``data`` and saves it to ``output_path``.

        Args:
            data: The structured, already-prepared report data to
                render.
            output_path: Filesystem path the PDF should be written to.
                Parent directories are created if they do not already
                exist.

        Returns:
            A :class:`PDFExportResult` describing whether generation
            succeeded, and on failure, a human-readable error message.
            This method never raises; all failures are captured and
            returned.
        """
        resolved_path = Path(output_path)

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception(
                "Failed to create output directory for PDF report: %s",
                resolved_path.parent,
            )
            return PDFExportResult(
                success=False,
                output_path=None,
                error_message=f"Could not create output directory: {exc}",
            )

        try:
            story = self._build_story(data)
            document = SimpleDocTemplate(
                str(resolved_path),
                pagesize=letter,
                title=data.title,
                topMargin=0.75 * inch,
                bottomMargin=0.75 * inch,
                leftMargin=0.75 * inch,
                rightMargin=0.75 * inch,
            )
            document.build(story)
        except OSError as exc:
            logger.exception("Filesystem error while writing PDF report: %s", resolved_path)
            return PDFExportResult(
                success=False,
                output_path=None,
                error_message=f"Filesystem error while writing PDF: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - ReportLab raises varied exception types
            logger.exception("ReportLab failed to generate PDF report: %s", resolved_path)
            return PDFExportResult(
                success=False,
                output_path=None,
                error_message=f"Failed to generate PDF: {exc}",
            )

        logger.info("PDF report generated at %s", resolved_path)
        return PDFExportResult(success=True, output_path=resolved_path, error_message=None)

    def _build_story(self, data: PDFReportData) -> list[Flowable]:
        """Assembles the full flowable story for the report.

        Args:
            data: The structured report data to render.

        Returns:
            An ordered list of ReportLab flowables ready to pass to
            ``SimpleDocTemplate.build``.
        """
        story: list[Flowable] = []
        story.extend(self._build_header(data))
        story.extend(self._build_summary(data))
        story.extend(self._build_statistics(data))
        story.extend(self._build_gesture_summary(data))
        story.extend(self._build_table(data))
        story.extend(self._build_notes(data))
        return story

    def _build_header(self, data: PDFReportData) -> list[Flowable]:
        """Builds the title and generation-timestamp header.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the header section.
        """
        return [
            Paragraph(data.title, self._styles["Title"]),
            Paragraph(f"Generated: {data.generated_at}", self._timestamp_style),
            Spacer(1, 0.25 * inch),
        ]

    def _build_summary(self, data: PDFReportData) -> list[Flowable]:
        """Builds the "Session Summary" section.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the summary section, or an empty
            list if ``data.session_summary`` is empty.
        """
        if not data.session_summary:
            return []

        return [
            Paragraph("Session Summary", self._section_heading_style),
            self._build_key_value_table(data.session_summary),
            Spacer(1, 0.2 * inch),
        ]

    def _build_statistics(self, data: PDFReportData) -> list[Flowable]:
        """Builds the "Statistics" section.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the statistics section, or an
            empty list if ``data.statistics`` is empty.
        """
        if not data.statistics:
            return []

        return [
            Paragraph("Statistics", self._section_heading_style),
            self._build_key_value_table(data.statistics),
            Spacer(1, 0.2 * inch),
        ]

    def _build_gesture_summary(self, data: PDFReportData) -> list[Flowable]:
        """Builds the "Gesture Summary" section.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the gesture summary section, or
            an empty list if ``data.gesture_counts`` is empty.
        """
        if not data.gesture_counts:
            return []

        return [
            Paragraph("Gesture Summary", self._section_heading_style),
            self._build_key_value_table(data.gesture_counts, key_header="Gesture", value_header="Count"),
            Spacer(1, 0.2 * inch),
        ]

    def _build_table(self, data: PDFReportData) -> list[Flowable]:
        """Builds the optional "Event Table" section.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the event table section, or an
            empty list if either ``data.event_table_headers`` or
            ``data.event_table_rows`` is ``None``.
        """
        if data.event_table_headers is None or data.event_table_rows is None:
            return []

        table_data: list[list[str]] = [data.event_table_headers, *data.event_table_rows]
        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(self._table_style)

        return [
            Paragraph("Event Table", self._section_heading_style),
            table,
            Spacer(1, 0.2 * inch),
        ]

    def _build_notes(self, data: PDFReportData) -> list[Flowable]:
        """Builds the optional "Notes" section.

        Args:
            data: The structured report data to render.

        Returns:
            A list of flowables for the notes section, or an empty
            list if ``data.notes`` is ``None``.
        """
        if not data.notes:
            return []

        return [
            Paragraph("Notes", self._section_heading_style),
            Paragraph(data.notes, self._styles["Normal"]),
        ]

    def _build_key_value_table(
        self,
        rows: list[tuple[str, str]],
        key_header: str = "Field",
        value_header: str = "Value",
    ) -> Table:
        """Builds a two-column, header-plus-data table from label/value pairs.

        Args:
            rows: Ordered ``(label, value)`` pairs to render as table
                rows.
            key_header: Header text for the first column.
            value_header: Header text for the second column.

        Returns:
            A styled ``Table`` flowable with one header row followed
            by one row per entry in ``rows``.
        """
        table_data: list[list[str]] = [[key_header, value_header]]
        table_data.extend([label, value] for label, value in rows)

        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(self._table_style)
        return table


def export_session_report(
    data: PDFReportData, output_path: str | Path
) -> PDFExportResult:
    """Generates and saves a PDF session report.

    Convenience wrapper around :class:`PDFReportGenerator` for
    callers that do not need to reuse a generator instance across
    multiple exports.

    Args:
        data: The structured, already-prepared report data to render.
        output_path: Filesystem path the PDF should be written to.

    Returns:
        A :class:`PDFExportResult` describing whether generation
        succeeded.
    """
    generator = PDFReportGenerator()
    return generator.generate(data, output_path)
