"""Export pipeline: reconstruct a full document with redactions filled in.

Produces two files:
  1. A Markdown file with the document text, redacted spans replaced by
     the best-fit candidate (annotated with confidence).
  2. A companion report file with per-redaction details: position, font,
     candidates, scores, and reasoning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import ImageFont

from unredact.pipeline.ocr import OcrChar, OcrLine
from unredact.pipeline.analyze_page import RedactionAnalysis, PageAnalysis
from unredact.pipeline.solver import SolveResult
from unredact.pipeline.dictionary import (
    solve_name_dictionary,
    solve_full_name_dictionary,
    solve_word_dictionary,
)
from unredact.pipeline.llm_validate import validate_candidates

log = logging.getLogger(__name__)

# Maximum candidates to send for LLM validation per redaction
_MAX_VALIDATE = 50


@dataclass
class ResolvedRedaction:
    """A single redaction with its best-guess fill."""

    redaction: RedactionAnalysis
    candidates: list[dict]        # [{text, width_px, error_px, source}, ...]
    best_text: str | None = None
    best_score: int = 0
    scored: bool = False          # whether LLM validation ran


@dataclass
class ExportResult:
    """Full export result for a document."""

    doc_name: str
    page_count: int
    pages: dict[int, PageExport]
    total_redactions: int = 0
    resolved_count: int = 0


@dataclass
class PageExport:
    """Export data for a single page."""

    page_num: int
    lines: list[OcrLine]
    redactions: list[ResolvedRedaction]


async def solve_single_redaction(
    ra: RedactionAnalysis,
    font: ImageFont.FreeTypeFont,
    modes: list[str] | None = None,
) -> list[dict]:
    """Run solvers on a single redaction across multiple modes.

    Returns a deduplicated list of candidate dicts sorted by error.
    """
    if modes is None:
        modes = ["name", "full_name", "word"]

    gap_w = float(ra.box.w)
    tolerance = 1.0
    left = ra.left_text
    right = ra.right_text
    seen: set[str] = set()
    candidates: list[dict] = []

    for mode in modes:
        results: list[SolveResult] = []
        if mode == "name":
            results = solve_name_dictionary(
                font, gap_w, tolerance, left, right, casing="capitalized",
            )
        elif mode == "full_name":
            results = solve_full_name_dictionary(
                font, gap_w, tolerance, left, right, casing="capitalized",
            )
        elif mode == "word":
            results = list(solve_word_dictionary(
                font, gap_w, tolerance, left, right, casing="lowercase",
            ))

        for r in results:
            if r.text not in seen:
                seen.add(r.text)
                candidates.append({
                    "text": r.text,
                    "width_px": round(r.width, 2),
                    "error_px": round(r.error, 2),
                    "source": mode,
                })

    candidates.sort(key=lambda c: c["error_px"])
    return candidates


async def resolve_page(
    analysis: PageAnalysis,
    page_image,
    page_num: int,
    validate: bool = True,
    on_progress=None,
) -> PageExport:
    """Solve and optionally validate all redactions on a page."""
    resolved: list[ResolvedRedaction] = []

    for i, ra in enumerate(analysis.redactions):
        if on_progress:
            on_progress("solving", {
                "page": page_num,
                "redaction": i + 1,
                "total": len(analysis.redactions),
            })

        font = ra.font.to_pil_font()
        candidates = await solve_single_redaction(ra, font)

        rr = ResolvedRedaction(redaction=ra, candidates=candidates)

        # LLM validation on top candidates
        if validate and candidates:
            top = candidates[:_MAX_VALIDATE]
            texts = [c["text"] for c in top]
            try:
                scores = await validate_candidates(
                    ra.left_text, ra.right_text, texts,
                )
                for j, s in enumerate(scores):
                    top[j]["llm_score"] = s
                # Pick best by LLM score
                best_idx = max(range(len(scores)), key=lambda k: scores[k])
                rr.best_text = top[best_idx]["text"]
                rr.best_score = scores[best_idx]
                rr.scored = True
            except Exception as exc:
                log.warning("LLM validation failed for redaction %d: %s", i, exc)
                # Fall back to lowest error
                if candidates:
                    rr.best_text = candidates[0]["text"]
                    rr.best_score = 0
        elif candidates:
            rr.best_text = candidates[0]["text"]
            rr.best_score = 0

        resolved.append(rr)

    return PageExport(
        page_num=page_num,
        lines=analysis.lines,
        redactions=resolved,
    )


def _reconstruct_line(
    line: OcrLine,
    redactions: list[ResolvedRedaction],
) -> str:
    """Rebuild a line's text, splicing in resolved redaction text."""
    chars = list(line.chars)
    if not chars:
        return ""

    # Build intervals for redactions on this line (by x-position)
    insertions: list[tuple[int, int, str]] = []
    for rr in redactions:
        box = rr.redaction.box
        # Check vertical overlap with this line
        line_top = line.y
        line_bot = line.y + line.h
        box_top = box.y
        box_bot = box.y + box.h
        overlap = max(0, min(line_bot, box_bot) - max(line_top, box_top))
        if overlap < line.h * 0.3:
            continue

        fill = rr.best_text or "[???]"
        score = rr.best_score
        insertions.append((box.x, box.x + box.w, fill, score))

    if not insertions:
        return line.text

    insertions.sort(key=lambda t: t[0])

    # Walk through chars, replacing spans that fall inside redaction boxes
    parts: list[str] = []
    ins_idx = 0
    char_idx = 0

    while char_idx < len(chars):
        c = chars[char_idx]
        cx = c.x + c.w / 2  # char center

        # Check if this char falls inside the next insertion
        if ins_idx < len(insertions):
            rx_start, rx_end, fill, score = insertions[ins_idx]
            if cx >= rx_start and cx <= rx_end:
                # Skip all chars inside this redaction box
                parts.append(f"**[{fill}]** _(confidence: {score}%)_")
                while char_idx < len(chars) and chars[char_idx].x + chars[char_idx].w / 2 <= rx_end:
                    char_idx += 1
                ins_idx += 1
                continue
            elif cx > rx_end:
                # We passed this insertion without entering it — insert it
                parts.append(f"**[{fill}]** _(confidence: {score}%)_")
                ins_idx += 1
                continue

        parts.append(c.text)
        char_idx += 1

    # Append any remaining insertions
    while ins_idx < len(insertions):
        _, _, fill, score = insertions[ins_idx]
        parts.append(f" **[{fill}]** _(confidence: {score}%)_")
        ins_idx += 1

    return "".join(parts)


def generate_markdown(export: ExportResult) -> str:
    """Generate the Markdown document with redactions filled in."""
    lines: list[str] = []
    lines.append(f"# {export.doc_name}\n")
    lines.append(f"*Unredacted by Unredact — "
                 f"{export.resolved_count}/{export.total_redactions} "
                 f"redactions resolved*\n")

    for page_num in sorted(export.pages.keys()):
        page = export.pages[page_num]
        if export.page_count > 1:
            lines.append(f"\n---\n\n## Page {page_num}\n")

        # Build a lookup of redactions per line (by y-overlap)
        for ocr_line in page.lines:
            text = _reconstruct_line(ocr_line, page.redactions)
            if text.strip():
                lines.append(text)

        lines.append("")  # blank line after page

    return "\n".join(lines)


def generate_report(export: ExportResult) -> str:
    """Generate the detailed report file."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Unredaction Report: {export.doc_name}\n")
    lines.append(f"- **Generated**: {now}")
    lines.append(f"- **Pages**: {export.page_count}")
    lines.append(f"- **Total redactions detected**: {export.total_redactions}")
    lines.append(f"- **Successfully resolved**: {export.resolved_count}")
    if export.total_redactions > 0:
        pct = round(export.resolved_count / export.total_redactions * 100)
        lines.append(f"- **Resolution rate**: {pct}%")
    lines.append("")

    redaction_num = 0
    for page_num in sorted(export.pages.keys()):
        page = export.pages[page_num]
        if not page.redactions:
            continue

        lines.append(f"\n## Page {page_num}\n")

        for rr in page.redactions:
            redaction_num += 1
            ra = rr.redaction
            box = ra.box

            lines.append(f"### Redaction {redaction_num}\n")
            lines.append(f"| Property | Value |")
            lines.append(f"|----------|-------|")
            lines.append(f"| Position | ({box.x}, {box.y}) |")
            lines.append(f"| Size | {box.w} x {box.h} px |")
            lines.append(f"| Font | {ra.font.font_name} {ra.font.font_size}pt "
                         f"(match score: {ra.font.score:.2f}) |")
            lines.append(f"| Left context | `{ra.left_text}` |")
            lines.append(f"| Right context | `{ra.right_text}` |")
            lines.append(f"| Gap width | {box.w} px |")

            if rr.best_text:
                lines.append(f"| **Best match** | **{rr.best_text}** |")
                lines.append(f"| **Confidence** | **{rr.best_score}%** |")
            else:
                lines.append(f"| Best match | _(none found)_ |")

            lines.append("")

            if rr.candidates:
                # Show top 10 candidates
                top = sorted(
                    rr.candidates[:_MAX_VALIDATE],
                    key=lambda c: c.get("llm_score", 0),
                    reverse=True,
                )[:10]

                lines.append("**Top candidates:**\n")
                lines.append("| Rank | Text | Width (px) | Error (px) | Source | LLM Score |")
                lines.append("|------|------|-----------|-----------|--------|-----------|")
                for rank, c in enumerate(top, 1):
                    score_str = str(c.get("llm_score", "—"))
                    lines.append(
                        f"| {rank} | {c['text']} | {c['width_px']} | "
                        f"{c['error_px']} | {c['source']} | {score_str} |"
                    )
                lines.append("")

                if len(rr.candidates) > 10:
                    lines.append(
                        f"*...and {len(rr.candidates) - 10} more candidates*\n"
                    )
            else:
                lines.append("*No candidates found for this redaction.*\n")

    return "\n".join(lines)


async def export_document(
    doc_name: str,
    pages: dict[int, tuple[PageAnalysis, object]],
    output_dir: Path,
    validate: bool = True,
    on_progress=None,
) -> tuple[Path, Path]:
    """Run the full export pipeline.

    Args:
        doc_name: Original filename (e.g. "document.pdf").
        pages: Mapping of page_num -> (PageAnalysis, page_image).
        output_dir: Directory to write output files.
        validate: Whether to run LLM validation (default True).
        on_progress: Optional callback(event, data).

    Returns:
        Tuple of (markdown_path, report_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(doc_name).stem
    md_path = output_dir / f"{stem}_unredacted.md"
    report_path = output_dir / f"{stem}_report.md"

    page_exports: dict[int, PageExport] = {}
    total_redactions = 0
    resolved = 0

    for page_num in sorted(pages.keys()):
        analysis, page_image = pages[page_num]
        page_export = await resolve_page(
            analysis, page_image, page_num,
            validate=validate,
            on_progress=on_progress,
        )
        page_exports[page_num] = page_export
        total_redactions += len(page_export.redactions)
        resolved += sum(1 for rr in page_export.redactions if rr.best_text)

    export = ExportResult(
        doc_name=doc_name,
        page_count=len(pages),
        pages=page_exports,
        total_redactions=total_redactions,
        resolved_count=resolved,
    )

    md_content = generate_markdown(export)
    report_content = generate_report(export)

    md_path.write_text(md_content)
    report_path.write_text(report_content)

    if on_progress:
        on_progress("export_complete", {
            "markdown": str(md_path),
            "report": str(report_path),
            "total_redactions": total_redactions,
            "resolved": resolved,
        })

    return md_path, report_path
