#!/usr/bin/env python3
"""
MCP Server for Unredact — exposes the PDF redaction analysis pipeline
as tools that Claude Code (or any MCP client) can call.

Tools provided:
  Service management:  service_status, service_start, service_stop
  PDF pipeline:        upload_pdf, run_ocr, detect_redactions, get_page_data
  Solving:             solve_redaction, get_solve_results, validate_results
  Fonts:               list_fonts
  Data:                list_associates, search_associates
  Dev:                 run_tests, build_solver, build_word_lists, view_logs
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
PID_DIR = PROJECT_ROOT / ".pids"
MAKEFILE = PROJECT_ROOT / "Makefile"
DATA_DIR = PROJECT_ROOT / "unredact" / "data"

APP_PORT = int(os.environ.get("APP_PORT", "8000"))
SOLVER_PORT = int(os.environ.get("SOLVER_PORT", "3100"))
APP_URL = f"http://127.0.0.1:{APP_PORT}"
SOLVER_URL = f"http://127.0.0.1:{SOLVER_PORT}"

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

mcp = FastMCP(
    "unredact",
    instructions=(
        "MCP server for the Unredact PDF redaction analysis tool. "
        "Use these tools to manage the services, upload and analyze PDFs, "
        "solve redactions, validate results, and run dev tasks."
    ),
)

# ── Helpers ────────────────────────────────────────────────────────


def _pid_file(name: str) -> Path:
    return PID_DIR / f"{name}.pid"


def _log_file(name: str) -> Path:
    return PID_DIR / f"{name}.log"


def _is_running(name: str) -> tuple[bool, int | None]:
    pf = _pid_file(name)
    if not pf.exists():
        return False, None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (OSError, ValueError):
        return False, None


async def _api(method: str, path: str, **kwargs) -> dict:
    """Make an HTTP request to the Unredact FastAPI server."""
    async with httpx.AsyncClient(base_url=APP_URL, timeout=120) as client:
        resp = await getattr(client, method)(path, **kwargs)
        resp.raise_for_status()
        return resp.json()


async def _consume_sse(method: str, path: str, **kwargs) -> list[dict]:
    """Consume an SSE endpoint and return all parsed events."""
    events = []
    async with httpx.AsyncClient(base_url=APP_URL, timeout=None) as client:
        req = client.build_request(method.upper(), path, **kwargs)
        async with client.send(req, stream=True) as resp:
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        events.append(json.loads(line[6:]))
                    except (json.JSONDecodeError, ValueError):
                        pass
    return events


def _run_make(target: str, timeout: int = 300) -> str:
    """Run a Makefile target and return combined output."""
    result = subprocess.run(
        ["make", "-C", str(PROJECT_ROOT), target],
        capture_output=True, text=True, timeout=timeout,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0:
        return f"FAILED (exit {result.returncode}):\n{out}"
    return out


# ── Service Management Tools ──────────────────────────────────────


@mcp.tool()
def service_status() -> str:
    """Check whether the Unredact app and Rust solver are running."""
    app_running, app_pid = _is_running("app")
    solver_running, solver_pid = _is_running("solver")
    lines = []
    lines.append(f"App:    {'running' if app_running else 'stopped'}"
                 + (f" (pid {app_pid})" if app_pid else ""))
    lines.append(f"Solver: {'running' if solver_running else 'stopped'}"
                 + (f" (pid {solver_pid})" if solver_pid else ""))
    return "\n".join(lines)


@mcp.tool()
def service_start(service: str = "all") -> str:
    """Start the Unredact services.

    Args:
        service: Which service to start — "app", "solver", or "all" (default).
                 Starting "app" also starts the solver automatically.
    """
    target = {"app": "app", "solver": "solver", "all": "app"}.get(service, "app")
    return _run_make(target, timeout=120)


@mcp.tool()
def service_stop(service: str = "all") -> str:
    """Stop the Unredact services.

    Args:
        service: Which service to stop — "app", "solver", or "all" (default).
    """
    target = {
        "app": "stop-app",
        "solver": "stop-solver",
        "all": "stop",
    }.get(service, "stop")
    return _run_make(target, timeout=30)


@mcp.tool()
def view_logs(service: str = "app", lines: int = 50) -> str:
    """View recent log output from a service.

    Args:
        service: "app" or "solver".
        lines: Number of trailing lines to return (default 50).
    """
    log = _log_file(service)
    if not log.exists():
        return f"No log file found at {log}"
    content = log.read_text()
    tail = content.strip().split("\n")[-lines:]
    return "\n".join(tail)


# ── PDF Pipeline Tools ────────────────────────────────────────────


@mcp.tool()
async def upload_pdf(file_path: str) -> str:
    """Upload a PDF file to Unredact for analysis.

    Args:
        file_path: Absolute path to a PDF file on disk.

    Returns:
        JSON with doc_id and page_count.
    """
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    async with httpx.AsyncClient(base_url=APP_URL, timeout=120) as client:
        resp = await client.post(
            "/api/upload",
            files={"file": (p.name, p.read_bytes(), "application/pdf")},
        )
        resp.raise_for_status()
        data = resp.json()
    return json.dumps(data, indent=2)


@mcp.tool()
async def run_ocr(doc_id: str) -> str:
    """Run OCR on all pages of an uploaded document.

    Args:
        doc_id: The document ID returned by upload_pdf.

    Returns:
        Summary of OCR results per page.
    """
    events = await _consume_sse("GET", f"/api/doc/{doc_id}/ocr")
    summary = []
    for e in events:
        ev = e.get("event", e.get("status", ""))
        if ev == "page_ocr_complete":
            summary.append(f"Page {e['page']}: {e['num_lines']} lines")
        elif ev == "error":
            summary.append(f"Page {e.get('page', '?')}: ERROR — {e.get('message', '')}")
    return "\n".join(summary) if summary else "OCR complete (no page details)."


@mcp.tool()
async def detect_redactions(doc_id: str) -> str:
    """Detect redactions on all pages (runs OCR if needed, then font detection + redaction detection).

    Args:
        doc_id: The document ID returned by upload_pdf.

    Returns:
        Summary of redactions found per page.
    """
    events = await _consume_sse("GET", f"/api/doc/{doc_id}/analyze")
    summary = []
    for e in events:
        ev = e.get("event", e.get("status", ""))
        if ev == "page_complete":
            count = e.get("redaction_count", 0)
            summary.append(f"Page {e['page']}: {count} redaction(s) found")
        elif ev == "error":
            summary.append(f"Page {e.get('page', '?')}: ERROR — {e.get('message', '')}")
    return "\n".join(summary) if summary else "Analysis complete."


@mcp.tool()
async def get_page_data(doc_id: str, page: int) -> str:
    """Get detailed redaction data for a specific page (font, position, context).

    Args:
        doc_id: The document ID.
        page: Page number (1-indexed).

    Returns:
        JSON array of redactions with analysis details.
    """
    data = await _api("get", f"/api/doc/{doc_id}/page/{page}/data")
    return json.dumps(data, indent=2)


# ── Solving Tools ─────────────────────────────────────────────────


@mcp.tool()
async def solve_redaction(
    font_id: str,
    font_size: int,
    gap_width_px: float,
    mode: str = "name",
    tolerance_px: float = 0.0,
    left_context: str = "",
    right_context: str = "",
    known_start: str = "",
    known_end: str = "",
    charset: str = "lowercase",
    word_filter: str = "none",
    ensure_plural: bool = False,
    vocab_size: int = 0,
) -> str:
    """Run the constraint solver to find text that fits a redaction's pixel width.

    Args:
        font_id: Font identifier (e.g. "times-new-roman"). Use list_fonts to see options.
        font_size: Font size in points.
        gap_width_px: Width of the redacted gap in pixels.
        mode: Solving mode — "name", "full_name", "email", "word", or "enumerate".
        tolerance_px: Allowed width tolerance in pixels (default 0).
        left_context: Text immediately before the redaction.
        right_context: Text immediately after the redaction.
        known_start: Known starting characters of the hidden text.
        known_end: Known ending characters of the hidden text.
        charset: Character set — "lowercase", "uppercase", "alpha", "capitalized".
        word_filter: For enumerate mode: "none", "words", "nouns" (default "none").
        ensure_plural: If True, only return plural words (word mode).
        vocab_size: Limit to top N most common words (0 = no limit).

    Returns:
        JSON with matches (text, width_px, error_px) and summary.
    """
    payload = {
        "font_id": font_id,
        "font_size": font_size,
        "gap_width_px": gap_width_px,
        "tolerance_px": tolerance_px,
        "left_context": left_context,
        "right_context": right_context,
        "mode": mode,
        "hints": {"charset": charset},
        "known_start": known_start,
        "known_end": known_end,
        "word_filter": word_filter,
        "ensure_plural": ensure_plural,
        "vocab_size": vocab_size,
    }
    events = await _consume_sse("POST", "/api/solve", json=payload)

    matches = []
    solve_id = None
    total = 0
    for e in events:
        status = e.get("status", "")
        if status == "match":
            matches.append({
                "text": e["text"],
                "width_px": e["width_px"],
                "error_px": e["error_px"],
                "source": e.get("source", ""),
            })
        elif status == "done":
            solve_id = e.get("solve_id")
            total = e.get("total_found", len(matches))
        elif status == "page_complete":
            solve_id = e.get("solve_id")

    result = {
        "solve_id": solve_id,
        "matches_shown": len(matches),
        "total_found": total,
        "matches": matches[:50],  # Cap at 50 to keep context manageable
    }
    if total > 50:
        result["note"] = f"Showing first 50 of {total} matches. Use get_solve_results for more."
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_solve_results(solve_id: str, offset: int = 0, limit: int = 200) -> str:
    """Fetch paginated solve results.

    Args:
        solve_id: The solve ID from a previous solve_redaction call.
        offset: Starting index (default 0).
        limit: Max results to return (default 200).

    Returns:
        JSON with results array and pagination info.
    """
    data = await _api("get", f"/api/solve/{solve_id}/results",
                      params={"offset": offset, "limit": limit})
    return json.dumps(data, indent=2)


@mcp.tool()
async def validate_results(
    solve_id: str,
    left_context: str = "",
    right_context: str = "",
) -> str:
    """Run Claude LLM validation to score solve candidates by contextual fit.

    Args:
        solve_id: The solve ID from a previous solve_redaction call.
        left_context: Text before the redaction (for context).
        right_context: Text after the redaction (for context).

    Returns:
        Scored results sorted by LLM score (highest first).
    """
    events = await _consume_sse(
        "POST", f"/api/solve/{solve_id}/validate",
        json={"left_context": left_context, "right_context": right_context},
    )

    scored = []
    for e in events:
        status = e.get("status", "")
        if status == "batch_done":
            for r in e.get("results", []):
                scored.append({
                    "text": r["text"],
                    "llm_score": r.get("llm_score", 0),
                    "width_px": r.get("width_px"),
                    "error_px": r.get("error_px"),
                })

    scored.sort(key=lambda x: x["llm_score"], reverse=True)
    result = {
        "total_scored": len(scored),
        "top_results": scored[:30],
    }
    if len(scored) > 30:
        result["note"] = f"Showing top 30 of {len(scored)} scored results."
    return json.dumps(result, indent=2)


# ── Font Tools ────────────────────────────────────────────────────


@mcp.tool()
async def list_fonts() -> str:
    """List all candidate fonts and their availability on this system.

    Returns:
        JSON array of fonts with name, id, and available status.
    """
    data = await _api("get", "/api/fonts")
    return json.dumps(data, indent=2)


# ── Data Tools ────────────────────────────────────────────────────


@mcp.tool()
async def list_associates(limit: int = 50) -> str:
    """List known associate names from the database.

    Args:
        limit: Max names to return (default 50).

    Returns:
        JSON with name entries and their variants.
    """
    data = await _api("get", "/api/associates")
    names = data.get("names", {})
    # Return a subset with counts
    entries = []
    for name, info in list(names.items())[:limit]:
        entries.append({
            "name": name,
            "variants": info.get("variants", []) if isinstance(info, dict) else [],
        })
    return json.dumps({
        "total_names": len(names),
        "shown": len(entries),
        "entries": entries,
    }, indent=2)


@mcp.tool()
async def search_associates(query: str) -> str:
    """Search associate names matching a query string.

    Args:
        query: Name or partial name to search for (case-insensitive).

    Returns:
        Matching associate entries.
    """
    data = await _api("get", "/api/associates")
    names = data.get("names", {})
    q = query.lower()
    matches = []
    for name, info in names.items():
        if q in name.lower():
            matches.append({
                "name": name,
                "info": info if isinstance(info, dict) else {"raw": info},
            })
    return json.dumps({
        "query": query,
        "count": len(matches),
        "matches": matches[:100],
    }, indent=2)


# ── Dev Tools ─────────────────────────────────────────────────────


@mcp.tool()
def run_tests(filter: str = "") -> str:
    """Run the pytest test suite.

    Args:
        filter: Optional pytest filter expression (e.g. "test_solver" or "-k test_width").
                If empty, runs all tests.

    Returns:
        Test output with pass/fail summary.
    """
    cmd = [
        str(VENV_PYTHON), "-m", "pytest", "tests/",
        "--ignore=tests/test_alignment.py", "-v",
    ]
    if filter:
        if filter.startswith("-"):
            cmd.extend(filter.split())
        else:
            cmd.extend(["-k", filter])

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=300, cwd=str(PROJECT_ROOT),
    )
    out = result.stdout + result.stderr
    # Trim to last 200 lines to avoid overwhelming context
    lines = out.strip().split("\n")
    if len(lines) > 200:
        out = f"[...truncated {len(lines) - 200} lines...]\n" + "\n".join(lines[-200:])
    return out


@mcp.tool()
def build_solver() -> str:
    """Build the Rust constraint solver (release mode).

    Returns:
        Build output or error message.
    """
    return _run_make("build-solver", timeout=300)


@mcp.tool()
def build_word_lists() -> str:
    """Rebuild the word lists (nouns, adjectives) from WordNet.

    Returns:
        Build output.
    """
    return _run_make("build-word-lists", timeout=120)


@mcp.tool()
def build_associates() -> str:
    """Rebuild the associate name lists from associates.json.

    Returns:
        Build output.
    """
    return _run_make("build-associates", timeout=60)


# ── Full Pipeline Tool (convenience) ─────────────────────────────


@mcp.tool()
async def analyze_pdf(file_path: str) -> str:
    """Full pipeline: upload a PDF, run OCR, and detect redactions in one step.

    Args:
        file_path: Absolute path to a PDF file.

    Returns:
        Complete analysis summary including all detected redactions with
        font, position, and context information.
    """
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    # Upload
    async with httpx.AsyncClient(base_url=APP_URL, timeout=120) as client:
        resp = await client.post(
            "/api/upload",
            files={"file": (p.name, p.read_bytes(), "application/pdf")},
        )
        resp.raise_for_status()
        upload = resp.json()

    doc_id = upload["doc_id"]
    page_count = upload["page_count"]

    # OCR
    ocr_events = await _consume_sse("GET", f"/api/doc/{doc_id}/ocr")

    # Analyze
    analyze_events = await _consume_sse("GET", f"/api/doc/{doc_id}/analyze")

    # Gather page data
    pages = {}
    for page_num in range(1, page_count + 1):
        data = await _api("get", f"/api/doc/{doc_id}/page/{page_num}/data")
        pages[page_num] = data

    result = {
        "doc_id": doc_id,
        "page_count": page_count,
        "pages": {},
    }
    for page_num, data in pages.items():
        redactions = data.get("redactions", [])
        result["pages"][str(page_num)] = {
            "redaction_count": len(redactions),
            "redactions": redactions,
        }

    return json.dumps(result, indent=2)


@mcp.tool()
async def unredact_document(
    file_path: str,
    output_dir: str = "",
    validate: bool = True,
) -> str:
    """Full end-to-end pipeline: upload a PDF, analyze it, solve all redactions,
    and export a Markdown file with the unredacted text plus a detailed report.

    Args:
        file_path: Absolute path to a PDF file.
        output_dir: Directory to write output files (defaults to same dir as PDF).
        validate: Whether to run LLM validation for confidence scores (default True).

    Returns:
        Paths to the generated markdown and report files, plus a summary.
    """
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    out_dir = Path(output_dir) if output_dir else p.parent

    # Step 1: Upload
    async with httpx.AsyncClient(base_url=APP_URL, timeout=120) as client:
        resp = await client.post(
            "/api/upload",
            files={"file": (p.name, p.read_bytes(), "application/pdf")},
        )
        resp.raise_for_status()
        upload = resp.json()

    doc_id = upload["doc_id"]
    page_count = upload["page_count"]

    # Step 2: OCR
    await _consume_sse("GET", f"/api/doc/{doc_id}/ocr")

    # Step 3: Analyze (detect redactions + fonts)
    await _consume_sse("GET", f"/api/doc/{doc_id}/analyze")

    # Step 4: Export (solve + validate + write files)
    export_events = await _consume_sse(
        "POST",
        f"/api/doc/{doc_id}/export",
        json={"validate": validate, "output_dir": str(out_dir)},
    )

    # Find the completion event
    for e in export_events:
        if e.get("status") == "complete":
            return json.dumps({
                "markdown_path": e["markdown_path"],
                "report_path": e["report_path"],
                "doc_id": doc_id,
                "page_count": page_count,
                "message": "Document unredacted successfully. "
                           "Check the markdown file for the full text and "
                           "the report file for detailed analysis.",
            }, indent=2)
        elif e.get("status") == "error":
            return f"Export failed: {e.get('error', 'unknown error')}"

    return "Export completed but no completion event received."


# ── Entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
