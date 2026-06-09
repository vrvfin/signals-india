"""
pdf_ocr.py — shared OCR fallback (Phase 3 / T1.5).

Recovers text from scanned / image-only PDFs that LOCAL text extraction
(fitz / pdfplumber / pypdf) cannot read, using Gemini's native multimodal
vision via the existing BucketPool. No external OCR engine / system dependency —
stays on the free-tier Gemini pool, consistent with the project's golden rules.

ONE function, reused across phases. Wire it wherever a pipeline extracts text
LOCALLY and would otherwise skip a sub-threshold (scanned) PDF, e.g.
`daily_research_summary.py` (Workflow A / OT8).

Note: paths that already send the raw PDF to Gemini (`extract_concall.py`,
`company_deep_report.py` via `pool.call_pdf`) get OCR for free from Gemini vision
and do NOT need this fallback. This module is import-safe (no heavy deps) so it
can be imported anywhere, including modules also imported by Streamlit.

Usage:
    from pdf_ocr import ocr_pdf_via_gemini
    text = ocr_pdf_via_gemini(pool, pdf_bytes)   # "" on failure
"""
from __future__ import annotations

# Faithful-transcription prompt — OCR only, no summarisation/interpretation.
OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL readable text from this PDF VERBATIM, "
    "in natural reading order. Preserve numbers exactly, render tables as "
    "pipe-separated rows, and keep headings and bullet points. Do NOT summarise, "
    "interpret, translate, or add any commentary. Skip blank/unreadable pages. "
    "Output only the transcribed text."
)


def ocr_pdf_via_gemini(pool, pdf_bytes: bytes, prompt: str = OCR_PROMPT) -> str:
    """Transcribe a scanned/image PDF to text using Gemini vision.

    Args:
        pool: a gemini_pool.BucketPool (must expose call_pdf(bytes, prompt) ->
              (text, model_used)). Pass the caller's existing pool so the OCR call
              uses the same quota/key policy as the rest of that pipeline.
        pdf_bytes: raw PDF bytes.
        prompt: transcription instruction (defaults to OCR_PROMPT).

    Returns:
        Transcribed text (stripped), or "" if the pool is unusable, the input is
        empty, or the call fails (quota/transient/fatal). The caller decides what
        to do with "" — typically: mark the doc needs_ocr and retry next run.
    """
    if pool is None or not pdf_bytes:
        return ""
    try:
        text, _model = pool.call_pdf(pdf_bytes, prompt)
        return (text or "").strip()
    except Exception:
        # AllBucketsExhausted / FatalCallError / anything else — degrade to "".
        # Swallowing here keeps OCR a best-effort fallback that never crashes the
        # host pipeline; an exhausted-quota doc simply stays needs_ocr for next run.
        return ""
