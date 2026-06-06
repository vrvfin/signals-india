"""
llm_summariser.py — Generate 200-word company intelligence summaries using Gemini API.

Uses DAILY_GEMINI_KEY_1 and DAILY_GEMINI_KEY_2 from .env — dedicated keys for
PF tracking. IMPORTANT: keys must be from DIFFERENT Google Cloud projects to
have independent daily quotas. Same project = shared quota = both fail together.

Originally:
PF tracking, separate from the Phase 1/2 extraction keys.

Key rotation: round-robin across both keys; on 429 immediately tries the other key;
              if both exhausted logs a warning and returns a fallback string.
Inter-call sleep: 6s (matches project RPM protection from _extractor_base.py).
Model: gemini-2.5-flash (free tier, consistent with rest of project).

Public API:
    summarise_company(company_data: dict) -> str
    build_summary_section(...) -> str
    update_drive_company_page(drive, root_folder_id, isin, section) -> bool
    write_datewise_summary(drive, root_folder_id, summaries, local_dir) -> Path
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# gemini-2.5-flash free tier = 20 RPD (too low for 50+ companies).
# gemini-2.0-flash free tier = 200 RPD — sufficient for daily PF run.
# gemini-2.5-flash-lite is another option but availability varies by region.
_MODEL         = "gemini-2.0-flash"
_MODEL_FALLBACK = "gemini-2.0-flash-lite"   # if 2.0-flash also quota-hits
_INTER_SLEEP   = 3      # seconds between calls — RPM protection
_MAX_TOKENS    = 512    # ~200 words with some headroom
_MAX_RETRY_WAIT = 60    # max seconds to wait on quota retry_delay hint

_SYSTEM_PROMPT = (
    "You are a senior Indian equity research analyst. "
    "Write concise, information-dense portfolio intelligence updates. "
    "Be factual. Use numbers where available. Flag risks clearly. "
    "Write in crisp English. No fluff. No disclaimers."
)

_SUMMARY_PROMPT = """Write a 200-word intelligence update for {company_name} ({symbol}).

You have been given actual article text and YouTube transcripts below. Read them carefully and extract the key investment-relevant facts.

DATA:
{data_block}

Cover in order:
1. Latest quarter financials (use exact numbers from data — Revenue, PAT, EPS + YoY%)
2. Key insights from news articles (quote specific numbers/facts mentioned, cite source)
3. Key insights from YouTube (if transcript available, extract analyst views/targets)
4. Signal status (strategies flagging, buy/sell zone)
5. One-line risk or watchpoint

Format: flowing paragraph, no bullet points, no markdown headers.
Be specific — use numbers from the articles, not vague summaries.
If no news/YouTube available, state that clearly and focus on financials.
End with: "Risk/Watch: <one line>"
""".strip()


# ── Key pool (round-robin, 429-aware) ─────────────────────────────────────────

class _DailyKeyPool:
    """
    Rotates between DAILY_GEMINI_KEY_1 and DAILY_GEMINI_KEY_2.
    On 429 from one key, immediately switches to the other.
    Mirrors the GeminiKeyPool pattern in _extractor_base.py.
    """

    def __init__(self) -> None:
        keys = []
        for env_var in ("DAILY_GEMINI_KEY_1", "DAILY_GEMINI_KEY_2"):
            k = os.environ.get(env_var, "").strip()
            if k:
                keys.append(k)
        if not keys:
            raise RuntimeError(
                "Neither DAILY_GEMINI_KEY_1 nor DAILY_GEMINI_KEY_2 found in .env"
            )
        self._keys            = keys
        self._idx             = 0
        self._models: dict[str, object] = {}
        self._daily_exhausted = False   # set True when RPD quota hits

    def _get_client(self, key: str):
        """Return a google-genai client for the given key."""
        if key not in self._models:
            try:
                from google import genai as _genai
                self._models[key] = _genai.Client(api_key=key)
            except ImportError:
                # Fallback to deprecated google.generativeai
                import google.generativeai as _old_genai
                _old_genai.configure(api_key=key)
                self._models[key] = _old_genai.GenerativeModel(
                    model_name=_MODEL,
                    system_instruction=_SYSTEM_PROMPT,
                )
                self._models[key]._legacy = True
        return self._models[key]

    def _is_daily_quota_exhausted(self, err_str: str) -> bool:
        """True if the error is a daily (RPD) quota, not just RPM throttle."""
        return ("generaterequestsperday" in err_str.lower() or
                "limit: 0" in err_str.lower() or
                "per_day" in err_str.lower())

    def _retry_delay_seconds(self, err_str: str) -> int:
        import re
        m = re.search(r"retry[_\s]delay[^\d]*(\d+)", err_str, re.IGNORECASE)
        if m:
            return min(int(m.group(1)) + 2, _MAX_RETRY_WAIT)
        return 5

    def generate(self, prompt: str) -> str:
        """
        Call Gemini with round-robin key rotation.

        Key insight: if BOTH keys are on the SAME Google Cloud project they share
        the daily quota — switching keys won't help. We detect daily (RPD) exhaustion
        immediately and stop retrying to avoid burning minutes on 60s waits per company.

        RPM (per-minute) throttle → wait retry_delay and try next key.
        RPD (per-day) exhaustion → set flag, return fallback immediately.
        """
        if self._daily_exhausted:
            raise RuntimeError("Daily Gemini quota exhausted — skipping LLM for this run")

        last_err = None

        for model_name in (_MODEL, _MODEL_FALLBACK):
            for key_attempt in range(len(self._keys)):
                key = self._keys[self._idx % len(self._keys)]
                try:
                    client = self._get_client(key)

                    # google-genai SDK path
                    if not getattr(client, "_legacy", False):
                        from google import genai as _genai
                        from google.genai import types as _types
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=_types.GenerateContentConfig(
                                system_instruction=_SYSTEM_PROMPT,
                                max_output_tokens=_MAX_TOKENS,
                            ),
                        )
                        text = response.text
                    else:
                        # Legacy fallback
                        response = client.generate_content(prompt)
                        text = response.text

                    self._idx = (self._idx + 1) % len(self._keys)
                    time.sleep(_INTER_SLEEP)
                    return text.strip()

                except Exception as e:
                    err_str = str(e)
                    last_err = e
                    is_quota = ("429" in err_str or "quota" in err_str.lower() or
                                "resource_exhausted" in err_str.lower())

                    if not is_quota:
                        raise  # non-quota error — propagate

                    # Daily quota exhausted — no point trying anything else today
                    if self._is_daily_quota_exhausted(err_str):
                        self._daily_exhausted = True
                        log.warning(
                            f"Daily Gemini quota exhausted (RPD limit=0). "
                            f"NOTE: if both keys are on the same Google Cloud project "
                            f"they share the same quota. Use keys from DIFFERENT projects "
                            f"for independent quotas. Skipping LLM for remaining companies."
                        )
                        raise RuntimeError(
                            "Daily quota exhausted — skipping remaining LLM summaries"
                        )

                    # RPM throttle — wait and try next key
                    wait = self._retry_delay_seconds(err_str)
                    log.warning(
                        f"Key {self._idx % len(self._keys) + 1} / {model_name} "
                        f"RPM throttled — rotating key (wait {wait}s)"
                    )
                    self._idx = (self._idx + 1) % len(self._keys)
                    time.sleep(wait)

        raise RuntimeError(
            f"All Gemini key/model combos failed. Last error: {last_err}"
        )


# Module-level singleton — created once per process
_key_pool: Optional[_DailyKeyPool] = None


def _pool() -> _DailyKeyPool:
    global _key_pool
    if _key_pool is None:
        _key_pool = _DailyKeyPool()
    return _key_pool


# ── Summary generation ────────────────────────────────────────────────────────

def _build_data_block(company_data: dict) -> str:
    """
    Build the data block sent to Gemini.
    Includes: financials, signals, full article text, YouTube transcripts.
    News/YouTube items passed as dicts with keys:
      news:    [{title, source, url, full_text, snippet}]
      youtube: [{title, channel, url, transcript, description}]
    """
    fin  = company_data.get("financials", {})
    sigs = company_data.get("signals", [])
    news = company_data.get("news", [])[:3]
    yt   = company_data.get("youtube", [])[:3]

    lines = []

    # Financials
    if fin:
        q = fin.get("latest_quarter", "latest quarter")
        lines.append(f"FINANCIALS ({q}):")
        for m in ["Revenue", "EBITDA", "PAT", "EPS"]:
            val = fin.get(m)
            yoy = fin.get(f"{m}_YoY")
            if val is not None:
                yoy_str = f" ({yoy:+.1f}% YoY)" if yoy is not None else ""
                lines.append(f"  {m}: {val:,.1f} Cr{yoy_str}")
    else:
        lines.append("FINANCIALS: Not available")

    # GF4 + guidance
    gf4 = company_data.get("guidance_score")
    ag  = company_data.get("active_guidance", 0)
    if gf4 is not None:
        lines.append(f"GF4 Quality: {gf4}/10 | Active Guidance: {ag} items")

    # Signals
    if sigs:
        sig_str = ", ".join(
            f"{s.get('strategy','')}:{s.get('zone_type','')}" for s in sigs[:4]
        )
        lines.append(f"SIGNALS TODAY: {len(sigs)} — {sig_str}")
    else:
        lines.append("SIGNALS TODAY: None")

    # News — include full article text where available
    if news:
        lines.append("\nNEWS (last 24h):")
        for i, n in enumerate(news, 1):
            lines.append(f"\n  Article {i}: [{n.get('source','')}] {n.get('title','')}")
            lines.append(f"  URL: {n.get('url','')}")
            text = n.get("full_text", "") or n.get("snippet", "")
            if text:
                # Truncate per article to keep prompt manageable
                lines.append(f"  Content: {text[:800]}{'...' if len(text)>800 else ''}")
    else:
        lines.append("\nNEWS (last 24h): None from tracked sources")

    # YouTube — include transcript where available, else description + link
    if yt:
        lines.append("\nYOUTUBE (last 24h):")
        for i, v in enumerate(yt, 1):
            lines.append(f"\n  Video {i}: [{v.get('channel','')}] {v.get('title','')}")
            lines.append(f"  URL: {v.get('url','')}")
            transcript = v.get("transcript", "")
            desc       = v.get("description", "")
            if transcript:
                lines.append(f"  Transcript: {transcript[:800]}{'...' if len(transcript)>800 else ''}")
            elif desc:
                lines.append(f"  Description: {desc[:400]}")
            else:
                lines.append("  [No transcript available — link included in Excel]")
    else:
        lines.append("\nYOUTUBE (last 24h): None")

    return "\n".join(lines)


def summarise_company(company_data: dict) -> str:
    """
    Generate a ~200-word Gemini intelligence summary for one company.
    Uses DAILY_GEMINI_KEY_1 / DAILY_GEMINI_KEY_2 with round-robin + 429 fallback.
    Returns the summary string, or a descriptive fallback on failure.
    """
    try:
        pool = _pool()
    except RuntimeError as e:
        return f"[LLM summary unavailable — {e}]"

    prompt = _SUMMARY_PROMPT.format(
        company_name=company_data.get("company_name", ""),
        symbol=company_data.get("symbol", ""),
        data_block=_build_data_block(company_data),
    )

    try:
        return pool.generate(prompt)
    except RuntimeError as e:
        log.warning(f"All keys failed for {company_data.get('symbol')}: {e}")
        return f"[LLM summary unavailable — quota exhausted on both keys]"
    except Exception as e:
        log.warning(f"Gemini error for {company_data.get('symbol')}: {e}")
        return f"[LLM summary unavailable: {e}]"


# ── Drive helpers ─────────────────────────────────────────────────────────────

def _find_subfolder(drive, parent_id: str, name: str) -> Optional[str]:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    r = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return r[0]["id"] if r else None


def _get_or_create_folder(drive, parent_id: str, name: str) -> str:
    existing = _find_subfolder(drive, parent_id, name)
    if existing:
        return existing
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def _find_file(drive, folder_id: str, filename: str) -> Optional[str]:
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def _download_text(drive, file_id: str) -> str:
    from googleapiclient.http import MediaIoBaseDownload
    req  = drive.files().get_media(fileId=file_id)
    fh   = io.BytesIO()
    dl   = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue().decode("utf-8", errors="replace")


def _upload_or_update_text(drive, folder_id: str, filename: str, content: str) -> str:
    from googleapiclient.http import MediaIoBaseUpload
    encoded = content.encode("utf-8")
    media   = MediaIoBaseUpload(io.BytesIO(encoded), mimetype="text/plain", resumable=False)
    existing_id = _find_file(drive, folder_id, filename)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    f = drive.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]


# ── Markdown section builder ──────────────────────────────────────────────────

def build_summary_section(
    company_name: str,
    symbol: str,
    summary_text: str,
    news: list[dict],
    youtube: list[dict],
    date_str: Optional[str] = None,
) -> str:
    """Build the dated markdown block appended to company_page.md."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    news_lines = "\n".join(
        f"  - [{n.get('source','')}] [{n.get('title','')}]({n.get('url','')})"
        for n in news[:5]
    ) or "  - No news in last 24h from tracked sources"

    yt_lines = "\n".join(
        f"  - [{v.get('channel','')}] [{v.get('title','')}]({v.get('url','')})"
        for v in youtube[:3]
    ) or "  - No YouTube videos in last 24h"

    return f"""
---
## PF Intelligence Update — {date_str}

**Summary:**
{summary_text}

**News (last 24h):**
{news_lines}

**YouTube (last 24h):**
{yt_lines}
""".strip()


# ── Update company_page.md on Drive ──────────────────────────────────────────

def update_drive_company_page(
    drive,
    root_folder_id: str,
    isin: str,
    section: str,
) -> bool:
    """Append a dated intelligence section to company_repo/<ISIN>/company_page.md."""
    try:
        repo_id = _find_subfolder(drive, root_folder_id, "company_repo")
        if not repo_id:
            log.warning("company_repo folder not found on Drive")
            return False
        comp_id = _find_subfolder(drive, repo_id, isin.upper())
        if not comp_id:
            log.warning(f"No folder for ISIN {isin} in company_repo")
            return False
        page_id  = _find_file(drive, comp_id, "company_page.md")
        existing = _download_text(drive, page_id) if page_id else f"# {isin}\n"
        updated  = existing.rstrip() + "\n\n" + section + "\n"
        _upload_or_update_text(drive, comp_id, "company_page.md", updated)
        log.info(f"Updated company_page.md for {isin}")
        return True
    except Exception as e:
        log.warning(f"Failed to update company_page.md for {isin}: {e}")
        return False


# ── Datewise summary ──────────────────────────────────────────────────────────

def write_datewise_summary(
    drive,
    root_folder_id: str,
    summaries: list[dict],
    local_dir: Path,
) -> Path:
    """
    Build datewise_DDMMYYYY.md, upload to Drive daywisesummary/, save locally.
    Returns the local Path (for Obsidian + os.startfile).
    """
    today    = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    fname    = f"datewise_{today.strftime('%d%m%Y')}.md"

    lines = [
        f"# Portfolio Daily Intelligence — {date_str}",
        f"_Generated: {today.strftime('%d %b %Y %H:%M IST')}_",
        f"_Companies covered: {len(summaries)}_",
        "",
        "---",
        "",
    ]

    for s in summaries:
        symbol = s.get("symbol", "")
        name   = s.get("company_name", symbol)
        isin   = s.get("isin", "")
        text   = s.get("summary_text", "")
        news   = s.get("news", [])
        yt     = s.get("youtube", [])
        fin    = s.get("financials", {})
        sigs   = s.get("signals", [])

        lines += [f"## {symbol} — {name}", f"_ISIN: {isin}_", ""]

        # Financials snapshot
        if fin:
            q     = fin.get("latest_quarter", "")
            parts = []
            for m in ["Revenue", "PAT", "EPS"]:
                v   = fin.get(m)
                yoy = fin.get(f"{m}_YoY")
                if v is not None:
                    yoy_str = f" ({yoy:+.1f}% YoY)" if yoy is not None else ""
                    parts.append(f"{m}: {v:,.1f}{yoy_str}")
            if parts:
                lines += [f"**{q} Financials:** " + " | ".join(parts), ""]

        # Signals
        if sigs:
            sig_str = ", ".join(
                f"{s2.get('strategy','')} [{s2.get('zone_type','')}]" for s2 in sigs[:4]
            )
            lines += [f"**Signals:** {sig_str}", ""]

        # LLM summary
        if text:
            lines += ["**Summary:**", text, ""]

        # News
        if news:
            lines += ["**News (last 24h):**"]
            for n in news[:4]:
                lines += [f"- [{n.get('source','')}] [{n.get('title','')}]({n.get('url','')})"]
            lines += [""]

        # YouTube
        if yt:
            lines += ["**YouTube (last 24h):**"]
            for v in yt[:3]:
                lines += [f"- [{v.get('channel','')}] [{v.get('title','')}]({v.get('url','')})"]
            lines += [""]

        lines += ["---", ""]

    md_content = "\n".join(lines)

    # Upload to Drive — retry up to 3 times on transient connection errors
    for attempt in range(3):
        try:
            dw_id = _get_or_create_folder(drive, root_folder_id, "daywisesummary")
            _upload_or_update_text(drive, dw_id, fname, md_content)
            log.info(f"Uploaded {fname} to Drive daywisesummary/")
            break
        except Exception as e:
            err = str(e).lower()
            is_transient = any(x in err for x in
                               ["10053", "10054", "connection", "aborted", "reset", "timeout"])
            if is_transient and attempt < 2:
                wait = 5 * (attempt + 1)
                log.warning(f"Drive upload attempt {attempt+1} failed (transient) — retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                log.warning(f"Drive upload of datewise summary failed after {attempt+1} attempt(s): {e}")
                break

    # Save locally for Obsidian (always — even if Drive upload fails)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / fname
    local_path.write_text(md_content, encoding="utf-8")
    log.info(f"Saved local: {local_path}")

    return local_path
