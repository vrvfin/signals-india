#!/usr/bin/env python3
"""model_registry — one place that says which LLM models exist, and which are ALIVE.

THE PROBLEM THIS REPLACES. Model chains were hard-coded in 14+ scripts:
_extractor_base (P1_MODELS, BACKFILL_EXTRA_MODELS), extract_concall (CONCALL_MODELS),
ingest_announcements, ask_company, build_catalyst_notes, build_classification,
build_investigative_fraud, build_fraud_risk, daily_ar_summary, daily_research_summary,
company_deep_report, eval_providers, extract_structure, extract_mgmt_quotes. When
`gemini-2.0-flash` was retired by Google it went 404 in every one of them, and because
it sat LAST in P1_MODELS the failure only showed when the two models ahead of it were
overloaded - so extraction died exactly on the busy days. Measured 2026-09-03: it is
still 404, and it is still referenced in at least two live chains.

RETIRED, MEASURED, DO NOT RE-ADD (probed 2026-09-04, one FREE_POOL key):
    gemini-2.0-flash        404 "no longer available"
    gemini-2.0-flash-lite   404 "no longer available"
The -lite one was the ONLY fallback in build_classification and build_catalyst_notes,
both of which run nightly - so once gemini-2.5-flash-lite was busy those two had no
second model at all. That is the failure this module exists to make impossible, which
is why they were the first two migrated.

THE SPLIT THAT MAKES THIS WORK. Two different questions, answered in two different ways:

  AVAILABILITY  is discovered. It changes without warning when a provider retires a
                model, so it is probed daily and cached on Drive.
  PREFERENCE    is declared. No probe can tell you that a forensic annual-report pass
                wants a stronger model than a one-line announcement summary; that is a
                judgement about the WORK, and it belongs in code, reviewed like code.

resolve() composes them: walk the declared preference order, keep what the probe found
alive. So a retired model disappears from every chain the day after it dies, and nothing
silently downgrades to a weaker model than the caller asked for.

FAILING SAFE IS THE POINT. If the registry is missing, unreadable, stale, or empty,
resolve() returns the declared chain UNFILTERED. A registry outage must never be able to
stop extraction - the worst case has to be today's behaviour, not less.

Usage:
    python scripts/model_registry.py --probe          # daily: discover + probe + weekly exam
    python scripts/model_registry.py --dry-run        # no generation: what would be probed/scored
    python scripts/model_registry.py --show           # what each chain resolves to now
    python scripts/model_registry.py --self-test

    from model_registry import resolve
    models = resolve("P1", drive, index_id)           # -> ["gemini-2.5-flash-lite", ...]

    # v2 (2026-10-03) — THE standard for new callers: kind of work, not a model list
    from model_registry import pick, overload_keys
    models = pick("WRITE", fallback=[...])            # or "BULK"
    pool = BucketPool(keys, models, model_overload_keys=overload_keys(len(keys)))
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_D), ".env"))

REGISTRY_FILE = "model_registry.json"
STALE_DAYS = 7          # beyond this the probe is not trusted and chains pass through


# ------------------------------------------------------------------ #
#  DECLARED PREFERENCE - reviewed like code, never auto-generated     #
# ------------------------------------------------------------------ #
# Order is best-suited first. A chain is a fallback ladder, not a set: the first LIVE
# entry is used, so put the model whose OUTPUT you want at the top and cheaper stand-ins
# below it. Quota is per (project, model, day), so two chains sharing a model share its
# daily bucket - which is why the lite tiers are kept distinct from the quality tiers.
CHAINS: dict[str, list[str]] = {
    # Phase-2 structured extraction (AR / results / rating / presentation). Lite tier:
    # these run in bulk and the parse is bounded, so throughput beats eloquence.
    #
    # ORDERED BY MEASURED FAILURE RATE, NOT BY VERSION NUMBER. From gemini_usage.parquet
    # over 2026-08-05..09-04, 4,253 rows:
    #     gemini-3.1-flash-lite   9,837 ok /   805 fail =  7.6%
    #     gemini-2.5-flash-lite   5,934 ok / 2,039 fail = 25.6%
    # 2.5-flash-lite led this chain on judgement and was wrong: it fails more than three
    # times as often as the model behind it, and the chain is walked in order, so the
    # worse model was absorbing the first attempt on every document.
    "P1": ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"],
    # Extra buckets a backfill may burn once the P1 buckets are spent. Measured:
    # 2.5-flash 21.7% fail, 3.5-flash 27.8% - so the steadier one goes first here too.
    "BACKFILL_EXTRA": ["gemini-2.5-flash", "gemini-3.5-flash"],
    # Concall (P0). Deliberately DISJOINT from P1 so a backfill can never eat the daily
    # bucket the live concall run depends on.
    "CONCALL": ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"],
    # Long-form synthesis: deep dives, research digests. Quality first.
    "QUALITY": ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash",
                "gemini-2.5-flash-lite"],
    # Short utility passes: announcement one-liners, classification, tagging.
    "LITE_UTILITY": ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
                     "gemini-3.5-flash-lite"],
    # N7/N8 narrative passes (company structure, management quotes). Every record is
    # kept only if its evidence span is found VERBATIM in the source, so a lite model's
    # paraphrasing costs records outright - this stays on the flash tier. Led by
    # gemini-2.5-flash rather than gemini-3.5-flash on purpose: 3.5-flash leads CONCALL
    # (P0), and an enrichment pass must never eat the live concall run's daily bucket.
    # It sits third here, reachable only once the two ahead of it are spent.
    "NARRATIVE": ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.5-flash"],
    # Video understanding: Gemini watches a public YouTube URL (management interviews,
    # fetch_mgmt_interviews.py, 2026-09-11). Led by models NO other chain names, so the
    # interviews never share a daily bucket with Phase 2 - and never led by the CONCALL
    # leader. Probed 2026-09-11 (--probe-video, 120 s clip of a 17-min interview): all
    # four accept YouTube URLs; 3.7/3.6/3.5-lite took ~6 s and 10,943 tokens, 2.5-flash
    # 27 s and 35,422. No failure-rate data yet - re-order from gemini_usage once it exists.
    "MEDIA": ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
              "gemini-2.5-flash"],
}


# ------------------------------------------------------------------ #
#  Availability - probed                                             #
# ------------------------------------------------------------------ #

def _candidates() -> list[str]:
    """Every model any chain might want, de-duplicated, order preserved."""
    seen, out = set(), []
    for chain in CHAINS.values():
        for m in chain:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def _client(api_key: str):
    """google-genai client, the SAME SDK gemini_pool uses (the old google.generativeai
    import crashed every CI probe from 2026-09-05: that package is not installed there,
    so the registry froze at 2026-09-04 and every chain ran unfiltered for a month)."""
    from google import genai
    from google.genai import types as gt
    return genai.Client(api_key=api_key, http_options=gt.HttpOptions(
        timeout=120_000, retry_options=gt.HttpRetryOptions(attempts=1)))


def probe(api_key: str, models: list[str] | None = None) -> dict:
    """Ask the provider which of our candidates actually answer. Returns the registry.

    ONE key is enough. "This model is no longer available" is an account-wide fact, not
    a per-key one, and probing every key would burn N times the quota to learn the same
    thing. A key-specific failure (rate limit, 503) is NOT treated as unavailability -
    only an explicit not-found is, so a busy afternoon cannot empty the registry.
    A free-tier quota of ZERO ("limit: 0") is recorded separately as no_free_quota: the
    model exists but this account can never call it, so ranking must skip it.
    """
    client = _client(api_key)
    models = models or _candidates()
    live, dead, unknown, no_quota = [], {}, {}, {}
    for m in models:
        try:
            client.models.generate_content(
                model=m, contents="Reply with the single word: ok",
                config={"max_output_tokens": 64, "temperature": 0})
            live.append(m)            # any answer at all = the model is callable
        except Exception as e:
            msg = str(e).replace("\n", " ")[:160]
            low = msg.lower()
            if "not found" in low or "404" in low or "no longer available" in low:
                dead[m] = msg
            elif "limit: 0" in low:
                no_quota[m] = msg
            else:
                # 429 / 503 / transient. Availability is UNKNOWN, and unknown must not
                # read as dead - a rate-limited probe would otherwise delete a healthy
                # model from every chain for a day.
                unknown[m] = msg
    return {"checked_at": datetime.now().isoformat(timespec="seconds"),
            "live": live, "dead": dead, "unknown": unknown, "no_free_quota": no_quota}


# ------------------------------------------------------------------ #
#  v2 (2026-10-03): DISCOVER -> PROBE -> QUALITY-RANK -> pick(profile) #
# ------------------------------------------------------------------ #
# User: "new models keep coming: 1) create the latest model list 2) check which work
# 3) rank by quality of output and pick the best — a standard for all processes".
# CHAINS + resolve() above stay as they are for scripts not yet migrated; new callers use
# pick(profile, fallback=[...]) — one line, local and CI read the same registry on Drive.

# Profiles = the KIND of work, not a model list:
#   WRITE  final write-ups (deep dive synthesis, research answers): best quality first.
#   BULK   many calls (chunk / document / theme summaries, daily research docs): quality
#          ranked, but a model busy (503) on more than BUSY_DEMOTE_PCT of its calls in the
#          last BUSY_WINDOW_DAYS goes to the back (user decision 2026-10-03).
PROFILES = ("WRITE", "BULK")
BUSY_DEMOTE_PCT = 25.0
BUSY_WINDOW_DAYS = 7
QUALITY_STALE_DAYS = 7          # re-score weekly, and at once for a newly discovered model

# Google lists image / voice / music / video / embedding / live / agent models next to the
# text ones. Only text generation is a candidate; "-latest" aliases are skipped because
# they silently re-point to another model (and share its daily quota).
_TEXT_FAMILY = ("gemini-", "gemma-")
_NOT_TEXT = re.compile(r"image|tts|audio|live|embedding|transcribe|robotics|computer-use|"
                       r"customtools|translate|omni|-latest$", re.IGNORECASE)


def discover(api_key: str) -> list[str]:
    """Every text-generation model the account can see today (models.list, no generation)."""
    out = []
    client = _client(api_key)     # keep a reference: google-genai 2.x closes an unreferenced
    for m in client.models.list(config={"page_size": 200}):   # client mid-iteration
        name = (m.name or "").replace("models/", "")
        acts = m.supported_actions or []
        if (name.startswith(_TEXT_FAMILY) and "generateContent" in acts
                and not _NOT_TEXT.search(name)):
            out.append(name)
    return sorted(set(out))


# ---- quality: one fixed exam with known answers, scored by code -------------------
# A proxy for output quality, chosen because it is what our work needs most: pull exact
# figures from a results note, compute from them without inventing, refuse to fill a
# missing field, and follow an output format. It does NOT measure long-form writing.
_EXAM_TEXT = (
    "Q2 FY27 results (quarter ended 30 Sep 2026). Revenue from operations Rs 1,248.6 crore "
    "vs Rs 1,102.3 crore in Q2 FY26. EBITDA Rs 212.4 crore; EBITDA margin 17.0% (16.1% a "
    "year ago). PAT Rs 131.9 crore vs Rs 118.4 crore. Exports were 38% of revenue. The "
    "board declared an interim dividend of Rs 4 per share. Management guided FY27 revenue "
    "growth of 14-16% and said the new Pune plant (capex Rs 350 crore) will start in Q4 "
    "FY27. Net debt fell to Rs 96 crore from Rs 180 crore in March 2026.")
_EXAM_PROMPT = (
    "Read the results note and return ONLY a JSON object (no prose, no code fences) with "
    "exactly these keys: revenue_cr, revenue_prev_cr, ebitda_cr, ebitda_margin_pct, pat_cr, "
    "pat_prev_cr, export_share_pct, dividend_per_share_rs, capex_cr, net_debt_cr, "
    "guidance_low_pct, guidance_high_pct (numbers); plant_city, plant_start (strings); "
    "order_book_cr (number, or null if the note does not state it); revenue_yoy_pct and "
    "pat_yoy_pct (computed, one decimal); net_debt_change_cr (current minus March 2026); "
    "bullets (a list of exactly 3 strings, each at most 20 words and each containing a "
    "figure from the note).\n\nNOTE:\n" + _EXAM_TEXT)
_EXAM_FIGURES = {"revenue_cr": 1248.6, "revenue_prev_cr": 1102.3, "ebitda_cr": 212.4,
                 "ebitda_margin_pct": 17.0, "pat_cr": 131.9, "pat_prev_cr": 118.4,
                 "export_share_pct": 38, "dividend_per_share_rs": 4, "capex_cr": 350,
                 "net_debt_cr": 96, "guidance_low_pct": 14, "guidance_high_pct": 16}
_EXAM_COMPUTED = {"revenue_yoy_pct": 13.3, "pat_yoy_pct": 11.4, "net_debt_change_cr": -84}


def _num(v):
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def score_exam(text: str) -> tuple[int, list[str]]:
    """0-100 and the names of the checks that failed. Pure function: unit-tested."""
    failed, total = [], 0
    raw = (text or "").strip()
    total += 1
    try:
        json.loads(raw)                                   # format: pure JSON, no fences
    except Exception:
        failed.append("pure_json")
    body = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        d = json.loads(body[body.index("{"): body.rindex("}") + 1])
    except Exception:
        d = {}
    for k, want in _EXAM_FIGURES.items():
        total += 1
        got = _num(d.get(k))
        if got is None or abs(got - want) > 0.05:
            failed.append(k)
    for k, want in _EXAM_COMPUTED.items():
        total += 1
        got = _num(d.get(k))
        if got is None or abs(got - want) > 0.15:
            failed.append(k)
    total += 1
    if str(d.get("plant_city", "")).strip().lower() != "pune":
        failed.append("plant_city")
    total += 1
    if "q4fy27" not in str(d.get("plant_start", "")).replace(" ", "").lower():
        failed.append("plant_start")
    total += 1
    if "order_book_cr" not in d or d.get("order_book_cr") is not None:
        failed.append("order_book_null")                  # must not invent a figure
    b = d.get("bullets")
    total += 1
    if not (isinstance(b, list) and len(b) == 3):
        failed.append("bullets_three")
    total += 1
    if not (isinstance(b, list) and b and all(isinstance(x, str) and len(x.split()) <= 20
                                               and re.search(r"\d", x) for x in b)):
        failed.append("bullets_rules")
    return round(100 * (total - len(failed)) / total), failed


def quality_eval(api_key: str, models: list[str]) -> dict:
    """{model: {score, failed, latency_s, checked_at}}. A busy/limited model is skipped
    (no entry), so a bad hour cannot overwrite last week's score."""
    import time
    client, out = _client(api_key), {}
    for m in models:
        t0 = time.time()
        try:
            # 16k: THINKING tokens count against max_output_tokens. At 4,096 measured
            # gemini-3-flash-preview spent 3,931 thinking, was cut off mid-JSON and scored
            # 0 — a budget artefact, not its quality. Our real calls are not capped low.
            r = client.models.generate_content(
                model=m, contents=_EXAM_PROMPT,
                config={"temperature": 0, "max_output_tokens": 16384})
            text = r.text or ""
        except Exception as e:
            print(f"  exam skipped for {m}: {str(e).replace(chr(10), ' ')[:90]}")
            continue
        score, failed = score_exam(text)
        um = getattr(r, "usage_metadata", None)
        fin = r.candidates[0].finish_reason if getattr(r, "candidates", None) else None
        out[m] = {"score": score, "failed": failed,
                  "latency_s": round(time.time() - t0, 1),
                  "thinking_tokens": getattr(um, "thoughts_token_count", None),
                  "finish": str(getattr(fin, "name", fin)),
                  "checked_at": datetime.now().isoformat(timespec="seconds")}
    return out


def busy_rates(drive, index_id: str, days: int = BUSY_WINDOW_DAYS) -> dict:
    """{model: {pct, calls}} — share of calls answered 503 in the last `days`, from
    gemini_usage.parquet (every pool run logs it). {} when unavailable."""
    try:
        import pandas as pd
        from _extractor_base import load_parquet, GEMINI_USAGE_COLS
        u = load_parquet(drive, index_id, "gemini_usage.parquet", GEMINI_USAGE_COLS)
        u["ts"] = pd.to_datetime(u["ts"], errors="coerce")
        u = u[u["ts"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
        out = {}
        for m, g in u.groupby("model"):
            ok = pd.to_numeric(g["ok"], errors="coerce").fillna(0).sum()
            busy = pd.to_numeric(g["overload_503"], errors="coerce").fillna(0).sum()
            if ok + busy:
                out[str(m)] = {"pct": round(100 * busy / (ok + busy), 1),
                               "calls": int(ok + busy)}
        return out
    except Exception:
        return {}


def _tier(m: str) -> tuple:
    """Tie-break when scores are equal: pro > flash > lite, newer first, GA before preview."""
    tier = 3 if "pro" in m else 1 if ("lite" in m or m.startswith("gemma-")) else 2
    v = re.match(r"gemini-(\d+(?:\.\d+)?)", m)
    return (tier, float(v.group(1)) if v else 0.0, 0 if "preview" in m else 1)


def reserved_for_media() -> set[str]:
    """Models ONLY the MEDIA chain names (management interviews lead with them so no other
    job shares their daily quota). Text profiles never use them — user 2026-10-03.
    Today: gemini-3.7-flash, gemini-3.6-flash; follows the MEDIA chain if it changes."""
    others = {m for name, c in CHAINS.items() if name != "MEDIA" for m in c}
    return set(CHAINS["MEDIA"]) - others


def rank(reg: dict, profile: str) -> list[str]:
    """The live, scored models for a profile, best first. Unscored models are left out
    until the weekly exam has scored them — "rank by quality, then pick"."""
    q = reg.get("quality") or {}
    skip = (set(reg.get("dead") or {}) | set(reg.get("no_free_quota") or {})
            | reserved_for_media())
    pool = [m for m in (reg.get("live") or []) if m in q and m not in skip
            and m in set(reg.get("discovered") or reg.get("live") or [])]
    ordered = sorted(pool, key=lambda m: (q[m].get("score", 0), *_tier(m)), reverse=True)
    if profile == "BULK":
        busy = reg.get("busy_7d") or {}
        hot = {m for m in ordered if (busy.get(m) or {}).get("pct", 0) > BUSY_DEMOTE_PCT}
        ordered = [m for m in ordered if m not in hot] + [m for m in ordered if m in hot]
    # Never LEAD with the live concall (P0) model: same daily bucket — see CHAINS notes.
    p0 = CHAINS["CONCALL"][0]
    if len(ordered) > 1 and ordered[0] == p0:
        ordered = ordered[1:2] + [p0] + ordered[2:]
    return ordered


_REG_CACHE: dict = {}


def pick(profile: str, fallback: list[str], drive=None, index_id: str = "",
         log=print) -> list[str]:
    """THE standard call for every process: the models to use for this kind of work,
    best first. Reads the registry on Drive (connects itself when no handle is given),
    and returns `fallback` unchanged when the registry is missing, stale, or has nothing
    ranked yet — so a registry outage degrades to today's behaviour, never to no model.
        models = pick("WRITE", fallback=["gemini-2.5-flash", ...])
    """
    try:
        if "reg" not in _REG_CACHE:
            if drive is None or not index_id:
                from _extractor_base import get_drive, get_or_create_subfolder
                drive = drive or get_drive()
                repo = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"],
                                               "company_repo")
                index_id = get_or_create_subfolder(drive, repo, "_index")
            _REG_CACHE["reg"] = load_registry(drive, index_id)
        reg = _REG_CACHE["reg"]
        if not reg or not is_fresh(reg):
            log(f"  models[{profile}]: registry {'stale' if reg else 'missing'} — "
                f"using the static list")
            return list(fallback)
        got = rank(reg, profile)
        if not got:
            log(f"  models[{profile}]: nothing ranked yet — using the static list")
            return list(fallback)
        q = reg.get("quality") or {}
        log(f"  models[{profile}] (registry {str(reg.get('checked_at'))[:16]}): "
            + ", ".join(f"{m} {q[m]['score']}" for m in got[:6]))
        return got
    except Exception as e:
        log(f"  models[{profile}]: registry unavailable ({str(e)[:80]}) — static list")
        return list(fallback)


def overload_keys(n_keys: int) -> int:
    """Give-up rule (user 2026-10-03): drop a model for the run only after busy (503)
    replies on a third of the keys, not a fixed 3 — pass as BucketPool(model_overload_keys=)."""
    return max(3, n_keys // 3)


def load_registry(drive=None, index_id: str = "") -> dict:
    """The cached registry from Drive, or {} when unavailable."""
    try:
        from _extractor_base import find_file, download_bytes
        if drive is None or not index_id:
            return {}
        fid = find_file(drive, index_id, REGISTRY_FILE)
        if not fid:
            return {}
        return json.loads(download_bytes(drive, fid).decode("utf-8"))
    except Exception:
        return {}


def save_registry(drive, index_id: str, reg: dict) -> None:
    from _extractor_base import upload_bytes, find_file
    data = json.dumps(reg, indent=2).encode("utf-8")
    fid = find_file(drive, index_id, REGISTRY_FILE)
    upload_bytes(drive, index_id, REGISTRY_FILE, data, "application/json",
                 existing_id=fid)


def is_fresh(reg: dict, stale_days: int = STALE_DAYS, now: datetime | None = None) -> bool:
    try:
        t = datetime.fromisoformat(str(reg.get("checked_at", ""))[:19])
        return (now or datetime.now()) - t <= timedelta(days=stale_days)
    except Exception:
        return False


# ------------------------------------------------------------------ #
#  Resolution - preference filtered by availability                  #
# ------------------------------------------------------------------ #

def resolve(chain: str, drive=None, index_id: str = "", reg: dict | None = None,
            stale_days: int = STALE_DAYS) -> list[str]:
    """The models to try, best first, with anything known-dead removed.

    Returns the DECLARED chain unchanged when the registry is missing, stale, or would
    empty the chain. That last guard matters: if a probe ran during an outage and marked
    everything dead, filtering would hand the caller an empty list and every extraction
    would fail. Passing the chain through instead degrades to today's behaviour.
    """
    declared = list(CHAINS.get(chain, []))
    if not declared:
        return []
    reg = reg if reg is not None else load_registry(drive, index_id)
    if not reg or not is_fresh(reg, stale_days):
        return declared
    dead = set((reg.get("dead") or {}).keys())
    kept = [m for m in declared if m not in dead]
    return kept or declared


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="Probe the provider and write the registry to Drive.")
    ap.add_argument("--show", action="store_true",
                    help="Print what each chain resolves to right now.")
    ap.add_argument("--key-prefix", default="FREE_POOL",
                    help="Env prefix for the key used to probe (default FREE_POOL).")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --probe: probe + exam, write nothing. Alone: NO generation "
                         "calls — list today's models, say what would be probed/scored, "
                         "and show the ranking from the stored registry.")
    ap.add_argument("--no-eval", action="store_true",
                    help="With --probe: skip the quality exam this run.")
    ap.add_argument("--eval-all", action="store_true",
                    help="With --probe: re-score every live model now, not only due ones.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import get_drive, get_or_create_subfolder, log
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    old = load_registry(drive, idx)

    from gemini_pool import load_keys_multi
    keys = load_keys_multi(os.environ, args.key_prefix)
    if (args.probe or args.dry_run) and not keys:
        print(f"ERROR: no {args.key_prefix}* keys in env")
        sys.exit(1)

    if args.probe or args.dry_run:
        # 1. DISCOVER (metadata only — no generation call)
        try:
            found = discover(keys[0])
        except Exception as e:
            log(f"discovery FAILED ({str(e)[:100]}) — probing the known models only")
            found = list(old.get("discovered") or [])
        new = sorted(set(found) - set(old.get("discovered") or []))
        cands = list(dict.fromkeys(found + _candidates()))
        log(f"discovered {len(found)} text model(s)"
            + (f"; NEW since last run: {new}" if new else ""))

    if args.dry_run and not args.probe:
        oq = old.get("quality") or {}
        due = [m for m in found if m not in oq or not is_fresh(oq[m], QUALITY_STALE_DAYS)]
        log(f"DRY RUN — no generation calls, nothing written.")
        log(f"  would probe {len(cands)}: {cands}")
        log(f"  would score (new or >{QUALITY_STALE_DAYS}d old, if live): {due}")
    elif args.probe:
        # 2. PROBE (one tiny call each)
        reg = probe(keys[0], cands)
        reg["discovered"] = found
        log(f"probed {len(cands)} model(s): {len(reg['live'])} live, {len(reg['dead'])} "
            f"dead, {len(reg['unknown'])} busy/unknown, "
            f"{len(reg['no_free_quota'])} with no free-tier quota")
        for kind in ("dead", "unknown", "no_free_quota"):
            for m, why in reg[kind].items():
                log(f"   {kind:<13} {m}  {why[:70]}")
        # 3. QUALITY — keep last scores; exam the live ones that are new or due
        quality = dict(old.get("quality") or {})
        if not args.no_eval:
            due = [m for m in reg["live"] if m in found and (
                args.eval_all or m not in quality
                or not is_fresh(quality[m], QUALITY_STALE_DAYS))]
            if due:
                log(f"quality exam for {len(due)} model(s): {due}")
                quality.update(quality_eval(keys[0], due))
        reg["quality"] = quality
        # 4. RELIABILITY from what our pools actually saw
        reg["busy_7d"] = busy_rates(drive, idx)
        reg["ranked"] = {p: rank(reg, p) for p in PROFILES}
        if args.dry_run:
            log("DRY RUN — registry not written.")
        else:
            save_registry(drive, idx, reg)
            log(f"registry written -> _index/{REGISTRY_FILE}")
        old = reg

    reg = old
    print(f"\nregistry: checked_at={reg.get('checked_at', 'never')} "
          f"fresh={is_fresh(reg)}")
    q, busy = reg.get("quality") or {}, reg.get("busy_7d") or {}
    if q:
        print(f"\n  {'model':<32} {'score':>5} {'secs':>5} {'think':>6} {'busy7d':>7} "
              f"{'calls':>6}  failed checks")
        for m in sorted(q, key=lambda m: -q[m].get("score", 0)):
            b = busy.get(m) or {}
            print(f"  {m:<32} {q[m].get('score', ''):>5} {str(q[m].get('latency_s', '-')):>5} "
                  f"{str(q[m].get('thinking_tokens', '-')):>6} {str(b.get('pct', '-')):>6}% "
                  f"{str(b.get('calls', '-')):>6}  {', '.join(q[m].get('failed', []))[:60]}")
    for p in PROFILES:
        print(f"  {p:<6} -> {rank(reg, p)}")
    for name in CHAINS:
        got = resolve(name, reg=reg)
        drop = [m for m in CHAINS[name] if m not in got]
        print(f"  {name:<16} {got}" + (f"   (dropped: {drop})" if drop else ""))


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    # RELATIVE TO NOW, NEVER A WALL-CLOCK DATE. This fixture used to be pinned to
    # datetime(2026, 9, 3): is_fresh() compares checked_at against the REAL clock and
    # STALE_DAYS is 7, so on 2026-09-10 the fixture silently aged out, resolve() stopped
    # filtering, and the two assertions below began to fail. They are in the fail-fast
    # block of pf_daily_mails.yml, so from 2026-09-11 every scheduled mail run died at
    # that step and NO MAIL WAS SENT FOR FOUR DAYS. A test that expires is worse than no
    # test: it fails long after the change that "broke" it, pointing at nothing.
    now = datetime.now()
    fresh = {"checked_at": now.isoformat(), "live": [], "dead": {}, "unknown": {}}

    check("every chain is non-empty", all(CHAINS.values()))
    check("no chain still names the retired gemini-2.0-flash",
          not any("gemini-2.0-flash" in m for c in CHAINS.values() for m in c))
    check("candidates de-duplicate across chains",
          len(_candidates()) == len(set(_candidates())))
    check("concall stays disjoint from P1 (separate daily buckets)",
          not (set(CHAINS["CONCALL"]) & set(CHAINS["P1"])))
    check("MEDIA is never led by the CONCALL (P0) leader",
          CHAINS["MEDIA"][0] != CHAINS["CONCALL"][0])

    # resolution
    reg = dict(fresh, dead={"gemini-2.5-flash-lite": "404"})
    got = resolve("P1", reg=reg, stale_days=7)
    check("a dead model is dropped from the chain",
          "gemini-2.5-flash-lite" not in got)
    check("the survivors keep their declared order",
          got == [m for m in CHAINS["P1"] if m != "gemini-2.5-flash-lite"])
    check("an unknown model is NOT treated as dead",
          "gemini-2.5-flash-lite" in resolve(
              "P1", reg=dict(fresh, unknown={"gemini-2.5-flash-lite": "429"})))

    # fail-safe
    # The guard that would have caught the expiry the moment it happened.
    check("the fixture this file tests with is itself FRESH", is_fresh(fresh))
    check("no registry -> declared chain", resolve("P1", reg={}) == CHAINS["P1"])
    stale = {"checked_at": (now - timedelta(days=30)).isoformat(), "dead": {"x": "y"}}
    check("stale registry -> declared chain",
          resolve("P1", reg=stale) == CHAINS["P1"])
    allde = dict(fresh, dead={m: "404" for m in CHAINS["P1"]})
    check("a registry that would empty the chain is ignored",
          resolve("P1", reg=allde) == CHAINS["P1"])
    check("an unknown chain name yields nothing, not a crash",
          resolve("NO_SUCH_CHAIN", reg=fresh) == [])

    # freshness
    check("today is fresh", is_fresh(fresh, 7, now))
    check("eight days old is stale",
          not is_fresh({"checked_at": (now - timedelta(days=8)).isoformat()}, 7, now))
    check("a malformed timestamp is stale", not is_fresh({"checked_at": "nonsense"}))
    check("an empty registry is stale", not is_fresh({}))

    # ---- v2: discover filter, exam scoring, ranking, pick fallback --------------------
    keep = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview",
            "gemma-4-31b-it"]
    drop = ["gemini-2.5-flash-image", "gemini-3.8-flash-tts", "gemini-3.8-live",
            "gemini-embedding-2", "gemini-3.5-transcribe", "gemini-flash-latest",
            "gemini-2.5-computer-use-preview-10-2025", "gemini-omni-flash-preview"]
    check("text models pass the discover filter",
          not any(_NOT_TEXT.search(m) for m in keep))
    check("image/voice/live/embedding/alias models are filtered out",
          all(_NOT_TEXT.search(m) for m in drop))
    perfect = json.dumps({**_EXAM_FIGURES, **_EXAM_COMPUTED, "plant_city": "Pune",
                          "plant_start": "Q4 FY27", "order_book_cr": None,
                          "bullets": ["Revenue rose to Rs 1,248.6 crore",
                                      "EBITDA margin 17.0%", "Net debt down to Rs 96 crore"]})
    check("a perfect exam answer scores 100", score_exam(perfect) == (100, []))
    fenced = "```json\n" + perfect + "\n```"
    s_f, f_f = score_exam(fenced)
    check("code fences cost only the format point", f_f == ["pure_json"] and 90 <= s_f < 100)
    invented = json.loads(perfect)
    invented["order_book_cr"] = 5000
    check("an invented order book is caught",
          "order_book_null" in score_exam(json.dumps(invented))[1])
    check("garbage scores low", score_exam("not json at all")[0] < 10)

    r2 = dict(fresh, live=["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite",
                           "gemini-2.5-pro", "gemini-9-unscored"],
              discovered=["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite",
                          "gemini-2.5-pro", "gemini-9-unscored"],
              no_free_quota={"gemini-2.5-pro": "limit: 0"},
              quality={"gemini-3.5-flash": {"score": 100}, "gemini-3.8-flash": {"score": 95},
                       "gemini-3.5-flash-lite": {"score": 95},
                       "gemini-2.5-pro": {"score": 100}},
              busy_7d={"gemini-3.8-flash": {"pct": 40.0, "calls": 50},
                       "gemini-3.5-flash-lite": {"pct": 3.7, "calls": 7000}})
    w = rank(r2, "WRITE")
    check("WRITE: unscored and no-free-quota models are left out",
          "gemini-9-unscored" not in w and "gemini-2.5-pro" not in w)
    check("WRITE: the concall (P0) leader never leads, even when best",
          w[0] != CHAINS["CONCALL"][0] and CHAINS["CONCALL"][0] in w)
    check("WRITE: equal scores tie-break flash over lite", w.index("gemini-3.8-flash")
          < w.index("gemini-3.5-flash-lite"))
    b = rank(r2, "BULK")
    check("BULK: a model busy >25% in 7 days goes to the back",
          b[-1] == "gemini-3.8-flash" and b.index("gemini-3.5-flash-lite") < b.index("gemini-3.8-flash"))
    check("the interview (MEDIA-only) models are reserved: 3.7-flash, 3.6-flash",
          reserved_for_media() == {"gemini-3.7-flash", "gemini-3.6-flash"})
    r3 = dict(r2, live=r2["live"] + ["gemini-3.7-flash"],
              discovered=r2["discovered"] + ["gemini-3.7-flash"],
              quality={**r2["quality"], "gemini-3.7-flash": {"score": 100}})
    check("text profiles never use a model reserved for interviews",
          "gemini-3.7-flash" not in rank(r3, "WRITE") + rank(r3, "BULK"))
    _REG_CACHE["reg"] = {}
    check("pick: no registry -> the caller's static list",
          pick("WRITE", ["x", "y"], log=lambda *_: None) == ["x", "y"])
    _REG_CACHE["reg"] = dict(r2, checked_at=(now - timedelta(days=30)).isoformat())
    check("pick: stale registry -> the caller's static list",
          pick("BULK", ["x"], log=lambda *_: None) == ["x"])
    _REG_CACHE["reg"] = r2
    check("pick: fresh registry -> the ranked list",
          pick("WRITE", ["x"], log=lambda *_: None) == w)
    _REG_CACHE.clear()
    check("give-up rule scales with keys, never below 3",
          (overload_keys(2), overload_keys(11), overload_keys(34)) == (3, 3, 11))

    print(f"\nmodel_registry self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
