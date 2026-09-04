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
    python scripts/model_registry.py --probe          # daily: refresh availability
    python scripts/model_registry.py --show           # what each chain resolves to now
    python scripts/model_registry.py --self-test

    from model_registry import resolve
    models = resolve("P1", drive, index_id)           # -> ["gemini-2.5-flash-lite", ...]
"""
from __future__ import annotations

import argparse
import json
import os
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


def probe(api_key: str, models: list[str] | None = None) -> dict:
    """Ask the provider which of our candidates actually answer. Returns the registry.

    ONE key is enough. "This model is no longer available" is an account-wide fact, not
    a per-key one, and probing every key would burn N times the quota to learn the same
    thing. A key-specific failure (rate limit, 503) is NOT treated as unavailability -
    only an explicit not-found is, so a busy afternoon cannot empty the registry.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    models = models or _candidates()
    live, dead, unknown = [], {}, {}
    for m in models:
        try:
            genai.GenerativeModel(m).generate_content(
                "ok", generation_config={"max_output_tokens": 4})
            live.append(m)
        except Exception as e:
            msg = str(e).replace("\n", " ")[:160]
            low = msg.lower()
            if "not found" in low or "404" in low or "no longer available" in low:
                dead[m] = msg
            else:
                # 429 / 503 / transient. Availability is UNKNOWN, and unknown must not
                # read as dead - a rate-limited probe would otherwise delete a healthy
                # model from every chain for a day.
                unknown[m] = msg
    return {"checked_at": datetime.now().isoformat(timespec="seconds"),
            "live": live, "dead": dead, "unknown": unknown}


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
                    help="With --probe: report findings, write nothing.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import get_drive, get_or_create_subfolder, log
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    if args.probe:
        from gemini_pool import load_keys_multi
        keys = load_keys_multi(os.environ, args.key_prefix)
        if not keys:
            print(f"ERROR: no {args.key_prefix}* keys in env")
            sys.exit(1)
        reg = probe(keys[0])
        log(f"probed {len(_candidates())} model(s): "
            f"{len(reg['live'])} live, {len(reg['dead'])} dead, "
            f"{len(reg['unknown'])} unknown")
        for m, why in reg["dead"].items():
            log(f"   DEAD    {m}  {why[:70]}")
        for m, why in reg["unknown"].items():
            log(f"   unknown {m}  {why[:70]}")
        if args.dry_run:
            log("DRY RUN — registry not written.")
        else:
            save_registry(drive, idx, reg)
            log(f"registry written -> _index/{REGISTRY_FILE}")

    reg = load_registry(drive, idx)
    print(f"\nregistry: checked_at={reg.get('checked_at', 'never')} "
          f"fresh={is_fresh(reg)}")
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

    now = datetime(2026, 9, 3, 12, 0, 0)
    fresh = {"checked_at": now.isoformat(), "live": [], "dead": {}, "unknown": {}}

    check("every chain is non-empty", all(CHAINS.values()))
    check("no chain still names the retired gemini-2.0-flash",
          not any("gemini-2.0-flash" in m for c in CHAINS.values() for m in c))
    check("candidates de-duplicate across chains",
          len(_candidates()) == len(set(_candidates())))
    check("concall stays disjoint from P1 (separate daily buckets)",
          not (set(CHAINS["CONCALL"]) & set(CHAINS["P1"])))

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

    print(f"\nmodel_registry self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
