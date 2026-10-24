r"""
galleries_home.py — ONE page for every chart gallery, refreshed only when there is
something new.

  refresh   1) fetch the CI-built pages that are newer than yours (gallery_sync fetch, seconds)
            2) build locally: any CI page missing/failed in Drive, and the local-only pages
               (market state, Zanger, PF history, sold) — but ONLY when new Phase 1 data has
               arrived since you last built them (marker = the CI full run's build time)
            3) write galleries_home.html (menu + one viewer) and open it — one tab

A page whose local build failed for this marker is not retried on every click (a slow
failure would make every click slow); `--force` retries.

Run by gallery_all.bat / pf_gallery.bat (click) and by the GalleriesRefresh scheduled
task (weekdays 18:30 IST + at logon, catches up when the PC was off) with --hidden:
no console windows, output to logs/galleries_refresh.log.

Usage:
    python scripts/galleries_home.py refresh [--focus all|pf] [--no-open] [--hidden] [--force] [--dry-run]
    python scripts/galleries_home.py home    [--focus all|pf] [--no-open]     # page only, no Drive
Plan: .claude/plans/galleries-ci-fetch.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import gallery_sync as GS                           # noqa: E402  (loads .env)

ROOT = GS.ROOT
HOME = "galleries_home.html"
LOG = os.path.join(ROOT, "logs", "galleries_refresh.log")
STATE = os.path.join(ROOT, "logs", "galleries_state.json")
HIDDEN = False

# pages built on THIS PC only (need the local 25y store / Zanger library, or an
# uncommitted builder)
LOCAL = {
    "market_state.html":       [["guru/build_market_breadth.py"],
                                ["scripts/build_market_state_html.py", "--no-open"]],
    "gallery_zanger.html":     [["guru/patterns/build_pattern_gallery.py", "--no-open"]],
    "gallery_pf_history.html": [["scripts/build_pf_history.py", "--no-open"]],
    "gallery_sold.html":       [["scripts/build_sold_gallery.py", "--no-open"]],
}
# menu order + labels
MENU = [
    ("Market", "market_state.html", "Market state (macro)"),
    ("Market", "gallery.html", "All signals"),
    ("Market", "gallery_new_today.html", "New ideas today"),
    ("Market", "gallery_additions.html", "Recent additions (14d)"),
    ("Market", "gallery_drops.html", "Recent drops (7d)"),
    ("Market", "gallery_ipo.html", "Recent IPOs (1y)"),
    ("Market", "gallery_guru.html", "Guru screen"),
    ("Market", "gallery_darvas.html", "Darvas boxes"),
    ("Market", "gallery_zanger.html", "Zanger patterns"),
    ("Market", "gallery_guidance_watchlist.html", "Guidance watchlist"),
    ("Market", "gallery_tracker.html", "Decision tracker"),
    ("Portfolio", "gallery_pf.html", "PF daily"),
    ("Portfolio", "gallery_pf_weekly.html", "PF weekly"),
    ("Portfolio", "gallery_pf_monthly.html", "PF monthly"),
    ("Portfolio", "gallery_pf_history.html", "PF technical history"),
    ("Portfolio", "gallery_sold.html", "Sold stocks"),
]


def log(m):
    print(datetime.now().strftime("%H:%M:%S"), m, flush=True)


def _mtime(page):
    p = os.path.join(ROOT, page)
    return datetime.fromtimestamp(os.path.getmtime(p), timezone.utc) if os.path.exists(p) else None


def _utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _run(page, cmds) -> bool:
    """run a page's local build command(s); hidden mode: no console, output to the log."""
    kw = {"cwd": ROOT}
    if HIDDEN:
        kw.update(stdout=sys.stdout, stderr=subprocess.STDOUT,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    exe = sys.executable.replace("pythonw.exe", "python.exe")
    for c in cmds:
        log(f"  building {page}: {' '.join(c)}")
        if subprocess.run([exe] + c, **kw).returncode != 0:
            log(f"  FAILED: {page}")
            return False
    for src_page, spec in GS.PAGES.items():           # Darvas writes under guru/darvas/
        if src_page == page and spec.get("local_out"):
            with open(os.path.join(ROOT, spec["local_out"]), "rb") as a, \
                 open(os.path.join(ROOT, page), "wb") as b:
                b.write(a.read())
    return True


def _manifest():
    """-> manifest dict, plus `_darvas_modified` (Darvas lives outside the manifest)."""
    from _extractor_base import find_file, download_bytes
    from drive_io import drive_call
    d, repo, sub = GS._drive()
    fid = drive_call(lambda: sub(d, repo, GS.FOLDER), label="galleries folder")
    mid = drive_call(lambda: find_file(d, fid, GS.MANIFEST), label=GS.MANIFEST)
    man = json.loads(drive_call(lambda: download_bytes(d, mid), label=GS.MANIFEST)) if mid else {}
    folder, name = GS.PAGES["gallery_darvas.html"]["drive"]
    dfid = drive_call(lambda: sub(d, repo, folder), label=folder)
    xid = drive_call(lambda: find_file(d, dfid, name), label=name)
    if xid:
        man["_darvas_modified"] = drive_call(
            lambda: d.files().get(fileId=xid, fields="modifiedTime").execute(), label=name
        ).get("modifiedTime")
    return man


def _marker(man):
    """time new Phase 1 data reached Drive = the CI full run's build time (the guru-only
    re-render later in the evening does not count)."""
    v = man.get("pages", {}).get("gallery.html", {})
    return _utc(v.get("built_at_utc")) if v.get("status") == "ok" else None


def _state():
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                  # noqa: BLE001
        return {}


def refresh(force: bool, dry_run: bool) -> dict:
    man = {}
    try:
        man = _manifest()
    except Exception as e:                             # noqa: BLE001  offline: local only
        log(f"Drive unreachable ({str(e)[:80]}) — showing the pages already on this PC")
    marker = _marker(man)
    log(f"new Phase 1 data marker (CI full run): {GS._ist(marker.isoformat()) if marker else '-'}")
    st = _state()
    seen = {"manifest": man.get("built_at_utc"), "darvas": man.get("_darvas_modified")}
    missing = [p for p in GS.PAGES if not os.path.exists(os.path.join(ROOT, p))]
    if man and (force or missing or st.get("_fetched") != seen):
        for which in ("all", "pf"):
            GS.fetch(which, dry_run=dry_run, no_open=True, no_fallback=True)
        if not dry_run:
            st["_fetched"] = seen
    elif man:
        log("Drive has nothing newer since the last fetch — skipping the download step")

    mk = marker.isoformat() if marker else "none"
    todo = []
    for page, spec in GS.PAGES.items():                # CI page missing/failed in Drive
        v = man.get("pages", {}).get(page)
        if man and "drive" not in spec and (not v or v.get("status") != "ok"):
            todo.append((page, [spec["cmd"] + (["--no-open"] if "darvas" not in spec["cmd"][0] else [])]))
        elif "drive" in spec and not os.path.exists(os.path.join(ROOT, page)):
            todo.append((page, [spec["cmd"]]))
    for page, cmds in LOCAL.items():                   # local pages: stale vs the marker
        t = _mtime(page)
        if t is None or (marker and t < marker):
            todo.append((page, cmds))
    plan = []
    for page, cmds in todo:
        done = st.get(page, {})
        if not force and done.get("marker") == mk:     # already tried for this data
            log(f"  skip {page}: already built for this data"
                + ("" if done.get("ok") else " and it FAILED (use --force to retry)"))
            continue
        plan.append((page, cmds))
    if not plan:
        log("nothing to build — every page is up to date for the latest data")
    for page, cmds in plan:
        if dry_run:
            log(f"  WOULD build {page}: {' | '.join(' '.join(c) for c in cmds)}")
            continue
        st[page] = {"marker": mk, "ok": _run(page, cmds)}
    if not dry_run:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)
    return man


def write_home(man: dict, focus: str) -> str:
    st = _state()
    items = []
    for group, page, label in MENU:
        t = _mtime(page)
        v = man.get("pages", {}).get(page, {}) if man else {}
        built = _utc(v.get("built_at_utc")) if v.get("status") == "ok" else None
        if page == "gallery_darvas.html":               # own Drive file, not in the manifest
            built = _utc(man.get("_darvas_modified")) if man else None
        src = "CI" if built and t and t >= built else "this PC"
        when = built if src == "CI" and built else t
        failed = st.get(page, {}).get("ok") is False
        items.append({"g": group, "f": page, "l": label, "ok": t is not None,
                      "when": GS._ist(when.isoformat()) if when else "not built",
                      "src": src, "bad": failed})
    first = next((i["f"] for i in items if i["g"] == ("Portfolio" if focus == "pf" else "Market") and i["ok"]),
                 items[0]["f"])
    html = (_TPL.replace("__ITEMS__", json.dumps(items))
            .replace("__FOCUS__", json.dumps(first if focus else ""))
            .replace("__NOW__", datetime.now().strftime("%d %b %Y %H:%M")))
    out = os.path.join(ROOT, HOME)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"wrote {HOME}")
    return out


_TPL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chart galleries</title>
<style>
 :root{--bg:#f4f6f9;--panel:#fff;--ink:#1d2433;--mute:#6b7280;--line:#e3e7ee;--acc:#1a3d6e;--accbg:#e8eef8;--bad:#b42318;--badbg:#fdecea}
 @media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11151c;--panel:#181d26;--ink:#e6e9ef;--mute:#9aa3b2;--line:#2a313d;--acc:#8fb4ff;--accbg:#22304a;--bad:#ff8a80;--badbg:#3a1f1f}}
 :root[data-theme="dark"]{--bg:#11151c;--panel:#181d26;--ink:#e6e9ef;--mute:#9aa3b2;--line:#2a313d;--acc:#8fb4ff;--accbg:#22304a;--bad:#ff8a80;--badbg:#3a1f1f}
 *{box-sizing:border-box} html,body{height:100%;margin:0}
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);display:flex}
 nav{width:250px;flex:none;background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;padding:10px 8px}
 nav h1{font-size:15px;margin:2px 8px 2px} nav .sub{font-size:11px;color:var(--mute);margin:0 8px 10px}
 nav h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:14px 8px 4px}
 nav button{display:block;width:100%;text-align:left;border:0;background:none;color:inherit;padding:6px 8px;border-radius:6px;cursor:pointer;font:inherit}
 nav button:hover{background:var(--accbg)} nav button.on{background:var(--accbg);box-shadow:inset 3px 0 0 var(--acc)}
 nav button b{display:block;font-size:13px;font-weight:600} nav button span{font-size:11px;color:var(--mute)}
 nav button.bad span{color:var(--bad)} nav button:disabled{opacity:.45;cursor:default}
 main{flex:1;display:flex;flex-direction:column;min-width:0}
 .bar{display:flex;gap:10px;align-items:center;padding:6px 12px;border-bottom:1px solid var(--line);background:var(--panel);font-size:12px;color:var(--mute)}
 .bar a{color:var(--acc)} .bar strong{color:var(--ink);font-size:13px}
 iframe{flex:1;border:0;width:100%;background:#fff}
 @media (max-width:700px){body{flex-direction:column} nav{width:auto;max-height:40vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<nav id="nav"><h1>Chart galleries</h1><div class="sub">Page written __NOW__</div></nav>
<main><div class="bar"><strong id="ttl"></strong><span id="meta"></span><a id="pop" href="#" target="_blank">open in its own tab &#8599;</a></div>
<iframe id="view" title="gallery"></iframe></main>
<script>
const I=__ITEMS__, FOCUS=__FOCUS__, nav=document.getElementById('nav');
let g='';
I.forEach(it=>{
  if(it.g!==g){g=it.g;const h=document.createElement('h2');h.textContent=g;nav.appendChild(h);}
  const b=document.createElement('button');b.dataset.f=it.f;b.disabled=!it.ok;
  if(it.bad)b.classList.add('bad');
  b.innerHTML='<b></b><span></span>';b.querySelector('b').textContent=it.l;
  b.querySelector('span').textContent=it.ok?(it.when+' · '+it.src+(it.bad?' · last build FAILED':'')):'not built yet';
  b.onclick=()=>show(it.f);nav.appendChild(b);
});
function show(f){
  const it=I.find(x=>x.f===f); if(!it||!it.ok)return;
  document.getElementById('view').src=f;
  document.getElementById('ttl').textContent=it.l;
  document.getElementById('meta').textContent='updated '+it.when+' · built by '+it.src+(it.bad?' · last build FAILED, showing the older copy':'');
  document.getElementById('pop').href=f;
  nav.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.f===f));
  try{localStorage.setItem('gal_last',f)}catch(e){}
}
let start=FOCUS; if(!start){try{start=localStorage.getItem('gal_last')}catch(e){}}
show((I.find(x=>x.f===start&&x.ok)||I.find(x=>x.ok)||{}).f);
</script></body></html>"""


def main():
    global HIDDEN
    ap = argparse.ArgumentParser(description="one page for all chart galleries")
    ap.add_argument("cmd", choices=["refresh", "home"])
    ap.add_argument("--focus", choices=["all", "pf"], default=None,
                    help="open on the market (all) or portfolio (pf) section")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--hidden", action="store_true",
                    help="scheduled-task mode: no console windows, output to logs/galleries_refresh.log")
    ap.add_argument("--force", action="store_true", help="retry pages whose build failed for this data")
    ap.add_argument("--dry-run", action="store_true", help="show what would be fetched/built; write nothing")
    a = ap.parse_args()
    if a.hidden:
        HIDDEN = True
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        if os.path.exists(LOG) and os.path.getsize(LOG) > 2_000_000:
            os.replace(LOG, LOG + ".old")
        sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
        log("=" * 20 + " scheduled refresh")
    man = {}
    if a.cmd == "refresh":
        man = refresh(a.force, a.dry_run)
        if a.dry_run:
            log("DRY RUN: nothing downloaded, built or written"); return
    else:
        try:
            man = _manifest()
        except Exception:                              # noqa: BLE001
            pass
    out = write_home(man, a.focus)
    if not a.no_open:
        webbrowser.open("file:///" + out.replace("\\", "/"))


if __name__ == "__main__":
    main()
