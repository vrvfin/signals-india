r"""
galleries_home.py — ONE page for every chart gallery; Drive is the single source.

  refresh   (gallery_all.bat / pf_gallery.bat — a click) NEVER builds anything:
            1) decide whether today's run is needed (IST, user rule 2026-10-25):
                 Sat/Sun                  -> take the latest automatically
                 weekday before 16:00     -> take the latest (yesterday's) automatically
                 weekday from 16:00       -> today's run on Drive? download it. If not, ask:
                                             [Y] open the latest now  [W] wait for today's
                                             [N] cancel   (Enter = Y; after 18:00 it says
                                             today's run is later than usual)
            2) download the pages newer than yours (skipped when Drive has nothing new)
            3) write galleries_home.html (menu + one viewer) and open ONE tab
  task      (GalleriesRefresh scheduled task, --hidden: weekdays 18:30-23:30 every 30 min
            + at logon, catches up when the PC was off):
            1) market-state inputs: rebuild the breadth table (self-skips) and push it +
               the deep macro history to Drive, so CI can build market_state.html
            2) download the CI pages
            3) build the PC-only pages (Zanger, PF technical history) when today's CI run
               is newer than them, and PUBLISH them to Drive (gallery_sync --owner pc)
            4) rewrite galleries_home.html (no browser)

"Today's run" = the latest CI build of the market pages (manifest) is dated today, IST.
A PC page whose build failed for a run is not retried until the next run (--force).

Usage:
    python scripts/galleries_home.py refresh [--focus all|pf] [--no-open] [--yes] [--at "YYYY-MM-DD HH:MM"] [--dry-run]
    python scripts/galleries_home.py task --hidden [--force] [--dry-run]
    python scripts/galleries_home.py home [--focus all|pf] [--no-open]
Plan: .claude/plans/galleries-ci-fetch.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import gallery_sync as GS                           # noqa: E402  (loads .env)

ROOT = GS.ROOT
IST = GS.IST
HOME = "galleries_home.html"
LOG = os.path.join(ROOT, "logs", "galleries_refresh.log")
STATE = os.path.join(ROOT, "logs", "galleries_state.json")
HIDDEN = False
CHECK_FROM, LATE_FROM = 16, 18                      # IST hours (user rule)
WAIT_POLL_S, WAIT_MAX_H = 120, 3

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


def _fmt(t):
    return t.astimezone(IST).strftime("%a %d %b, %H:%M IST") if t else "-"


def _state():
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                  # noqa: BLE001
        return {}


def _save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)


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


def latest_run(man):
    """time of the latest CI build of the market pages = when today's Phase 1 data
    became charts (the guru-only re-render later in the evening does not count)."""
    ts = [_utc(v.get("built_at_utc")) for p, v in man.get("pages", {}).items()
          if v.get("status") == "ok" and v.get("builder", "ci") == "ci"
          and p != "gallery_guru.html" and v.get("built_at_utc")]
    return max(ts) if ts else None


def _gh_status() -> str:
    """one info line on today's Phase 1 / chart build from GitHub, when gh works."""
    def last(wf):
        r = subprocess.run(["gh", "run", "list", "--workflow", wf, "--limit", "1", "--json",
                            "status,conclusion,createdAt,updatedAt"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        runs = json.loads(r.stdout or "[]") if r.returncode == 0 else []
        return runs[0] if runs else None
    try:
        p1, gal = last("daily.yml"), last("galleries.yml")
    except Exception:                                  # noqa: BLE001  gh missing / offline
        return ""
    if not p1:
        return ""
    today = datetime.now(IST).date()
    started = _utc(p1["createdAt"])
    if started.astimezone(IST).date() != today:
        return "Phase 1 has not started today (GitHub)."
    if p1["status"] != "completed":
        return f"Phase 1 is still running (started {started.astimezone(IST):%H:%M} IST)."
    if p1["conclusion"] != "success":
        return f"Phase 1 finished today with '{p1['conclusion']}' — charts will not be rebuilt."
    done = _utc(p1["updatedAt"]).astimezone(IST)
    if gal and gal["status"] != "completed" and _utc(gal["createdAt"]) >= _utc(p1["updatedAt"]):
        return f"Phase 1 finished {done:%H:%M} IST; the charts are being built now."
    return f"Phase 1 finished {done:%H:%M} IST."


def decide(man, now, assume_yes: bool) -> str:
    """-> 'go' (download what is there and open) or 'cancel'. May wait for today's run."""
    run = latest_run(man)
    if now.weekday() >= 5:
        log(f"weekend — opening the latest run ({_fmt(run)})"); return "go"
    if now.hour < CHECK_FROM:
        log(f"before {CHECK_FROM}:00 IST — opening the latest run ({_fmt(run)})"); return "go"
    if run and run.astimezone(IST).date() == now.date():
        log(f"today's run is on Drive ({_fmt(run)})"); return "go"
    late = now.hour >= LATE_FROM
    print()
    print("  Today's charts are not on Drive yet"
          + (" — later than usual (Phase 1 or the chart build may have failed)." if late else "."))
    print(f"  Latest available: {_fmt(run)}")
    info = _gh_status()
    if info:
        print(f"  {info}")
    print("  [Y] open the latest now   [W] wait and open today's when ready   [N] cancel")
    if assume_yes or not sys.stdin or not sys.stdin.isatty():
        print("  -> Y (no prompt)"); return "go"
    try:
        ans = (input("  Choice (Enter = Y): ").strip().upper() or "Y")[:1]
    except EOFError:
        ans = "Y"
    if ans == "N":
        return "cancel"
    if ans != "W":
        return "go"
    print(f"  Waiting for today's run (checking every {WAIT_POLL_S // 60} min, up to "
          f"{WAIT_MAX_H} h; Ctrl+C stops waiting and opens the latest) ...")
    end = time.time() + WAIT_MAX_H * 3600
    try:
        while time.time() < end:
            time.sleep(WAIT_POLL_S)
            man.clear(); man.update(_manifest())
            run = latest_run(man)
            if run and run.astimezone(IST).date() == datetime.now(IST).date():
                log(f"today's run arrived ({_fmt(run)})"); return "go"
            log(f"  not yet — {_gh_status() or 'still waiting'}")
        log("gave up waiting — opening the latest run")
    except KeyboardInterrupt:
        log("stopped waiting — opening the latest run")
    return "go"


def download(man, force=False, dry_run=False):
    st = _state()
    seen = {"manifest": man.get("built_at_utc"), "darvas": man.get("_darvas_modified")}
    missing = [p for p in GS.PAGES if not os.path.exists(os.path.join(ROOT, p))]
    if force or missing or st.get("_fetched") != seen:
        for which in ("all", "pf"):
            GS.fetch(which, dry_run=dry_run, no_open=True, no_fallback=True)
        if not dry_run:
            st["_fetched"] = seen
            _save_state(st)
    else:
        log("Drive has nothing newer since the last download — nothing to fetch")


def _run(page, cmds) -> bool:
    """run a build command (task mode): no console window, output to the log."""
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
    return True


def task(force: bool, dry_run: bool) -> dict:
    t0 = time.time()
    # 1) market-state inputs for CI (breadth self-skips when already newer than the store)
    if not dry_run:
        _run("market breadth table", [["guru/build_market_breadth.py"]])
    GS.inputs_push(dry_run)
    # 2) CI pages
    man = _manifest()
    run = latest_run(man)
    log(f"latest CI run: {_fmt(run)}")
    download(man, force, dry_run)
    # 3) PC-only pages: rebuild when the CI run is newer than them, then publish
    st, mk, built = _state(), (run.isoformat() if run else "none"), []
    for page, spec in GS.PAGES.items():
        if GS._builder(spec) != "pc":
            continue
        t = _mtime(page)
        if t is not None and (run is None or t >= run):
            continue                                   # already built for this run
        if not force and st.get(page, {}).get("marker") == mk:
            log(f"  skip {page}: already tried for this run"
                + ("" if st[page].get("ok") else " and it FAILED (--force retries)"))
            continue
        if dry_run:
            log(f"  WOULD build + publish {page}"); continue
        ok = _run(page, [spec["cmd"] + ["--no-open"]])
        st[page] = {"marker": mk, "ok": ok}
        if ok:
            built.append(page)
    if not dry_run:
        _save_state(st)
        if built:
            GS.publish(t0, False, only=built, owner="pc")
            man = _manifest()
    if not built:
        log("no PC page needed a rebuild")
    return man


def write_home(man: dict, focus: str | None) -> str:
    st = _state()
    run = latest_run(man) if man else None
    items = []
    for group, page, label in MENU:
        t = _mtime(page)
        v = man.get("pages", {}).get(page, {}) if man else {}
        built = _utc(v.get("built_at_utc")) if v.get("status") == "ok" else None
        if page == "gallery_darvas.html":               # own Drive file, not in the manifest
            built = _utc(man.get("_darvas_modified")) if man else None
        pc = GS._builder(GS.PAGES.get(page, {})) == "pc"
        who = "this PC" if pc else "CI"
        when = built if built and t and t >= built else t
        old = bool(run and when and when.astimezone(IST).date() < run.astimezone(IST).date())
        bad = v.get("status") == "failed" or (pc and st.get(page, {}).get("ok") is False)
        items.append({"g": group, "f": page, "l": label, "ok": t is not None,
                      "when": GS._ist(when.isoformat()) if when else "not built",
                      "src": who, "bad": bad, "old": old})
    now = datetime.now(IST)
    if run and run.astimezone(IST).date() == now.date():
        banner = f"Today's run · built {run.astimezone(IST):%H:%M} IST"
    elif run:
        banner = f"Latest run: {_fmt(run)}" + ("" if now.weekday() >= 5 or now.hour < CHECK_FROM
                                               else " · today's not ready yet")
    else:
        banner = "Drive not reachable — pages already on this PC"
    first = next((i["f"] for i in items if i["g"] == ("Portfolio" if focus == "pf" else "Market") and i["ok"]),
                 items[0]["f"])
    html = (_TPL.replace("__ITEMS__", json.dumps(items))
            .replace("__FOCUS__", json.dumps(first if focus else ""))
            .replace("__BANNER__", banner)
            .replace("__NOW__", now.strftime("%d %b %Y %H:%M")))
    out = os.path.join(ROOT, HOME)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"wrote {HOME}")
    return out


_TPL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chart galleries</title>
<style>
 :root{--bg:#f4f6f9;--panel:#fff;--ink:#1d2433;--mute:#6b7280;--line:#e3e7ee;--acc:#1a3d6e;--accbg:#e8eef8;--bad:#b42318;--badbg:#fdecea;--warn:#8a5300;--warnbg:#fff4e0}
 @media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11151c;--panel:#181d26;--ink:#e6e9ef;--mute:#9aa3b2;--line:#2a313d;--acc:#8fb4ff;--accbg:#22304a;--bad:#ff8a80;--badbg:#3a1f1f;--warn:#f2c46b;--warnbg:#3a2f1a}}
 :root[data-theme="dark"]{--bg:#11151c;--panel:#181d26;--ink:#e6e9ef;--mute:#9aa3b2;--line:#2a313d;--acc:#8fb4ff;--accbg:#22304a;--bad:#ff8a80;--badbg:#3a1f1f;--warn:#f2c46b;--warnbg:#3a2f1a}
 *{box-sizing:border-box} html,body{height:100%;margin:0}
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);display:flex}
 nav{width:250px;flex:none;background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;padding:10px 8px}
 nav h1{font-size:15px;margin:2px 8px 2px} nav .sub{font-size:11px;color:var(--mute);margin:0 8px 6px}
 nav .run{font-size:11.5px;margin:0 8px 8px;padding:4px 7px;border-radius:5px;background:var(--accbg)}
 nav h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:14px 8px 4px}
 nav button{display:block;width:100%;text-align:left;border:0;background:none;color:inherit;padding:6px 8px;border-radius:6px;cursor:pointer;font:inherit}
 nav button:hover{background:var(--accbg)} nav button.on{background:var(--accbg);box-shadow:inset 3px 0 0 var(--acc)}
 nav button b{display:block;font-size:13px;font-weight:600} nav button span{font-size:11px;color:var(--mute)}
 nav button.old span{color:var(--warn)} nav button.bad span{color:var(--bad)} nav button:disabled{opacity:.45;cursor:default}
 main{flex:1;display:flex;flex-direction:column;min-width:0}
 .bar{display:flex;gap:10px;align-items:center;padding:6px 12px;border-bottom:1px solid var(--line);background:var(--panel);font-size:12px;color:var(--mute)}
 .bar a{color:var(--acc)} .bar strong{color:var(--ink);font-size:13px}
 iframe{flex:1;border:0;width:100%;background:#fff}
 @media (max-width:700px){body{flex-direction:column} nav{width:auto;max-height:40vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<nav id="nav"><h1>Chart galleries</h1><div class="sub">Page written __NOW__</div><div class="run">__BANNER__</div></nav>
<main><div class="bar"><strong id="ttl"></strong><span id="meta"></span><a id="pop" href="#" target="_blank">open in its own tab &#8599;</a></div>
<iframe id="view" title="gallery"></iframe></main>
<script>
const I=__ITEMS__, FOCUS=__FOCUS__, nav=document.getElementById('nav');
let g='';
function note(it){return (it.old?' · older data':'')+(it.bad?' · last build FAILED':'');}
I.forEach(it=>{
  if(it.g!==g){g=it.g;const h=document.createElement('h2');h.textContent=g;nav.appendChild(h);}
  const b=document.createElement('button');b.dataset.f=it.f;b.disabled=!it.ok;
  if(it.bad)b.classList.add('bad'); else if(it.old)b.classList.add('old');
  b.innerHTML='<b></b><span></span>';b.querySelector('b').textContent=it.l;
  b.querySelector('span').textContent=it.ok?(it.when+' · '+it.src+note(it)):'not built yet';
  b.onclick=()=>show(it.f);nav.appendChild(b);
});
function show(f){
  const it=I.find(x=>x.f===f); if(!it||!it.ok)return;
  document.getElementById('view').src=f;
  document.getElementById('ttl').textContent=it.l;
  document.getElementById('meta').textContent='updated '+it.when+' · built by '+it.src+note(it);
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
    ap.add_argument("cmd", choices=["refresh", "task", "home"])
    ap.add_argument("--focus", choices=["all", "pf"], default=None,
                    help="open on the market (all) or portfolio (pf) section")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--yes", action="store_true", help="never prompt; take the latest run")
    ap.add_argument("--at", help='pretend the IST clock is "YYYY-MM-DD HH:MM" (testing the time rule)')
    ap.add_argument("--hidden", action="store_true",
                    help="scheduled-task mode: no console windows, output to logs/galleries_refresh.log")
    ap.add_argument("--force", action="store_true", help="task: retry PC pages that failed for this run")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen; write nothing")
    a = ap.parse_args()
    if a.hidden:
        HIDDEN = True
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        if os.path.exists(LOG) and os.path.getsize(LOG) > 2_000_000:
            os.replace(LOG, LOG + ".old")
        sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
        log("=" * 20 + " scheduled refresh")
        if a.cmd == "refresh":                          # old task command line
            a.cmd = "task"
    man = {}
    if a.cmd == "task":
        man = task(a.force, a.dry_run)
        if a.dry_run:
            log("DRY RUN: nothing downloaded, built, published or written"); return
        write_home(man, None)
        return
    try:
        man = _manifest()
    except Exception as e:                             # noqa: BLE001  offline: local copies
        log(f"Drive unreachable ({str(e)[:80]}) — showing the pages already on this PC")
    if a.cmd == "refresh" and man:
        now = (datetime.strptime(a.at, "%Y-%m-%d %H:%M").replace(tzinfo=IST) if a.at
               else datetime.now(IST))
        if decide(man, now, a.yes) == "cancel":
            log("cancelled — nothing downloaded"); return
        download(man, dry_run=a.dry_run)
        if a.dry_run:
            log("DRY RUN: nothing downloaded or written"); return
    out = write_home(man, a.focus)
    if not a.no_open:
        webbrowser.open("file:///" + out.replace("\\", "/"))


if __name__ == "__main__":
    main()
