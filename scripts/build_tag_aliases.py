r"""
build_tag_aliases.py — tag vocabulary v2: canonical taxonomy + large alias map.

Replaces the thin v1 seed (build_tag_vocab.py, 138 tags). Inputs:
  tag_taxonomy_seed.py       hand-curated canonical tags + curated aliases (edit there)
  tag_aliases_manual.csv     hand-reviewed mappings of mined strings (text -> tag / UNMAPPED)
  research_index.parquet + company_classification.csv   (Drive, read-only, or --cache DIR)
Outputs (scripts/):
  tag_taxonomy.csv           one row per canonical tag (slug, type, level, parent, group ...)
  tag_aliases.csv            alias -> tag rows (curated + spelling variants + mined strings)
  tag_vocabulary.csv/.parquet   SAME schema as v1 (tag_slug, tag_type, display_name, status,
                             aliases, notes) — read by daily_research_summary.py
  _vocab_report/             local review files: unmapped.csv, ambiguous.csv, review_data.json

Run:  python scripts/build_tag_aliases.py                      # build + validate locally
      python scripts/build_tag_aliases.py --cache DIR          # use cached Drive inputs
      python scripts/build_tag_aliases.py --upload             # ALSO push vocab files to Drive
Drive is only WRITTEN with --upload (company_repo/_index/tag_vocabulary.parquet + csv,
tag_taxonomy.csv, tag_aliases.parquet).
"""
from __future__ import annotations
import os, re, io, sys, json, argparse, collections, unicodedata
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tag_taxonomy_seed as seed

OUT_TAXONOMY = SCRIPTS_DIR / "tag_taxonomy.csv"
OUT_ALIASES  = SCRIPTS_DIR / "tag_aliases.csv"
OUT_VOCAB    = SCRIPTS_DIR / "tag_vocabulary.csv"
OUT_VOCAB_PQ = SCRIPTS_DIR / "tag_vocabulary.parquet"
MANUAL       = SCRIPTS_DIR / "tag_aliases_manual.csv"
REPORT_DIR   = SCRIPTS_DIR.parent / "_vocab_report"          # gitignored review output

DRIVE_INDEX = "company_repo/_index"

# v1 slugs (build_tag_vocab.py) that must stay valid so existing tags never break.
LEGACY = {
    "sector": """auto auto_ancillary banks_private banks_public nbfc housing_finance microfinance
        insurance capital_markets amc cement chemicals specialty_chemicals agrochemicals fertilisers
        construction infrastructure roads_highways defence aerospace fmcg consumer_durables retail
        ecommerce healthcare pharma cdmo hospitals diagnostics it_services software_products
        internet_platforms telecom media metals_steel metals_nonferrous mining oil_gas refining
        gas_distribution power_generation power_transmission renewables ev batteries realty hotels
        logistics ports shipping railways paper sugar textiles footwear capital_goods
        industrial_machinery electronics_manufacturing semiconductors packaging paints jewellery
        agritech food_processing qsr education fintech other_sector""",
    "subsector": """two_wheelers four_wheelers commercial_vehicles tractors tyres generic_pharma
        api_bulk_drug branded_formulations wires_cables transformers data_center_reit qib_smallcap
        midcap largecap sme""",
    "doc_type": """single_company_ar single_company_note single_company_drhp single_company_rating
        single_company_policy multi_company_seminar multi_company_sector govt_policy macro_report
        concall results presentation other""",
    "theme": """capex_cycle capacity_expansion order_book_growth china_plus_one pli_scheme
        import_substitution export_growth premiumisation rural_demand urban_demand margin_expansion
        margin_pressure deleveraging debt_raise working_capital_stress credit_growth asset_quality
        nim_trend deposit_growth monsoon commodity_inflation commodity_deflation rate_hike rate_cut
        inr_depreciation gst_change budget_impact ev_transition renewable_energy data_center_demand
        defence_indigenisation real_estate_upcycle demand_slowdown market_share_gain
        new_product_launch ma_consolidation promoter_pledge governance_concern""",
}
LEGACY = {k: set(v.split()) for k, v in LEGACY.items()}

SECTOR_TYPES = ("sector_group", "sector", "subsector")
SPECIFICITY = {"subsector": 3, "sector": 2, "sector_group": 1}
# which tag types a mined string of a given origin may map to
ORIGIN_TYPES = {
    "cls_macro_sector": SECTOR_TYPES, "cls_sector": SECTOR_TYPES, "cls_industry": SECTOR_TYPES,
    "cls_subsector": SECTOR_TYPES, "cls_peer_group": SECTOR_TYPES,
    "research_phrase": SECTOR_TYPES + ("theme", "macro_factor", "doc_type", "policy"),
    "policy_string": ("policy", "theme", "macro_factor"),
    "source_string": ("source",),
    "doc_type_string": ("doc_type",),
}
NAME_FIRST = {"source_string", "policy_string"}

# --------------------------------------------------------------------------- normalise
# (UK form, US form) — used to GENERATE alias variants for the daily normaliser, which
# only lower-cases; matching here folds both to one form via _FOLD instead.
_SPELL = [("isation", "ization"), ("defence", "defense"), ("fertiliser", "fertilizer"),
          ("jewellery", "jewelry"), ("aluminium", "aluminum"), ("tyre", "tire"),
          ("centre", "center"), ("programme", "program"), ("fibre", "fiber"),
          ("specialty", "speciality"), ("labour", "labor")]
# word-anchored folds (so "entire" never becomes "entyre")
_FOLD = [(r"ization(s?)\b", r"isation\1"), (r"\b(\w{2,})ize(s|d|r|rs)?\b", r"\1ise\2"),
         (r"\bdefense\b", "defence"), (r"\bfertilizer", "fertiliser"), (r"\bjewelry\b", "jewellery"),
         (r"\baluminum\b", "aluminium"), (r"\btire(s?)\b", r"tyre\1"), (r"\bcenter(s?)\b", r"centre\1"),
         (r"\bprogram(s?)\b", r"programme\1"), (r"\bspeciality\b", "specialty"),
         (r"\bfiber(s?)\b", r"fibre\1"), (r"\blabor\b", "labour"), (r"\bcolor", "colour"),
         (r"organiz", "organis")]

def norm(s: str) -> str:
    """Canonical match key: lower, &->and, separators/punctuation->space, spelling unified."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[_/\-–—.,;:()\[\]'\"’`|*]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in _FOLD:
        s = re.sub(pat, rep, s)
    return s

def _singular(tok: str) -> str:
    if len(tok) < 4 or not tok.isalpha():
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith(("sses", "shes", "ches", "xes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok

def match_key(s: str) -> str:
    """norm() + every token singularised — plural/singular variants collide on purpose."""
    return " ".join(_singular(t) for t in norm(s).split())

def compact_key(s: str) -> str:
    return match_key(s).replace(" ", "")

# --------------------------------------------------------------------------- seed parse
def _parse_lines(block: str):
    for raw in block.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        parts = [p.strip() for p in raw.strip().split("|")]
        slug, disp = parts[0], parts[1]
        aliases = [a.strip() for a in (parts[2] if len(parts) > 2 else "").split(";") if a.strip()]
        yield indent, slug, disp, aliases

def build_taxonomy():
    nodes, cur = [], {}
    for indent, slug, disp, al in _parse_lines(seed.SECTOR_TREE):
        level = {0: 1, 2: 2, 4: 3}[indent]
        ttype = {1: "sector_group", 2: "sector", 3: "subsector"}[level]
        cur[level] = slug
        parent = cur.get(level - 1, "") if level > 1 else ""
        group = cur[1]
        nodes.append(dict(slug=slug, tag_type=ttype, level=level, parent_slug=parent,
                          group_slug=group, display_name=disp, aliases=al))
    for block, ttype in ((seed.THEMES, "theme"), (seed.MACRO_FACTORS, "macro_factor"),
                         (seed.POLICIES, "policy"), (seed.SOURCES, "source"),
                         (seed.DOC_TYPES, "doc_type"), (seed.EVENT_TYPES, "event_type")):
        for _, slug, disp, al in _parse_lines(block):
            nodes.append(dict(slug=slug, tag_type=ttype, level=0, parent_slug="",
                              group_slug="", display_name=disp, aliases=al))
    tax = pd.DataFrame(nodes)
    tax["legacy"] = [s in LEGACY.get(t, set()) for s, t in zip(tax.slug, tax.tag_type)]
    tax["status"] = "closed"
    return tax

# --------------------------------------------------------------------------- inputs
def load_inputs(cache: str | None):
    if cache:
        ri = pd.read_parquet(Path(cache) / "research_index.parquet")
        cl = pd.read_csv(Path(cache) / "classification.csv", keep_default_na=False)
        return ri, cl
    from dotenv import load_dotenv
    load_dotenv(SCRIPTS_DIR.parent / ".env")
    from daily_research_summary import drive_service, drive_download
    import build_classification as bcl
    svc = drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]
    ri = pd.read_parquet(io.BytesIO(drive_download(svc, f"{DRIVE_INDEX}/research_index.parquet", root)))
    cl = bcl.load_classification(svc, root)
    return ri, cl

def _jl(s):
    try:
        return json.loads(s) if isinstance(s, str) else []
    except Exception:
        return []

def mine(ri: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    rows = collections.Counter()
    for s in ri.get("summary_md", pd.Series(dtype=str)).fillna(""):
        m = re.search(r"\|\s*Industry/Theme\s*\|\s*(.+?)\s*\|", s)
        if not m:
            continue
        for part in re.split(r"[,;/]|\band\b|&|\(|\)", m.group(1)):
            p = re.sub(r"\s+", " ", part).strip(" .-*").lower()
            if 2 < len(p) < 70 and p not in ("na", "n/a"):
                rows[("research_phrase", p)] += 1
    for c in ("macro_sector", "sector", "industry", "subsector", "peer_group"):
        if c in cl.columns:
            for k, n in cl[c].astype(str).str.strip().value_counts().items():
                if k and k.lower() not in ("nan", "", "?", "unknown", "none"):
                    rows[(f"cls_{c}", k)] += int(n)
    for s in ri.get("policies", pd.Series(dtype=str)).fillna("[]"):
        for x in _jl(s):
            x = re.sub(r"\s+", " ", str(x)).strip()
            if x and x.lower() not in ("na", "none"):
                rows[("policy_string", x)] += 1
    for k, n in ri.get("source", pd.Series(dtype=str)).astype(str).str.strip().value_counts().items():
        rows[("source_string", k)] += int(n)
    for k, n in ri.get("doc_type", pd.Series(dtype=str)).astype(str).value_counts().items():
        rows[("doc_type_string", k)] += int(n)
    return pd.DataFrame([(o, t, n) for (o, t), n in rows.items()], columns=["origin", "text", "n"])

# --------------------------------------------------------------------------- alias build
def curated_aliases(tax: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in tax.itertuples():
        for a, origin in [(r.slug, "slug"), (r.display_name, "display")] + [(a, "curated") for a in r.aliases]:
            rows.append(dict(alias=a, tag_type=r.tag_type, slug=r.slug, origin=origin,
                             confidence="high", reviewed=True, n_seen=0))
    types = dict(zip(tax.slug, tax.tag_type))
    for raw in getattr(seed, "EXTRA_ALIASES", "").splitlines():
        if "|" not in raw:
            continue
        slug, al = (p.strip() for p in raw.split("|", 1))
        if slug not in types:
            raise SystemExit(f"EXTRA_ALIASES: unknown slug '{slug}'")
        for a in (x.strip() for x in al.split(";")):
            if a:
                rows.append(dict(alias=a, tag_type=types[slug], slug=slug, origin="curated",
                                 confidence="high", reviewed=True, n_seen=0))
    return pd.DataFrame(rows)

ROLE_WORDS = set(getattr(seed, "ROLE_WORDS", "").split())

def strip_roles(key: str) -> str:
    """Drop trailing generic role words: 'pump manufacturer' -> 'pump'."""
    toks = key.split()
    while len(toks) > 1 and toks[-1] in {match_key(w) for w in ROLE_WORDS}:
        toks.pop()
    return " ".join(toks)

def load_rules():
    rules = []
    for raw in getattr(seed, "RULES", "").splitlines():
        if raw.count("|") < 3:
            continue
        parts = [p.strip() for p in raw.split(" | ")]
        origin, rx, ttype, slug = parts[0], " | ".join(parts[1:-2]), parts[-2], parts[-1]
        rules.append((origin, re.compile(rx), ttype, slug))
    return rules

def _index(al: pd.DataFrame):
    """match_key -> [(tag_type, slug)], plus compact-key index (no spaces)."""
    idx, cidx = collections.defaultdict(set), collections.defaultdict(set)
    for r in al.itertuples():
        idx[match_key(r.alias)].add((r.tag_type, r.slug))
        cidx[compact_key(r.alias)].add((r.tag_type, r.slug))
    return idx, cidx

def _pick(hits, allowed):
    """Filter to allowed types; within the sector hierarchy keep only the most specific."""
    hits = {h for h in hits if h[0] in allowed}
    sec = [h for h in hits if h[0] in SPECIFICITY]
    if sec:
        best = max(SPECIFICITY[h[0]] for h in sec)
        hits = {h for h in hits if h[0] not in SPECIFICITY or SPECIFICITY[h[0]] == best}
    return hits

def _contained(key, idx, allowed, min_len=5):
    """Longest curated alias appearing as whole words inside key (medium confidence)."""
    best, hits = 0, set()
    padded = f" {key} "
    for k, hs in idx.items():
        if len(k) < min_len or len(k) < best or k == key:
            continue
        if f" {k} " in padded:
            h = _pick(hs, allowed)
            if not h:
                continue
            if len(k) > best:
                best, hits = len(k), set(h)
            elif len(k) == best:
                hits |= h
    return hits

def map_mined(mined: pd.DataFrame, cur: pd.DataFrame, manual: pd.DataFrame):
    idx, cidx = _index(cur)
    rules = load_rules()
    man = {}
    if manual is not None and not manual.empty:
        for r in manual.itertuples():
            man.setdefault(match_key(r.text), []).append((str(r.tag_type), str(r.slug), str(getattr(r, "note", "") or "")))
    out, unmapped = [], []
    for r in mined.itertuples():
        allowed = ORIGIN_TYPES[r.origin]
        key = match_key(r.text)
        hits, conf, via = set(), "high", "exact"
        if key in man:
            m = [x for x in man[key] if x[0] in allowed or x[1] == "UNMAPPED"]
            if m and m[0][1] == "UNMAPPED":
                unmapped.append(dict(origin=r.origin, text=r.text, n=r.n, reason=m[0][2] or "manual: no fitting tag"))
                continue
            hits, via = {(t, s) for t, s, _ in m if s != "UNMAPPED"}, "manual"
        if not hits:
            hits = _pick(idx.get(key, set()), allowed)
        if not hits:
            hits = _pick(cidx.get(compact_key(r.text), set()), allowed)
        if not hits and strip_roles(key) != key:
            hits = _pick(idx.get(strip_roles(key), set()), allowed)
        # Sources/policies: a known name INSIDE the text ("Motilal Oswal Financial Services
        # Ltd") beats the generic long-tail rules. Topics: rules first ("Nifty Bank ETF"
        # must be an ETF, not a bank).
        steps = ("contains", "rule") if r.origin in NAME_FIRST else ("rule", "contains")
        for step in steps:
            if hits:
                break
            if step == "rule":
                nk = norm(r.text)
                for o, rx, t, s in rules:
                    if r.origin.startswith(o) and t in allowed and rx.search(nk):
                        hits, conf, via = {(t, s)}, "medium", "rule"
                        break
            else:
                hits, conf, via = (_contained(strip_roles(key), idx, allowed,
                                               min_len=3 if r.origin == "source_string" else 5),
                                    "medium", "contains")
        if not hits:
            unmapped.append(dict(origin=r.origin, text=r.text, n=r.n, reason="no match"))
            continue
        for t, s in sorted(hits):
            out.append(dict(alias=r.text, tag_type=t, slug=s, origin=r.origin,
                            confidence=conf if via != "manual" else "high",
                            reviewed=(via == "manual"), n_seen=int(r.n), via=via))
    return pd.DataFrame(out), pd.DataFrame(unmapped)

def spelling_variants(al: pd.DataFrame) -> pd.DataFrame:
    """US spellings, &/and, hyphen/space and plural forms of curated aliases."""
    rows = []
    cur = al[al.origin.isin(["curated", "display"])]
    for r in cur.itertuples():
        a = str(r.alias).lower()
        vs = set()
        for uk, us in _SPELL:
            if uk in a:
                vs.add(a.replace(uk, us))
        if " and " in a:
            vs.add(a.replace(" and ", " & "))
        if "&" in a:
            vs.add(a.replace("&", "and"))
        if "-" in a:
            vs.add(a.replace("-", " ")); vs.add(a.replace("-", ""))
        toks = a.split()
        if toks and len(toks[-1]) >= 4 and toks[-1].isalpha():
            last = toks[-1]
            alt = _singular(last) if _singular(last) != last else (last + ("es" if last.endswith(("s", "x", "ch", "sh")) else "s"))
            vs.add(" ".join(toks[:-1] + [alt]))
        for v in vs - {a}:
            rows.append(dict(alias=v, tag_type=r.tag_type, slug=r.slug, origin="variant",
                             confidence="high", reviewed=True, n_seen=0))
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- validate
def validate(tax, aliases, manual=None):
    errs, warns = [], []
    if manual is not None and not manual.empty:
        types = dict(zip(tax.slug, tax.tag_type))
        bad = [f"{r.text} -> {r.tag_type}:{r.slug}" for r in manual.itertuples()
               if r.slug != "UNMAPPED" and types.get(r.slug) != r.tag_type]
        if bad:
            errs.append(f"tag_aliases_manual.csv rows with unknown slug or wrong type: {bad[:15]}")
    dup = tax[tax.slug.duplicated(keep=False)]
    if not dup.empty:
        errs.append(f"duplicate slugs: {sorted(set(dup.slug))}")
    slugs = set(tax.slug)
    bad_rule = [s for _, _, _, s in load_rules() if s not in slugs]
    if bad_rule:
        errs.append(f"RULES point to unknown slugs: {bad_rule}")
    for t, need in LEGACY.items():
        missing = sorted(s for s in need if s not in set(tax[tax.tag_type == t].slug))
        if missing:
            errs.append(f"legacy {t} slugs missing: {missing}")
    bad_parent = tax[(tax.parent_slug != "") & ~tax.parent_slug.isin(slugs)]
    if not bad_parent.empty:
        errs.append(f"unknown parents: {bad_parent.slug.tolist()}")
    orphan = aliases[~aliases.slug.isin(slugs)]
    if not orphan.empty:
        errs.append(f"aliases pointing to unknown slugs: {sorted(set(orphan.slug))[:20]}")
    cur = aliases[aliases.origin.isin(["slug", "display", "curated", "variant"])].copy()
    cur["k"] = cur.alias.map(match_key)
    amb = (cur.groupby(["k", "tag_type"]).slug.nunique().reset_index())
    amb = amb[amb.slug > 1]
    if not amb.empty:
        warns.append(f"{len(amb)} curated aliases map to >1 tag of the same type")
    return errs, warns, amb

# --------------------------------------------------------------------------- legacy vocab
def legacy_vocab(tax, aliases):
    """v1 schema. aliases column = pipe-joined lower-case aliases the daily normaliser
    (daily_research_summary.alias_map) matches after lower()+space-collapse."""
    al = aliases.copy()
    al["a"] = al.alias.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    grp = al.groupby("slug").a.apply(lambda s: "|".join(sorted(set(x for x in s if x and "|" not in x))))
    rows = []
    for r in tax.itertuples():
        rows.append(dict(tag_slug=r.slug, tag_type=r.tag_type, display_name=r.display_name,
                         status="closed", aliases=grp.get(r.slug, ""),
                         notes=f"parent={r.parent_slug}" if r.parent_slug else ""))
    rows += [
        dict(tag_slug="<company>", tag_type="company", display_name="Resolved to ISIN", status="open",
             aliases="", notes="Resolve mention -> ISIN via universe. Company aliases live in a separate table."),
        dict(tag_slug="<promoter>", tag_type="promoter", display_name="Promoter / KMP name", status="open",
             aliases="", notes="Free text -> lowercase_underscore slug."),
        dict(tag_slug="<fy_or_quarter>", tag_type="temporal", display_name="FY / quarter", status="open",
             aliases="", notes="Pattern fyYY or qXfyYY (e.g. fy26, q4fy26)."),
    ]
    return pd.DataFrame(rows, columns=["tag_slug", "tag_type", "display_name", "status", "aliases", "notes"])

# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="dir with research_index.parquet + classification.csv (skip Drive read)")
    ap.add_argument("--upload", action="store_true", help="push vocab files to Drive company_repo/_index/")
    args = ap.parse_args()

    tax = build_taxonomy()
    cur = curated_aliases(tax)
    ri, cl = load_inputs(args.cache)
    mined = mine(ri, cl)
    manual = pd.read_csv(MANUAL, keep_default_na=False) if MANUAL.exists() else pd.DataFrame()
    mapped, unmapped = map_mined(mined, cur, manual)
    var = spelling_variants(cur)
    aliases = pd.concat([cur, var, mapped.drop(columns=["via"], errors="ignore")], ignore_index=True)
    aliases["alias"] = aliases.alias.astype(str).str.strip()
    aliases = aliases[aliases.alias != ""]
    aliases["alias_norm"] = aliases.alias.map(norm)
    rank = {"slug": 0, "display": 1, "curated": 2, "variant": 3}
    aliases["_r"] = aliases.origin.map(lambda o: rank.get(o, 4))
    aliases = (aliases.sort_values(["_r"])
               .groupby(["alias", "tag_type", "slug"], as_index=False)
               .agg(alias_norm=("alias_norm", "first"), origin=("origin", "first"),
                    confidence=("confidence", "first"), reviewed=("reviewed", "max"),
                    n_seen=("n_seen", "sum")))

    errs, warns, amb = validate(tax, aliases, manual)

    # outputs
    tax_out = tax.drop(columns=["aliases"])
    tax_out.to_csv(OUT_TAXONOMY, index=False)
    aliases.sort_values(["tag_type", "slug", "alias"]).to_csv(OUT_ALIASES, index=False)
    vocab = legacy_vocab(tax, aliases)
    vocab.to_csv(OUT_VOCAB, index=False); vocab.to_parquet(OUT_VOCAB_PQ, index=False)
    REPORT_DIR.mkdir(exist_ok=True)
    unmapped.sort_values("n", ascending=False).to_csv(REPORT_DIR / "unmapped.csv", index=False)
    amb.to_csv(REPORT_DIR / "ambiguous.csv", index=False)
    # compact payload for the browsable review page
    cov_rows = []
    cov = mined.merge(unmapped[["origin", "text"]].assign(u=1), how="left", on=["origin", "text"])
    for o, g in cov.groupby("origin"):
        cov_rows.append(dict(origin=o, distinct=int(len(g)), mapped=int(g.u.isna().sum()),
                             weighted=round(float((g.n * g.u.isna()).sum() / max(1, g.n.sum())), 4)))
    payload = dict(
        taxonomy=tax_out[["slug", "tag_type", "level", "parent_slug", "group_slug", "display_name",
                          "legacy"]].to_dict("records"),
        aliases=aliases[["alias", "tag_type", "slug", "origin", "confidence", "n_seen"]].values.tolist(),
        unmapped=unmapped.sort_values("n", ascending=False)[["origin", "text", "n", "reason"]].values.tolist(),
        coverage=cov_rows)
    (REPORT_DIR / "review_data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                                            default=int), encoding="utf-8")

    # report
    print(f"taxonomy: {len(tax)} tags  " + str(tax.tag_type.value_counts().to_dict()))
    print(f"aliases : {len(aliases):,} rows ({aliases.alias_norm.nunique():,} distinct keys)  "
          + str(aliases.origin.value_counts().to_dict()))
    print("confidence:", aliases.confidence.value_counts().to_dict())
    cov = mined.merge(unmapped[["origin", "text"]].assign(u=1), how="left", on=["origin", "text"])
    cov["mapped"] = cov.u.isna()
    print("coverage of mined strings (distinct / weighted by frequency):")
    for o, g in cov.groupby("origin"):
        print(f"  {o:<18} {g.mapped.mean():6.1%}  /  {(g.n * g.mapped).sum() / max(1, g.n.sum()):6.1%}   (unmapped {int((~g.mapped).sum())})")
    for w in warns:
        print("WARN:", w)
    if errs:
        for e in errs:
            print("ERROR:", e)
        sys.exit(1)
    print("validation: OK")

    if args.upload:
        from dotenv import load_dotenv
        load_dotenv(SCRIPTS_DIR.parent / ".env")
        from daily_research_summary import drive_service, drive_upload_with_retry
        svc = drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]
        al_buf = io.BytesIO(); aliases.to_parquet(al_buf, index=False)
        for name, data, mime in (
                ("tag_vocabulary.parquet", OUT_VOCAB_PQ.read_bytes(), "application/octet-stream"),
                ("tag_vocabulary.csv", OUT_VOCAB.read_bytes(), "text/csv"),
                ("tag_taxonomy.csv", OUT_TAXONOMY.read_bytes(), "text/csv"),
                ("tag_aliases.parquet", al_buf.getvalue(), "application/octet-stream")):
            drive_upload_with_retry(svc, f"{DRIVE_INDEX}/{name}", root, data, mime)
            print(f"uploaded {DRIVE_INDEX}/{name}")

if __name__ == "__main__":
    main()
