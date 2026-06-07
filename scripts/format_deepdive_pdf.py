"""
format_deepdive_pdf.py — convert a deep-dive .md report to a beautifully formatted PDF.

Uses markdown -> styled HTML -> PDF via weasyprint.
Tables are rendered with alternating row shading, bold headers, and borders.

Usage:
    python scripts/format_deepdive_pdf.py <report.md> [--out <output.pdf>]

Or import:
    from format_deepdive_pdf import md_to_pdf
    pdf_bytes = md_to_pdf(md_text, company_name, symbol, isin)
"""
from __future__ import annotations
import os, io, re, sys, argparse, datetime as dt

# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #999;
    }
}

body {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 9pt;
    line-height: 1.55;
    color: #1a1a2e;
    background: #fff;
}

/* ── Cover page ─────────────────────────────────────────────────────── */
.cover {
    text-align: center;
    padding-top: 60mm;
    page-break-after: always;
}
.cover h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: 1px;
    margin-bottom: 6mm;
}
.cover .company {
    font-size: 18pt;
    font-weight: 600;
    color: #2c3e7a;
    margin-bottom: 3mm;
}
.cover .meta {
    font-size: 10pt;
    color: #666;
    margin-top: 2mm;
}
.cover .divider {
    width: 60mm;
    height: 2px;
    background: #2c3e7a;
    margin: 8mm auto;
}

/* ── Headings ────────────────────────────────────────────────────────── */
h1 {
    font-size: 14pt;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 2px solid #2c3e7a;
    padding-bottom: 2mm;
    margin-top: 8mm;
    margin-bottom: 3mm;
    page-break-after: avoid;
}
h2 {
    font-size: 11pt;
    font-weight: 700;
    color: #2c3e7a;
    margin-top: 5mm;
    margin-bottom: 2mm;
    page-break-after: avoid;
}
h3 {
    font-size: 10pt;
    font-weight: 600;
    color: #1a1a2e;
    margin-top: 4mm;
    margin-bottom: 1.5mm;
    page-break-after: avoid;
}

/* ── Body text ───────────────────────────────────────────────────────── */
p { margin-bottom: 2mm; }
strong { font-weight: 700; }
em { font-style: italic; }

ul, ol {
    margin-left: 5mm;
    margin-bottom: 2mm;
}
li { margin-bottom: 1mm; }

/* ── Tables ──────────────────────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 3mm 0 4mm 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}
thead tr {
    background-color: #2c3e7a;
    color: #ffffff;
}
thead th {
    padding: 2.5mm 3mm;
    text-align: left;
    font-weight: 700;
    border: 1px solid #1a2a6c;
    white-space: nowrap;
}
tbody tr:nth-child(even) {
    background-color: #f0f3ff;
}
tbody tr:nth-child(odd) {
    background-color: #ffffff;
}
tbody td {
    padding: 2mm 3mm;
    border: 1px solid #ccd3f0;
    vertical-align: top;
}
tbody tr:hover {
    background-color: #e6eaff;
}

/* ── Code / pre ──────────────────────────────────────────────────────── */
code {
    font-family: 'Courier New', monospace;
    font-size: 8pt;
    background: #f4f4f8;
    padding: 0 1mm;
    border-radius: 1px;
}
pre {
    background: #f4f4f8;
    padding: 3mm;
    font-size: 7.5pt;
    overflow-x: auto;
    margin: 2mm 0;
    page-break-inside: avoid;
}

/* ── Blockquote / highlights ─────────────────────────────────────────── */
blockquote {
    border-left: 3px solid #2c3e7a;
    padding-left: 4mm;
    color: #444;
    margin: 2mm 0;
}

/* ── HR ──────────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #ccd3f0;
    margin: 4mm 0;
}

/* ── Footer watermark ────────────────────────────────────────────────── */
.report-footer {
    font-size: 7.5pt;
    color: #aaa;
    text-align: center;
    margin-top: 6mm;
    border-top: 1px solid #eee;
    padding-top: 2mm;
}
"""

# --------------------------------------------------------------------------

def _make_cover(name: str, symbol: str, isin: str, coverage: str = "") -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cov_line = f"<p class='meta'>Coverage: {coverage}</p>" if coverage else ""
    return f"""
<div class="cover">
  <h1>DEEP DIVE REPORT</h1>
  <div class="divider"></div>
  <p class="company">{name}</p>
  <p class="meta">{symbol} &nbsp;·&nbsp; {isin}</p>
  <p class="meta">Generated {now}</p>
  {cov_line}
</div>
"""


def _normalise_tables(md_text: str) -> str:
    """python-markdown's `tables` extension only fires when a pipe-table is a
    standalone block (blank line before AND after). Models often glue a table to
    the heading above it. Insert the required blank lines so every table renders."""
    out, lines = [], md_text.splitlines()
    for i, line in enumerate(lines):
        is_tbl = line.lstrip().startswith("|")
        prev_tbl = out and out[-1].lstrip().startswith("|")
        if is_tbl and out and out[-1].strip() and not prev_tbl:
            out.append("")                       # blank line before table starts
        if (not is_tbl) and prev_tbl and line.strip():
            out.append("")                       # blank line after table ends
        out.append(line)
    return "\n".join(out)


def _md_to_html_body(md_text: str) -> str:
    md_text = _normalise_tables(md_text)
    try:
        import markdown as md_lib
        # NOTE: no `nl2br` — it converts table newlines to <br> and breaks the
        # tables extension. Paragraph spacing is handled by CSS instead.
        return md_lib.markdown(
            md_text,
            extensions=["tables", "fenced_code", "sane_lists"],
        )
    except ImportError:
        # minimal fallback — convert markdown tables manually
        lines = []
        in_table = False
        table_rows: list[str] = []

        def flush_table():
            nonlocal in_table, table_rows
            if not table_rows:
                return ""
            html = ["<table><thead>"]
            header_done = False
            for row in table_rows:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if all(re.match(r"^[-:]+$", c) for c in cells):
                    if not header_done:
                        html.append("</tr></thead><tbody>")
                        header_done = True
                    continue
                tag = "th" if not header_done else "td"
                html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            html.append("</tbody></table>")
            table_rows = []
            in_table = False
            return "\n".join(html)

        result = []
        for line in md_text.splitlines():
            if line.strip().startswith("|"):
                if not in_table:
                    in_table = True
                table_rows.append(line)
            else:
                if in_table:
                    result.append(flush_table())
                stripped = line.strip()
                if stripped.startswith("### "):
                    result.append(f"<h3>{stripped[4:]}</h3>")
                elif stripped.startswith("## "):
                    result.append(f"<h2>{stripped[3:]}</h2>")
                elif stripped.startswith("# "):
                    result.append(f"<h1>{stripped[2:]}</h1>")
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    result.append(f"<li>{stripped[2:]}</li>")
                elif stripped == "---" or stripped == "***":
                    result.append("<hr>")
                elif stripped:
                    result.append(f"<p>{stripped}</p>")
                else:
                    result.append("")
        if in_table:
            result.append(flush_table())
        return "\n".join(result)


def _build_full_html(name: str, symbol: str, isin: str,
                     md_text: str, coverage: str = "") -> str:
    cover = _make_cover(name, symbol, isin, coverage)
    body  = _md_to_html_body(md_text)
    footer = (f"<div class='report-footer'>signals-india · Deep Dive · "
              f"{name} ({symbol}) · {dt.date.today()}</div>")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>{CSS}</style>
</head>
<body>
{cover}
{body}
{footer}
</body>
</html>"""


def md_to_pdf(md_text: str, name: str = "", symbol: str = "",
              isin: str = "", coverage: str = "") -> bytes:
    try:
        from weasyprint import HTML as WP_HTML
    except ImportError:
        sys.exit("weasyprint not installed. Run: pip install weasyprint")

    html = _build_full_html(
        name or "Company", symbol or "", isin or "", md_text, coverage)
    buf = io.BytesIO()
    WP_HTML(string=html).write_pdf(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .md report file")
    ap.add_argument("--out",  help="output .pdf path (default: same dir, .pdf extension)")
    ap.add_argument("--name",   default="")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--isin",   default="")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        md_text = f.read()

    m = re.search(r"#\s+Deep Dive.*?([A-Z][^(]+)\s+\((\w+)\s*/\s*(INE\w+)\)", md_text)
    name   = args.name   or (m.group(1).strip() if m else os.path.splitext(os.path.basename(args.input))[0])
    symbol = args.symbol or (m.group(2) if m else "")
    isin   = args.isin   or (m.group(3) if m else "")

    pdf_bytes = md_to_pdf(md_text, name, symbol, isin)
    out_path = args.out or os.path.splitext(args.input)[0] + ".pdf"
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
