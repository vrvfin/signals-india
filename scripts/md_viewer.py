"""
md_viewer.py — local markdown viewer with proper table rendering.

Converts any .md file to HTML and opens it in your default browser.
Tables, code blocks, and all GFM formatting render correctly.

Usage:
    python scripts/md_viewer.py path/to/file.md
    python scripts/md_viewer.py "C:/Downloads/concall_29_may2026.md"

Requirements (already in scripts/requirements.txt):
    pip install markdown

No Drive access needed — reads local files only.
"""

from __future__ import annotations

import sys
import tempfile
import webbrowser
from pathlib import Path

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 24px 32px;
  line-height: 1.65; color: #1a1a1a; background: #fff;
}
h1 { border-bottom: 2px solid #2c3e50; padding-bottom: 8px; font-size: 1.8em; }
h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 28px; }
h3 { color: #2c3e50; margin-top: 20px; }

/* ── Tables ── */
table {
  border-collapse: collapse; width: 100%;
  margin: 1em 0; font-size: 0.9em;
}
th {
  background: #2c3e50; color: #fff;
  padding: 8px 12px; text-align: left;
  font-weight: 600;
}
td { border: 1px solid #dce1e7; padding: 7px 11px; vertical-align: top; }
tr:nth-child(even) td { background: #f8f9fa; }
tr:hover td { background: #eaf4fb; }

/* ── Code ── */
code {
  background: #f0f3f4; padding: 2px 5px;
  border-radius: 3px; font-size: 0.88em; font-family: "Consolas", monospace;
}
pre {
  background: #f0f3f4; padding: 14px 16px; border-radius: 5px;
  overflow-x: auto; line-height: 1.4;
}
pre code { background: none; padding: 0; }

/* ── Blockquotes ── */
blockquote {
  border-left: 4px solid #3498db; margin: 0; padding: 4px 16px;
  color: #555; background: #eaf4fb;
}

/* ── HR ── */
hr { border: none; border-top: 1px solid #dce1e7; margin: 20px 0; }

/* ── TOC-style section links ── */
a { color: #2980b9; }
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert_md_to_html(md_text: str, title: str = "Document") -> str:
    try:
        import markdown as md_lib
        body = md_lib.markdown(
            md_text,
            extensions=["tables", "fenced_code", "nl2br", "toc"],
        )
    except ImportError:
        # Fallback: basic <pre> wrap if markdown package not installed
        escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = f"<pre style='white-space:pre-wrap'>{escaped}</pre>"
        body += ("<p style='color:orange'>"
                 "<b>Tip:</b> Install <code>markdown</code> for proper table rendering: "
                 "<code>pip install markdown</code></p>")
    return HTML_TEMPLATE.format(title=title, css=CSS, body=body)


def view_file(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p}")
        sys.exit(1)

    text = p.read_text(encoding="utf-8", errors="replace")
    html = convert_md_to_html(text, title=p.stem)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        tmp_path = f.name

    webbrowser.open(f"file:///{Path(tmp_path).as_posix()}")
    print(f"Opened in browser: {tmp_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    view_file(sys.argv[1])
