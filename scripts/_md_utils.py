"""
_md_utils.py — Shared markdown post-processing utilities.

Used by fetch_latest_concall.py and fetch_company_intel.py to fix
LLM-generated markdown for Obsidian rendering.
"""

from __future__ import annotations
import re


def fix_markdown_for_obsidian(text: str) -> str:
    """Fix LLM-generated markdown so Obsidian renders tables correctly.

    Problems addressed (in order):

    1. SPLIT ROWS — Gemini wraps wide table rows across 2+ lines.
       e.g.  | col1 | col2 | col3 | col4 | NEAR_TERM |
             Explicit | Yes | 37.5 | FY27 | Panjak |
       Fix: join continuation lines back onto the parent row.
       Heuristic: continuation = line that ends with | but doesn't start with |,
       immediately following a | line.

    2. LEADING WHITESPACE — Obsidian treats lines with 4+ leading spaces as
       code blocks even if they start with |.
       Fix: strip all leading whitespace from | rows.

    3. BLANK LINE BEFORE TABLE — Obsidian sometimes fails to render a table
       that immediately follows a text paragraph with no blank line.
       Fix: insert a blank line before the first row of each new table block.

    4. FENCED TABLE — Gemini occasionally wraps tables inside ``` fences.
       Fix: unwrap fences whose content is purely table rows.

    5. LINE ENDINGS — normalise \\r\\n and \\r to \\n.
    """
    # ── 0. Normalise line endings ────────────────────────────────────────────
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ── 1. Unwrap tables inside code fences ─────────────────────────────────
    def _unwrap_table_fence(m: re.Match) -> str:
        inner = m.group(1)
        if any(l.lstrip().startswith('|') for l in inner.splitlines()):
            return inner          # remove fence, keep table rows
        return m.group(0)        # leave real code blocks untouched

    text = re.sub(r'```[^\n]*\n(.*?)```', _unwrap_table_fence,
                  text, flags=re.DOTALL)

    # ── 2. Join split table rows ─────────────────────────────────────────────
    # A row continuation: does NOT start with |, DOES end with |, has content,
    # and immediately follows a | line.
    lines = text.split('\n')
    joined: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('|'):
            # Gobble continuation lines
            while i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith('|') and nxt.endswith('|'):
                    line = line.rstrip() + ' ' + nxt
                    i += 1
                else:
                    break
            joined.append(line)
        else:
            joined.append(line)
        i += 1

    # ── 3. Strip leading whitespace + insert blank lines before tables ───────
    out: list[str] = []
    in_fence = False
    for line in joined:
        if re.match(r'^```', line):
            in_fence = not in_fence
            out.append(line)
            continue

        if not in_fence:
            stripped = line.lstrip()
            if stripped.startswith('|'):
                line = stripped  # kill leading whitespace
                # Insert blank line before a new table block
                last_nonempty = next(
                    (l for l in reversed(out) if l.strip()), None)
                if (last_nonempty
                        and not last_nonempty.lstrip().startswith('|')
                        and out and out[-1].strip()):
                    out.append('')

        out.append(line)

    return '\n'.join(out)
