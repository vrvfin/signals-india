"""
scripts/screener_client.py — Screener.in cookie-authenticated client.

Detects cookie expiry and prints clear refresh instructions instead of crashing.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

BASE_URL = "https://www.screener.in"
RATE_LIMIT_SEC = 1.0     # polite default


class CookieExpiredError(Exception):
    pass


def _print_cookie_expired_banner():
    print("\n" + "=" * 64)
    print("  SCREENER.IN COOKIE EXPIRED  —  refresh required")
    print("=" * 64)
    print("  1. Open https://www.screener.in/ in Chrome and ensure you're logged in.")
    print("  2. Press F12 → Application → Cookies → screener.in")
    print("  3. Copy the value of the cookie named 'sessionid'.")
    print("  4. Open D:\\EMA_Screener\\claude\\signals-india\\.env and replace the")
    print("     SCREENER_SESSION_COOKIE=... line with the new value.")
    print("  5. On GitHub: Settings → Secrets → Actions → SCREENER_SESSION_COOKIE → Update")
    print("  6. Re-run the script.")
    print("=" * 64 + "\n")


def _parse_number(s: str | None):
    """Convert Screener-style 'X,Y' / 'X%' / '-' strings to float, or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("%", "").replace("₹", "")
    if s in ("", "--", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


class ScreenerClient:
    def __init__(self, cookie: str | None = None, rate_limit_sec: float = RATE_LIMIT_SEC):
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        cookie = cookie or os.environ.get("SCREENER_SESSION_COOKIE", "").strip()
        if not cookie:
            raise RuntimeError(
                "SCREENER_SESSION_COOKIE not set. Add it to your .env "
                "(see screener_client._print_cookie_expired_banner for steps)."
            )
        self.session = requests.Session()
        self.session.cookies.set("sessionid", cookie, domain=".screener.in")
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.rate_limit_sec = rate_limit_sec
        self._last_call = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self._last_call = time.time()

    def _check_auth(self, r: requests.Response, symbol: str):
        # Screener redirects unauthenticated users to /login/
        if r.url and "/login" in r.url.lower():
            _print_cookie_expired_banner()
            raise CookieExpiredError(f"Redirected to login while fetching {symbol}")
        if r.status_code in (401, 403):
            _print_cookie_expired_banner()
            raise CookieExpiredError(f"HTTP {r.status_code} while fetching {symbol}")

    def fetch_company(self, symbol: str) -> BeautifulSoup | None:
        """Return the parsed HTML of a company page. Tries consolidated then standalone."""
        for variant in ("consolidated/", ""):
            url = f"{BASE_URL}/company/{symbol}/{variant}"
            self._wait()
            try:
                r = self.session.get(url, timeout=30, allow_redirects=True)
            except requests.RequestException:
                continue
            if r.status_code == 404:
                continue
            self._check_auth(r, symbol)
            if r.status_code == 200 and "company" in r.url.lower():
                return BeautifulSoup(r.text, "lxml")
        return None

    # ---------- Parsers ----------

    def parse_top_ratios(self, soup: BeautifulSoup) -> dict:
        """The top-of-page ratio strip: Market Cap, P/E, Book Value, ROCE, etc."""
        out = {}
        ul = soup.find("ul", id="top-ratios")
        if not ul:
            return out
        for li in ul.find_all("li"):
            name_el = li.find("span", class_="name")
            val_el = li.find("span", class_="number")
            if not (name_el and val_el):
                # Fallback: whole text "Name: value"
                txt = li.get_text(separator=" ", strip=True)
                if ":" in txt:
                    name, val = txt.split(":", 1)
                    out[name.strip()] = _parse_number(val)
                continue
            out[name_el.get_text(strip=True)] = _parse_number(val_el.get_text(strip=True))
        return out

    def parse_table_section(self, soup: BeautifulSoup, section_id: str) -> dict:
        """Generic parser for #quarters, #profit-loss, #balance-sheet, #cash-flow.
        Returns {"headers": [...col labels...], "rows": {row_label: [values]}}."""
        section = soup.find("section", id=section_id)
        if not section:
            return {"headers": [], "rows": {}}
        table = section.find("table", class_=re.compile(r"data-table"))
        if not table:
            return {"headers": [], "rows": {}}
        headers = []
        thead = table.find("thead")
        if thead:
            ths = thead.find_all("th")
            headers = [th.get_text(strip=True) for th in ths[1:]]
        rows = {}
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                label = tds[0].get_text(strip=True).rstrip("+").strip()
                values = [_parse_number(td.get_text(strip=True)) for td in tds[1:]]
                if label:
                    rows[label] = values
        return {"headers": headers, "rows": rows}

    def parse_growth_panels(self, soup: BeautifulSoup) -> dict:
        """Compounded growth panels (Sales/Profit growth 1Y/3Y/5Y/10Y)."""
        out = {}
        for panel in soup.find_all("table", class_=re.compile(r"ranges-table")):
            heading = panel.find("th")
            if not heading:
                continue
            key = heading.get_text(strip=True)  # e.g. "Compounded Sales Growth"
            sub = {}
            for tr in panel.find_all("tr")[1:]:
                cells = tr.find_all("td")
                if len(cells) >= 2:
                    sub[cells[0].get_text(strip=True)] = _parse_number(cells[1].get_text(strip=True))
            out[key] = sub
        return out

    def extract_summary(self, symbol: str, soup: BeautifulSoup) -> dict:
        """One-row summary with key numbers used by CANSLIM + PEAD."""
        top = self.parse_top_ratios(soup)
        qtr = self.parse_table_section(soup, "quarters")
        ann = self.parse_table_section(soup, "profit-loss")
        growth = self.parse_growth_panels(soup)

        def _qtr_row(name_keys):
            for k in name_keys:
                if k in qtr["rows"]:
                    return qtr["rows"][k]
            return []

        eps_qtr = _qtr_row(["EPS in Rs", "EPS", "EPS in ₹"])
        np_qtr = _qtr_row(["Net Profit", "Net Profit +"])
        sales_qtr = _qtr_row(["Sales", "Revenue", "Revenue +", "Sales +"])

        latest_q_eps = eps_qtr[-1] if eps_qtr else None
        yoy_q_eps = eps_qtr[-5] if len(eps_qtr) >= 5 else None
        q_eps_yoy_pct = (
            (latest_q_eps - yoy_q_eps) / abs(yoy_q_eps) * 100
            if (latest_q_eps is not None and yoy_q_eps not in (None, 0))
            else None
        )

        eps_yrs = ann["rows"].get("EPS in Rs", []) or ann["rows"].get("EPS", [])
        latest_ann_eps = eps_yrs[-1] if eps_yrs else None
        prev_ann_eps = eps_yrs[-2] if len(eps_yrs) >= 2 else None
        ann_eps_yoy_pct = (
            (latest_ann_eps - prev_ann_eps) / abs(prev_ann_eps) * 100
            if (latest_ann_eps is not None and prev_ann_eps not in (None, 0))
            else None
        )

        sales_growth = growth.get("Compounded Sales Growth", {})
        profit_growth = growth.get("Compounded Profit Growth", {})

        return {
            "symbol": symbol,
            "market_cap_cr": top.get("Market Cap"),
            "pe": top.get("Stock P/E") or top.get("P/E"),
            "book_value": top.get("Book Value"),
            "roce_pct": top.get("ROCE"),
            "roe_pct": top.get("ROE"),
            "debt_to_equity": top.get("Debt to equity") or top.get("Debt / Equity"),
            "dividend_yield_pct": top.get("Dividend Yield"),
            "promoter_holding_pct": top.get("Promoter holding"),
            "latest_quarter_label": qtr["headers"][-1] if qtr["headers"] else None,
            "latest_quarter_eps": latest_q_eps,
            "q_eps_yoy_pct": q_eps_yoy_pct,
            "q_eps_last_4q": eps_qtr[-4:] if eps_qtr else None,
            "q_sales_last_4q": sales_qtr[-4:] if sales_qtr else None,
            "q_netprofit_last_4q": np_qtr[-4:] if np_qtr else None,
            "ann_eps_yoy_pct": ann_eps_yoy_pct,
            "sales_growth_1y": sales_growth.get("1 Year:"),
            "sales_growth_3y": sales_growth.get("3 Years:"),
            "sales_growth_5y": sales_growth.get("5 Years:"),
            "profit_growth_1y": profit_growth.get("1 Year:"),
            "profit_growth_3y": profit_growth.get("3 Years:"),
        }
