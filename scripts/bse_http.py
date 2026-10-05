"""bse_http.py — HTTP session that BSE still answers.

From ~2026-09-24 BSE's edge (Akamai) rejects python-requests by its TLS/HTTP
fingerprint: every api.bseindia.com call returns 403 "Access Denied", from CI
and from a home PC alike, whatever headers are sent. The same calls made by
curl_cffi impersonating Chrome return 200 — the price API, the scrip list,
the CorporateAction API and the official bhavcopy CSV (verified 2026-10-03).

curl_cffi is already installed wherever yfinance is (it is yfinance's own HTTP
layer). It is deliberately NOT in the root requirements.txt (Streamlit Cloud).
If it is missing, this falls back to a plain requests.Session so callers keep
working exactly as before (and get the 403s they got before).
"""
from __future__ import annotations

BSE_HOME = "https://www.bseindia.com/"


def bse_session(headers: dict | None = None, prime: bool = True):
    """A session for bseindia.com / api.bseindia.com. `headers` are added to
    every request. `prime` makes one homepage GET first so the session holds the
    site's cookies, as a browser would. Not thread-safe: one per thread."""
    try:
        from curl_cffi import requests as cr
        s = cr.Session(impersonate="chrome")
    except Exception:                       # curl_cffi unavailable
        import requests
        s = requests.Session()
    if headers:
        s.headers.update(headers)
    if prime:
        try:
            s.get(BSE_HOME, timeout=20)
        except Exception:
            pass                            # priming is best-effort
    return s
