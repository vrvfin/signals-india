import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Force load environment variables from your local config
load_dotenv()

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})

cookie = os.environ.get("SCREENER_SESSION_COOKIE", "").strip()
if not cookie:
    print("WARNING: SCREENER_SESSION_COOKIE variable is empty inside your environment configuration!")
else:
    s.cookies.set("sessionid", cookie, domain=".screener.in")

r = s.get("https://www.screener.in/results/latest/")
soup = BeautifulSoup(r.text, "lxml")

tables = soup.select("table.data-table")
print(f"Status Code: {r.status_code}")
print(f"Total structured data tables found: {len(tables)}")

if len(tables) > 0:
    print("\nSuccess! Sample data extracted from first table structure:")
    first_table = tables[0]
    for row in first_table.select("tr")[:3]:
        print([cell.get_text(strip=True) for cell in row.select("th, td")])
else:
    print("\nStill returning 0 tables. Double-check your browser session ID value or check if your account hit temporary scraping rate limits.")