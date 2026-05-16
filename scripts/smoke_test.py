"""
Stage 1 smoke test.

Verifies:
  1. Python environment OK
  2. .env loaded
  3. Service-account JSON readable
  4. Google Drive authentication works
  5. Target Drive folder is reachable
  6. We can write a file into it

Run from the project root, inside the activated `signals-india` conda env:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload


SCOPES = ["https://www.googleapis.com/auth/drive"]


def check(label: str, ok: bool, detail: str = "") -> None:
    """Print a check line and exit if it failed."""
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    print("Stage 1 smoke test")
    print("-" * 40)

    # 1. Python OK
    check("Python runtime", sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}")

    # 2. .env loaded
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    check(".env file present", env_path.exists(), str(env_path))
    load_dotenv(env_path)

    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    client_secret_path = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_PATH", "").strip()
    token_path = os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", "").strip()
    check("GDRIVE_FOLDER_ID set", bool(folder_id))
    check("GDRIVE_OAUTH_CLIENT_SECRET_PATH set", bool(client_secret_path))
    check("GDRIVE_OAUTH_TOKEN_PATH set", bool(token_path))

    # 3. OAuth client secret readable
    cs_file = Path(client_secret_path)
    check("OAuth client_secret.json exists", cs_file.exists(), client_secret_path)

    # 4. Authenticate (loads stored token, or runs browser consent on first run)
    token_file = Path(token_path)
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    check("Authenticated to Google Drive", True, "via OAuth user delegation")

    # 5. Folder reachable
    folder = drive.files().get(fileId=folder_id, fields="id,name,mimeType").execute()
    is_folder = folder.get("mimeType") == "application/vnd.google-apps.folder"
    check("Drive folder reachable", is_folder, f"{folder.get('name')} ({folder_id})")

    # 6. Write test file
    fname = f"stage1_smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    body = f"Smoke test passed at {datetime.now().isoformat()}.\n".encode()
    media = MediaInMemoryUpload(body, mimetype="text/plain")
    meta = {"name": fname, "parents": [folder_id]}
    created = drive.files().create(body=meta, media_body=media, fields="id,name").execute()
    check("Wrote test file to Drive", bool(created.get("id")), created.get("name"))

    print("-" * 40)
    print("All systems go: Python OK, Drive OK, write OK.")
    print(f"Verify on drive.google.com — you should see: {fname}")


if __name__ == "__main__":
    main()