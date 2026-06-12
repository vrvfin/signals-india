"""
_t4_store.py — shared Drive/local storage layer for the Phase 3 T4/T5/T7
producer scripts (valuation, fraud_risk, scorecard, investigative_fraud,
catalyst_notes, fraud_tracker).

Extracted 2026-06-12 (code review, user-approved): six scripts carried
byte-identical copies of this class; one copy had already drifted (missing
read_parquet) once. One definition, six importers.

Pattern: every method takes path_parts like ["company_repo", "_index", "x.parquet"].
local=True reads/writes under local_dir instead of Drive (offline testing).
Uses ONLY the shared _extractor_base helpers (CLAUDE.md global rule #4).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)


class Store:
    def __init__(self, local: bool, local_dir: Path | None):
        self.local = local
        self.local_dir = local_dir
        self.drive = None
        if not local:
            self.drive = get_drive()
            self.root = os.environ["GDRIVE_FOLDER_ID"]

    def _folder(self, parts):
        fid = self.root
        for p in parts:
            fid = get_or_create_subfolder(self.drive, fid, p)
        return fid

    def read_parquet(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return pd.read_parquet(fp) if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return pd.read_parquet(io.BytesIO(download_bytes(self.drive, fid)))

    def read_csv(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return pd.read_csv(fp) if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return pd.read_csv(io.BytesIO(download_bytes(self.drive, fid)))

    def read_text(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return fp.read_text(encoding="utf-8") if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return download_bytes(self.drive, fid).decode("utf-8", errors="replace")

    def write_df(self, path_parts, df: pd.DataFrame):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fp, index=False) if name.endswith(".csv") else df.to_parquet(fp, index=False)
            return
        if name.endswith(".csv"):
            data = df.to_csv(index=False).encode("utf-8")
            mime = "text/csv"
        else:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            data = buf.getvalue()
            mime = "application/octet-stream"
        folder_id = self._folder(folder)
        existing = find_file(self.drive, folder_id, name)
        upload_bytes(self.drive, folder_id, name, data, mime, existing_id=existing)

    def write_text(self, path_parts, text: str):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(text, encoding="utf-8")
            return
        folder_id = self._folder(folder)
        existing = find_file(self.drive, folder_id, name)
        upload_bytes(self.drive, folder_id, name, text.encode("utf-8"),
                     "text/markdown", existing_id=existing)
