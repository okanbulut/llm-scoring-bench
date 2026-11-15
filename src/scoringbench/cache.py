"""SQLite cache of model replies, keyed by a hash of everything that changes the reply."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def request_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")
            self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, reply: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, json.dumps(reply)))
            self._db.commit()
