# session_store.py
from __future__ import annotations

import os
import json
import hmac
import time
import secrets
import threading
from hashlib import sha256
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


@dataclass
class Session:
    data: Dict[str, Any] = field(default_factory=dict)
    expires_at: float = 0.0  # unix timestamp


class SessionStore:
    """
    Minimal, signed session store with optional file persistence.

    Quick start:
        store = SessionStore()  # secret auto-generated or from env SECRET_KEY
        sid = store.new({"user_id": 123}, ttl=3600)
        data = store.get(sid)
        store.set(sid, {"step": "arms"}, ttl=900)
        store.touch(sid, ttl=3600)
        store.delete(sid)

    Cookie helpers (Flask / Starlette-like):
        sid = store.read_cookie(request.cookies.get(store.cookie_name, ""))
        store.set_cookie(response, sid)
        store.clear_cookie(response)
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        persist_dir: Optional[str] = ".sessions",
        default_ttl: int = 60 * 60 * 24,  # 1 day
        cookie_name: str = "session_id",
    ) -> None:
        # Prefer explicit arg, then env, else generate a fresh random key
        sk = secret_key or os.getenv("SECRET_KEY")
        if not sk:
            # For dev/hackathon this is fine; in prod set SECRET_KEY in the environment
            sk = secrets.token_hex(32)

        self._secret = sk.encode("utf-8")
        self.default_ttl = int(default_ttl)
        self.cookie_name = cookie_name

        self._lock = threading.RLock()
        self._sessions: Dict[str, Session] = {}

        # Optional persistence directory
        self._persist_dir = persist_dir
        if self._persist_dir:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._load_all_from_disk()

    # --------- ID + signing ---------
    def _sign(self, sid: str) -> str:
        return hmac.new(self._secret, sid.encode("utf-8"), sha256).hexdigest()

    def _now(self) -> float:
        return time.time()

    def _cookie_value(self, sid: str) -> str:
        return f"{sid}.{self._sign(sid)}"

    def read_cookie(self, cookie_value: str | None) -> Optional[str]:
        """
        Validate a cookie value and return the session id if valid; otherwise None.
        Cookie value format: "<sid>.<signature>".
        """
        if not cookie_value or "." not in cookie_value:
            return None
        sid, sig = cookie_value.rsplit(".", 1)
        expected = self._sign(sid)
        if not hmac.compare_digest(sig, expected):
            return None
        return sid

    # --------- Core API ---------
    def new(self, data: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> str:
        """Create a new session and return its id (NOT the cookie value)."""
        sid = secrets.token_urlsafe(32)
        expires = self._now() + float(ttl or self.default_ttl)
        with self._lock:
            self._sessions[sid] = Session(data=dict(data or {}), expires_at=expires)
            self._save_one_to_disk(sid)
        return sid

    def get(self, sid: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get the session data dict if it exists and is not expired; else None."""
        if not sid:
            return None
        with self._lock:
            s = self._sessions.get(sid)
            if not s:
                s = self._load_one_from_disk(sid)
                if s:
                    self._sessions[sid] = s
            if not s:
                return None
            if s.expires_at <= self._now():
                self._delete_unlocked(sid)
                return None
            return s.data

    def set(self, sid: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Replace session data and (optionally) extend TTL. Returns True if session existed."""
        with self._lock:
            s = self._sessions.get(sid) or self._load_one_from_disk(sid)
            if not s:
                return False
            s.data = dict(data)
            if ttl is not None:
                s.expires_at = self._now() + float(ttl)
            self._sessions[sid] = s
            self._save_one_to_disk(sid)
            return True

    def touch(self, sid: str, ttl: Optional[int] = None) -> bool:
        """Extend session TTL without changing data."""
        with self._lock:
            s = self._sessions.get(sid) or self._load_one_from_disk(sid)
            if not s:
                return False
            s.expires_at = self._now() + float(ttl or self.default_ttl)
            self._sessions[sid] = s
            self._save_one_to_disk(sid)
            return True

    def delete(self, sid: str) -> bool:
        """Delete a session by id. Returns True if it existed."""
        with self._lock:
            return self._delete_unlocked(sid)

    def _delete_unlocked(self, sid: str) -> bool:
        existed = sid in self._sessions
        self._sessions.pop(sid, None)
        if self._persist_dir:
            path = self._path(sid)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return existed

    def cleanup(self) -> int:
        """Remove expired sessions. Returns number removed."""
        now = self._now()
        removed = 0
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.expires_at <= now]
            for sid in expired:
                if self._delete_unlocked(sid):
                    removed += 1
            # Also clean stray files on disk
            if self._persist_dir:
                for fname in os.listdir(self._persist_dir):
                    if not fname.endswith(".json"):
                        continue
                    sid = fname[:-5]
                    if sid not in self._sessions:
                        try:
                            s = self._load_one_from_disk(sid)
                        except Exception:
                            s = None
                        if (s is None) or (s.expires_at <= now):
                            try:
                                os.remove(self._path(sid))
                                removed += 1
                            except OSError:
                                pass
        return removed

    # --------- Persistence (optional) ---------
    def _path(self, sid: str) -> str:
        assert self._persist_dir is not None
        return os.path.join(self._persist_dir, f"{sid}.json")

    def _save_one_to_disk(self, sid: str) -> None:
        if not self._persist_dir:
            return
        s = self._sessions.get(sid)
        if not s:
            return
        blob = {"expires_at": s.expires_at, "data": s.data}
        tmp = self._path(sid) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, self._path(sid))

    def _load_one_from_disk(self, sid: str) -> Optional[Session]:
        if not self._persist_dir:
            return None
        path = self._path(sid)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            return Session(
                data=dict(blob.get("data", {})),
                expires_at=float(blob.get("expires_at", 0)),
            )
        except Exception:
            return None

    def _load_all_from_disk(self) -> None:
        assert self._persist_dir is not None
        for fname in os.listdir(self._persist_dir):
            if not fname.endswith(".json"):
                continue
            sid = fname[:-5]
            s = self._load_one_from_disk(sid)
            if s:
                self._sessions[sid] = s

    # --------- Cookie helpers ---------
    def cookie_value_for(self, sid: str) -> str:
        """Return the signed cookie value for a session id."""
        return self._cookie_value(sid)

    def set_cookie(
        self,
        response: Any,
        sid: Optional[str],
        *,
        max_age: Optional[int] = None,
        path: str = "/",
        secure: bool = True,
        http_only: bool = True,
        same_site: str = "Lax",
        domain: Optional[str] = None,
    ) -> None:
        """
        Set (or refresh) the session cookie on a response.

        Works with:
          - Flask Response: response.set_cookie(...)
          - Starlette/FastAPI Response: response.set_cookie(...)
        """
        if sid is None:
            return
        cookie_val = self._cookie_value(sid)
        max_age = int(max_age or self.default_ttl)

        # Flask/Starlette compatible
        if hasattr(response, "set_cookie"):
            response.set_cookie(
                key=self.cookie_name,
                value=cookie_val,
                max_age=max_age,
                path=path,
                secure=secure,
                httponly=http_only,  # Flask param is 'httponly'
                samesite=same_site,
                domain=domain,
            )
        else:
            # Fallback: append Set-Cookie header manually
            header_val = (
                f"{self.cookie_name}={cookie_val}; Path={path}; Max-Age={max_age}; "
                f"{'Secure; ' if secure else ''}HttpOnly; SameSite={same_site}"
            )
            if domain:
                header_val += f"; Domain={domain}"
            existing = getattr(response, "headers", None)
            if isinstance(existing, dict):
                response.headers.setdefault("Set-Cookie", header_val)
            else:
                try:
                    response.headers.append(("Set-Cookie", header_val))
                except Exception:
                    # last resort
                    try:
                        response.headers["Set-Cookie"] = header_val
                    except Exception:
                        pass

    def clear_cookie(self, response: Any, path: str = "/", domain: Optional[str] = None) -> None:
        """Delete the session cookie on client."""
        if hasattr(response, "delete_cookie"):
            response.delete_cookie(self.cookie_name, path=path, domain=domain)
            return
        # Fallback header
        header_val = f"{self.cookie_name}=; Path={path}; Expires=Thu, 01 Jan 1970 00:00:00 GMT"
        if domain:
            header_val += f"; Domain={domain}"
        try:
            response.headers.append(("Set-Cookie", header_val))
        except Exception:
            try:
                response.headers["Set-Cookie"] = header_val
            except Exception:
                pass
