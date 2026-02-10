"""
services/session.py
───────────────────
User registry (email → UUID) + session folder management.
No authentication service — just a static email-based identity.

User registry:
    data/users.json   ->  { "alice@co.com": {"uid": "a1b2c3d4", "created_at": ..., "last_login": ...}, ... }

Folder layout:
    data/
    ├── users.json                  # email → uid mapping
    └── {uid}/                      # 32-char hex UUID (no dashes)
        └── {session_id}/
            ├── upload/
            ├── centerline.json
            ├── mesh.json
            ├── segments.json
            ├── segments.pkl
            ├── embeddings.npy
            └── chat.jsonl

Frontend flow:
    1. Check localStorage for email
    2. If missing → redirect to login page
    3. POST /auth/login {email} → returns {uid, sessions, is_new}
    4. Store email + uid in localStorage
    5. All subsequent requests use uid (not email)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading


DATA_ROOT = Path("data")
USERS_FILE = DATA_ROOT / "users.json"
_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════
# USER REGISTRY (email → UUID)
# ══════════════════════════════════════════════════════════════════════

def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load the user registry. Returns {email: {uid, created_at, last_login}}."""
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Persist the user registry."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _make_uid() -> str:
    """Generate a 32-char hex UUID (no dashes)."""
    return uuid.uuid4().hex


def login_or_create_user(email: str) -> Dict[str, Any]:
    """
    Login flow:
      - If email exists → update last_login, return uid + is_new=False
      - If email doesn't exist → create uid + folder, return uid + is_new=True

    Returns:
        {
            "uid": "a1b2c3d4e5f6...",
            "email": "alice@example.com",
            "is_new": True/False,
            "created_at": "...",
            "last_login": "...",
        }
    """
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Invalid email address")

    with _lock:
        users = _load_users()
        now = datetime.utcnow().isoformat()

        if email in users:
            # Existing user — update last_login
            users[email]["last_login"] = now
            _save_users(users)

            return {
                "uid": users[email]["uid"],
                "email": email,
                "is_new": False,
                "created_at": users[email]["created_at"],
                "last_login": now,
            }
        else:
            # New user — create
            uid = _make_uid()
            users[email] = {
                "uid": uid,
                "created_at": now,
                "last_login": now,
            }
            _save_users(users)

            # Create user folder
            user_dir = DATA_ROOT / uid
            user_dir.mkdir(parents=True, exist_ok=True)

            return {
                "uid": uid,
                "email": email,
                "is_new": True,
                "created_at": now,
                "last_login": now,
            }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Look up a user by email. Returns None if not found."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return None
    u = users[email]
    return {"uid": u["uid"], "email": email, **u}


def get_user_by_uid(uid: str) -> Optional[Dict[str, Any]]:
    """Look up a user by uid. Returns None if not found."""
    users = _load_users()
    for email, u in users.items():
        if u["uid"] == uid:
            return {"uid": uid, "email": email, **u}
    return None


def validate_uid(uid: str) -> bool:
    """Check if a uid exists in the registry."""
    return get_user_by_uid(uid) is not None


# ══════════════════════════════════════════════════════════════════════
# USER FOLDER
# ══════════════════════════════════════════════════════════════════════

def get_user_dir(uid: str) -> Path:
    """Get (and create) the user's data folder."""
    d = DATA_ROOT / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def create_session(uid: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new session folder under the user's directory."""
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]

    session_dir = get_user_dir(uid) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "upload").mkdir(exist_ok=True)

    meta = {
        "session_id": session_id,
        "uid": uid,
        "created_at": datetime.utcnow().isoformat(),
        "path": str(session_dir),
    }

    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def get_session_dir(uid: str, session_id: str) -> Path:
    return DATA_ROOT / uid / session_id


def list_sessions(uid: str) -> List[Dict[str, Any]]:
    """List all sessions for a user (by uid)."""
    user_dir = DATA_ROOT / uid
    if not user_dir.exists():
        return []

    sessions = []
    for d in sorted(user_dir.iterdir()):
        if d.is_dir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                sessions.append(json.loads(meta_path.read_text()))
            else:
                sessions.append({
                    "session_id": d.name,
                    "uid": uid,
                    "path": str(d),
                })
    return sessions


# ══════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════

def append_chat(session_dir: Path, role: str, content: str, extra: Optional[Dict] = None):
    """Append a chat message to the session's chat history."""
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if extra:
        entry.update(extra)

    with open(session_dir / "chat.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_chat(session_dir: Path) -> List[Dict[str, Any]]:
    """Load chat history."""
    chat_path = session_dir / "chat.jsonl"
    if not chat_path.exists():
        return []
    lines = chat_path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]
