"""
SafeMedTwin — Authentication & Per-User Data Store
=====================================================
Simple username/password auth backed by SQLite, so each user can only see
their own predictions and history. Passwords are never stored in plain
text — PBKDF2-HMAC-SHA256 with a random per-user salt (stdlib `hashlib`,
no extra dependency, ~200k iterations per OWASP guidance).

This is appropriate for a capstone-project deployment (single small SQLite
file, no external auth service). It is NOT a substitute for a real
identity provider (OAuth/SSO) or HIPAA-grade access controls in an actual
clinical deployment — see the README note on this.
"""
import sqlite3
import hashlib
import os
import secrets
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.environ.get("SAFEMEDTWIN_DB", os.path.join(os.path.dirname(__file__), "..", "safemedtwin.db"))
PBKDF2_ITERATIONS = 200_000


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                disease TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_band TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, salt, pw_hash, datetime.utcnow().isoformat()),
            )
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def authenticate(username: str, password: str) -> tuple[bool, int | None, str]:
    username = username.strip().lower()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return False, None, "No account with that username."
    candidate = _hash_password(password, row["salt"])
    # constant-time comparison to avoid timing side-channels
    if not secrets.compare_digest(candidate, row["password_hash"]):
        return False, None, "Incorrect password."
    return True, row["id"], "Logged in."


def save_prediction(user_id: int, disease: str, probability: float, risk_band: str, features: dict):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO predictions (user_id, disease, probability, risk_band, features_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, disease, probability, risk_band, json.dumps(features), datetime.utcnow().isoformat()),
        )


def get_user_predictions(user_id: int, disease: str | None = None) -> list[dict]:
    with _conn() as conn:
        if disease:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? AND disease = ? ORDER BY created_at",
                (user_id, disease),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at", (user_id,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_latest_prediction_per_disease(user_id: int) -> dict:
    rows = get_user_predictions(user_id)
    latest = {}
    for r in rows:  # rows ordered ascending by time, so last write wins = latest
        latest[r["disease"]] = {
            "probability": r["probability"], "risk_band": r["risk_band"],
            "features": json.loads(r["features_json"]),
        }
    return latest


def save_chat_message(user_id: int, role: str, content: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.utcnow().isoformat()),
        )


def get_chat_history(user_id: int, limit: int = 40) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


init_db()
