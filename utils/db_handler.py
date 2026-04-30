import sqlite3
import hashlib
import os

DB_PATH = "olist_users.db"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """Create the users table and seed demo accounts if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            segment  TEXT NOT NULL DEFAULT 'Medium'
        )
    """)
    conn.commit()

    # Seed demo users
    demo_users = [
        ("admin@test.com",   "admin123",  "High"),
        ("silver@test.com",  "silver123", "Medium"),
        ("bronze@test.com",  "bronze123", "Low"),
    ]
    for email, pwd, seg in demo_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (email, password, segment) VALUES (?, ?, ?)",
            (email, _hash(pwd), seg),
        )
    conn.commit()
    conn.close()


def verify_user(email: str, password: str):
    """Return the user's segment string on success, None on failure."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT segment FROM users WHERE email = ? AND password = ?",
        (email.strip().lower(), _hash(password)),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def register_user(email: str, password: str, segment: str = "Low") -> bool:
    """Register a new user. Returns True on success, False if email exists."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (email, password, segment) VALUES (?, ?, ?)",
            (email.strip().lower(), _hash(password), segment),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
