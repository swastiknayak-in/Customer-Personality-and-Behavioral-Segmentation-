import sqlite3
import hashlib

DB_PATH = "retail_users.db"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """Create users table and seed demo accounts."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            segment  TEXT NOT NULL DEFAULT 'Regular'
        )
    """)
    conn.commit()

    demo_users = [
        ("admin@demo.com",  "admin123",  "Top"),
        ("user@demo.com",   "user123",   "Regular"),
        ("guest@demo.com",  "guest123",  "Budget"),
    ]
    for email, pwd, seg in demo_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (email, password, segment) VALUES (?, ?, ?)",
            (email, _hash(pwd), seg),
        )
    conn.commit()
    conn.close()


def verify_user(email: str, password: str):
    """Return segment string on success, None on failure."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        "SELECT segment FROM users WHERE email = ? AND password = ?",
        (email.strip().lower(), _hash(password)),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None
