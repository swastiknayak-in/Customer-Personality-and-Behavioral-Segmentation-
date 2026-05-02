import sqlite3, hashlib, os

DB = "shopindia.db"

def _h(p): return hashlib.sha256(p.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY, email TEXT UNIQUE,
        password TEXT, role TEXT, name TEXT)""")
    conn.commit()
    for email, pwd, role, name in [
        ("customer@demo.com", "customer123", "customer", "Priya Sharma"),
        ("manager@demo.com",  "manager123",  "manager",  "Rajesh Kumar"),
    ]:
        c.execute("INSERT OR IGNORE INTO users(email,password,role,name) VALUES(?,?,?,?)",
                  (email, _h(pwd), role, name))
    conn.commit(); conn.close()

def verify(email, pwd):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT role,name FROM users WHERE email=? AND password=?",
              (email.strip().lower(), _h(pwd)))
    row = c.fetchone(); conn.close()
    return row  # (role, name) or None
