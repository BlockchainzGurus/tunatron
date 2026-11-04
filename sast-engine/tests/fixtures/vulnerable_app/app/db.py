import sqlite3

def fetch_orders(uid):
    cursor = sqlite3.connect("db.sqlite").cursor()
    cursor.execute(f"SELECT * FROM orders WHERE user_id = {uid}")
    return cursor.fetchall()

def fetch_user(uid):
    cursor = sqlite3.connect("db.sqlite").cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (uid,))
    return cursor.fetchone()
