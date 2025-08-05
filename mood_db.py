import sqlite3
from datetime import datetime

def create_table():
    conn = sqlite3.connect("mood_tracker.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    mood TEXT,
                    confidence REAL,
                    date TEXT
                )''')
    conn.commit()
    conn.close()

def insert_entry(text, mood, confidence):
    conn = sqlite3.connect("mood_tracker.db")
    c = conn.cursor()
    c.execute("INSERT INTO entries (text, mood, confidence, date) VALUES (?, ?, ?, ?)", 
              (text, mood, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_all_entries():
    conn = sqlite3.connect("mood_tracker.db")
    c = conn.cursor()
    c.execute("SELECT * FROM entries ORDER BY date DESC")
    rows = c.fetchall()
    conn.close()
    return rows
