from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parents[1]
DB_FILE = BASE / "db" / "produce.db"
SCHEMA = BASE / "db" / "schema.sql"

def init_schema():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_FILE)) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    print(f"Initialized DB at: {DB_FILE}")

if __name__ == "__main__":
    init_schema()
