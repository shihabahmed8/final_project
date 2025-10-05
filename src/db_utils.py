import sqlite3, pathlib, hashlib, datetime
from typing import Optional, Dict, Any, Tuple

BASE = pathlib.Path(__file__).resolve().parents[1]
DB_FILE = BASE / "db" / "produce.db"

def connect():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _to_storable_path(image_path: str) -> str:
    p = pathlib.Path(image_path).resolve()
    try:
        return str(p.relative_to(BASE))
    except Exception:
        return str(p)

def insert_sample(image_path: str, item_name: str) -> tuple[int, bool]:
    """
    Insert a sample if it's not already in the DB based on file SHA1.
    Returns (sample_id, created).
      - created = True  => new row inserted
      - created = False => row already existed (same file_hash)
    """
    file_hash = sha1_file(image_path)  # hash on actual file
    scan_date = datetime.datetime.utcnow().isoformat(timespec="seconds")
    stored_path = _to_storable_path(image_path)

    with connect() as conn:
        cur = conn.cursor()

        # 1) Check if this hash already exists
        cur.execute("SELECT sample_id FROM Produce_Samples WHERE file_hash = ?", (file_hash,))
        row = cur.fetchone()
        if row:
            # already present
            return int(row[0]), False

        # 2) Not found => insert new
        cur.execute(
            """
            INSERT INTO Produce_Samples (file_hash, item_name, scan_date, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (file_hash, item_name, scan_date, stored_path),
        )
        conn.commit()
        return cur.lastrowid, True


def insert_quality(sample_id: int, quality: str, confidence: float, freshness: float) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO Quality_Results (sample_id, quality_class, confidence, freshness_index)
            VALUES (?, ?, ?, ?)
            """,
            (int(sample_id), str(quality), float(confidence), float(freshness)),
        )
        conn.commit()
        return cur.lastrowid

def insert_shelf(sample_id: int, predicted_days: int, optimal_temp: float, decay_rate: float) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO Shelf_Life_Metrics (
                sample_id,
                predicted_storage_days,
                optimal_temp_c,
                mock_decay_rate
            )
            VALUES (?, ?, ?, ?)
            """,
            (int(sample_id), int(predicted_days), float(optimal_temp), float(decay_rate)),
        )
        conn.commit()
        return cur.lastrowid

def delete_sample(sample_id: int) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM Produce_Samples WHERE sample_id=?", (sample_id,))
        conn.commit()
        return cur.rowcount

def delete_item(item_name: str) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM Produce_Samples WHERE item_name=?", (item_name,))
        conn.commit()
        return cur.rowcount
