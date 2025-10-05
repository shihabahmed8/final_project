PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Produce_Samples (
  sample_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  file_hash   TEXT UNIQUE,
  item_name   TEXT,
  scan_date   TEXT,
  image_path  TEXT
);

CREATE TABLE IF NOT EXISTS Quality_Results (
  result_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id        INTEGER NOT NULL,
  quality_class    TEXT,
  confidence       REAL,
  freshness_index  REAL,
  FOREIGN KEY(sample_id)
    REFERENCES Produce_Samples(sample_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Shelf_Life_Metrics (
  shelf_id             INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id            INTEGER NOT NULL,
  predicted_storage_days INTEGER,
  optimal_temp_c         REAL,
  mock_decay_rate        REAL,
  FOREIGN KEY(sample_id)
    REFERENCES Produce_Samples(sample_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_sample ON Quality_Results(sample_id);
CREATE INDEX IF NOT EXISTS idx_shelf_sample   ON Shelf_Life_Metrics(sample_id);
