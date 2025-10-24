-- SQLite schema
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS register (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    confirm_password TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    height REAL,
    weight REAL
);

CREATE TABLE IF NOT EXISTS curls (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    count INTEGER DEFAULT 0,
    timing_minutes INTEGER DEFAULT 0,
    session_day DATE,
    start_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_ts DATETIME,
    FOREIGN KEY (user_id) REFERENCES register(user_id)
);

CREATE TABLE IF NOT EXISTS squats (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    count INTEGER DEFAULT 0,
    timing_minutes INTEGER DEFAULT 0,
    session_day DATE,
    start_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_ts DATETIME,
    FOREIGN KEY (user_id) REFERENCES register(user_id)
);

CREATE TABLE IF NOT EXISTS wallpushups (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    count INTEGER DEFAULT 0,
    timing_minutes INTEGER DEFAULT 0,
    session_day DATE,
    start_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_ts DATETIME,
    FOREIGN KEY (user_id) REFERENCES register(user_id)
);

CREATE TABLE IF NOT EXISTS highsteps (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    count INTEGER DEFAULT 0,
    timing_minutes INTEGER DEFAULT 0,
    session_day DATE,
    start_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_ts DATETIME,
    FOREIGN KEY (user_id) REFERENCES register(user_id)
);

CREATE TABLE IF NOT EXISTS crunches (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    count INTEGER DEFAULT 0,
    timing_minutes INTEGER DEFAULT 0,
    session_day DATE,
    start_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_ts DATETIME,
    FOREIGN KEY (user_id) REFERENCES register(user_id)
);

