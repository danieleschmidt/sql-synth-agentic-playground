"""SQLite database management for the playground."""
import sqlite3
import os
from pathlib import Path

DEFAULT_DB = os.environ.get("SQLSYNTH_DB", ":memory:")


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a SQLite connection."""
    path = db_path or DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def execute_sql(sql: str, db_path: str = None) -> dict:
    """Execute a SQL statement and return results."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql)
        conn.commit()
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return {
            "success": True,
            "rows": [dict(r) for r in rows],
            "columns": columns,
            "rowcount": cursor.rowcount,
        }
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": str(e),
            "rows": [],
            "columns": [],
        }
    finally:
        conn.close()


def create_demo_db(db_path: str = ":memory:") -> sqlite3.Connection:
    """Create a demo database with sample tables."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employee (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary REAL,
            age INTEGER,
            country TEXT DEFAULT 'USA'
        );
        
        CREATE TABLE IF NOT EXISTS product (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL,
            category TEXT,
            stock INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS customer (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            country TEXT,
            age INTEGER
        );
        
        INSERT OR IGNORE INTO employee VALUES
            (1, 'Alice', 'Engineering', 95000, 30, 'USA'),
            (2, 'Bob', 'Marketing', 72000, 28, 'USA'),
            (3, 'Carol', 'Engineering', 105000, 35, 'Canada'),
            (4, 'Dave', 'Sales', 68000, 25, 'UK');
        
        INSERT OR IGNORE INTO product VALUES
            (1, 'Widget A', 29.99, 'Electronics', 100),
            (2, 'Widget B', 49.99, 'Electronics', 50),
            (3, 'Gadget X', 99.99, 'Electronics', 25),
            (4, 'Book', 14.99, 'Books', 200);
        
        INSERT OR IGNORE INTO customer VALUES
            (1, 'John', 'john@ex.com', 'USA', 32),
            (2, 'Jane', 'jane@ex.com', 'UK', 28),
            (3, 'Bob', 'bob@ex.com', 'USA', 45);
    """)
    conn.commit()
    return conn
