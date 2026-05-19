from __future__ import annotations

import re
from pathlib import Path

import duckdb


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def init_db(path: Path) -> None:
    """Create the database file and initialize all tables from schema.sql."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    con = duckdb.connect(str(path))
    try:
        for stmt in _split_statements(schema_sql):
            con.execute(stmt)
    finally:
        con.close()


def get_connection(path: Path) -> duckdb.DuckDBPyConnection:
    """Return a read-write connection to the runs database."""
    return duckdb.connect(str(path))
