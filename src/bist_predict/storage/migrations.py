"""Schema versioning and migrations.

Migrations are applied in order. Each migration is a SQL string that transforms
the schema from version N to N+1. The current schema version is stored in the
schema_version table.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: dict[int, str] = {}


def get_current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version."""
    row = conn.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()
    return row[0] if row and row[0] else 0


def apply_pending_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the final schema version."""
    current = get_current_version(conn)
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            conn.executescript(MIGRATIONS[version])
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,),
            )
            conn.commit()
            current = version
    return current
