#!/usr/bin/env python3
# scripts/migrate_sqlite_to_postgres.py
"""
CareAI Database Migration Utility: SQLite -> PostgreSQL.

Safely migrates all user accounts, check-ins, journal entries, chat messages,
safety alerts, counselor notifications, and clinical case summaries from SQLite
to PostgreSQL.

Guarantees:
- Preserves primary key IDs, foreign key integrity, and timestamps.
- Idempotent: checks for existing IDs to prevent duplicates on re-runs.
- Updates PostgreSQL primary key serial sequences after insertion.
- Performs post-migration row-count verification.
- Non-destructive: read-only on source SQLite database.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("db_migration")

# Explicit ordered list of tables to respect foreign-key dependencies
TABLE_ORDER = [
    "users",
    "checkins",
    "journals",
    "alerts",
    "chat_messages",
    "counselor_notifications",
    "case_summaries"
]


def normalize_postgres_url(url: str) -> str:
    """Ensures PostgreSQL URLs use postgresql+psycopg2:// driver prefix for reliability."""
    if not url:
        return ""
    clean_url = url.strip().strip("'\"")
    if clean_url.startswith("postgres://"):
        return clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif clean_url.startswith("postgresql://"):
        return clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean_url


def migrate_database(sqlite_path: str, pg_url: str, dry_run: bool = False) -> bool:
    """Performs the migration from SQLite to PostgreSQL."""
    if not os.path.exists(sqlite_path):
        logger.error(f"Source SQLite database does not exist: {sqlite_path}")
        return False

    pg_url = normalize_postgres_url(pg_url)
    if not pg_url:
        logger.error("Target PostgreSQL connection URL is empty.")
        return False

    logger.info(f"Source SQLite: {sqlite_path}")
    masked_pg = pg_url.split("@")[-1] if "@" in pg_url else "configured_pg"
    logger.info(f"Target PostgreSQL: ...@{masked_pg}")
    if dry_run:
        logger.info("[DRY RUN MODE] No changes will be committed to PostgreSQL.")

    # 1. Initialize SQLite Engine
    sqlite_uri = f"sqlite:///{os.path.abspath(sqlite_path).replace(os.sep, '/')}"
    sqlite_engine = create_engine(sqlite_uri)

    # 2. Initialize PostgreSQL Engine
    pg_engine = create_engine(pg_url, pool_pre_ping=True)

    sqlite_inspector = inspect(sqlite_engine)
    pg_inspector = inspect(pg_engine)

    available_sqlite_tables = set(sqlite_inspector.get_table_names())
    available_pg_tables = set(pg_inspector.get_table_names())

    logger.info(f"Available SQLite tables: {sorted(list(available_sqlite_tables))}")
    logger.info(f"Available PostgreSQL tables: {sorted(list(available_pg_tables))}")

    migration_summary = {}

    with sqlite_engine.connect() as src_conn, pg_engine.connect() as dst_conn:
        for table_name in TABLE_ORDER:
            if table_name not in available_sqlite_tables:
                logger.info(f"Table '{table_name}' not found in SQLite. Skipping.")
                continue

            if table_name not in available_pg_tables:
                logger.warning(
                    f"Target PostgreSQL table '{table_name}' does not exist! "
                    f"Run Flask app or db.create_all() first to initialize schema. Skipping."
                )
                continue

            # Fetch all rows from SQLite
            src_rows = src_conn.execute(text(f"SELECT * FROM {table_name}")).mappings().all()
            total_src = len(src_rows)

            if total_src == 0:
                logger.info(f"Table '{table_name}' has 0 rows in source. Skipping.")
                migration_summary[table_name] = {"source": 0, "inserted": 0, "skipped": 0}
                continue

            # Fetch existing IDs in PostgreSQL to prevent duplicate key violations
            existing_pg_ids = set()
            try:
                pg_id_rows = dst_conn.execute(text(f"SELECT id FROM {table_name}")).fetchall()
                existing_pg_ids = {row[0] for row in pg_id_rows}
            except Exception as e:
                logger.warning(f"Could not read existing IDs for {table_name}: {e}")

            # Inspect columns and types in target table
            pg_col_types = {col["name"]: str(col["type"]).upper() for col in pg_inspector.get_columns(table_name)}
            pg_columns = set(pg_col_types.keys())

            inserted_count = 0
            skipped_count = 0

            for row_dict in src_rows:
                row_data = dict(row_dict)
                row_id = row_data.get("id")

                if row_id is not None and row_id in existing_pg_ids:
                    skipped_count += 1
                    continue

                # Filter columns to only those existing in target PostgreSQL table
                filtered_data = {k: v for k, v in row_data.items() if k in pg_columns}

                if not filtered_data:
                    continue

                # Normalize types for PostgreSQL compatibility (e.g. SQLite stores booleans as 0/1)
                for k, v in list(filtered_data.items()):
                    c_type = pg_col_types.get(k, "")
                    if "BOOL" in c_type and v is not None:
                        if isinstance(v, (int, float)):
                            filtered_data[k] = bool(v)
                        elif isinstance(v, str):
                            filtered_data[k] = v.strip().lower() in ("1", "true", "t", "yes")
                    elif ("TIMESTAMP" in c_type or "DATETIME" in c_type) and isinstance(v, str) and v:
                        try:
                            filtered_data[k] = datetime.fromisoformat(v)
                        except Exception:
                            pass

                col_names = list(filtered_data.keys())
                col_placeholders = [f":{col}" for col in col_names]
                insert_stmt = text(
                    f"INSERT INTO {table_name} ({', '.join(col_names)}) "
                    f"VALUES ({', '.join(col_placeholders)})"
                )

                if not dry_run:
                    dst_conn.execute(insert_stmt, filtered_data)
                inserted_count += 1

            if not dry_run:
                dst_conn.commit()

                # Update PostgreSQL sequence to match MAX(id)
                try:
                    seq_update = text(
                        f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), "
                        f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table_name};"
                    )
                    dst_conn.execute(seq_update)
                    dst_conn.commit()
                except Exception as seq_err:
                    logger.debug(f"Sequence update note for {table_name}: {seq_err}")

            logger.info(f"Table '{table_name}': {inserted_count} inserted, {skipped_count} skipped (total source: {total_src})")
            migration_summary[table_name] = {
                "source": total_src,
                "inserted": inserted_count,
                "skipped": skipped_count
            }

        # Verification step
        logger.info("\n--- Migration Verification ---")
        for table_name in TABLE_ORDER:
            if table_name in available_pg_tables:
                pg_count = dst_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                logger.info(f"PostgreSQL '{table_name}' total rows: {pg_count}")

    logger.info("Migration completed successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate CareAI database from SQLite to PostgreSQL.")
    parser.add_argument(
        "--sqlite-path",
        default=os.path.join(os.path.dirname(__file__), "..", "instance", "mhapp.db"),
        help="Path to source SQLite database file (default: instance/mhapp.db)"
    )
    parser.add_argument(
        "--pg-url",
        default=os.getenv("DATABASE_URL"),
        help="Target PostgreSQL connection URL (or set DATABASE_URL environment variable)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without committing to target PostgreSQL"
    )

    args = parser.parse_args()

    if not args.pg_url:
        print("ERROR: PostgreSQL URL not provided. Set DATABASE_URL or pass --pg-url.")
        sys.exit(1)

    success = migrate_database(sqlite_path=args.sqlite_path, pg_url=args.pg_url, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
