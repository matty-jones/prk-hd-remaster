"""SQLite registry for catalogue assets, runs, and per-stage jobs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"packed", "rejected", "blocked", "failed", "qa_passed"}
STATUSES = {
    "pending", "extracted", "generated", "pbr", "qa_passed", "packed",
    "retry", "rejected", "blocked", "failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.events_path: Path | None = None

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                key TEXT PRIMARY KEY,
                rdb_type INTEGER NOT NULL,
                rdb_id INTEGER NOT NULL,
                source_sha256 TEXT,
                pixel_sha256 TEXT,
                source_width INTEGER,
                source_height INTEGER,
                source_format TEXT,
                name TEXT,
                kind TEXT,
                family TEXT,
                material_class TEXT,
                tags TEXT,
                disposition TEXT NOT NULL,
                family_prompt_version TEXT,
                source_metadata TEXT,
                risk TEXT,
                seam REAL,
                alpha REAL,
                duplicate_of TEXT,
                canonical_key TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                cohort TEXT,
                preset TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                config_json TEXT,
                detail TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                seed INTEGER,
                prompt TEXT,
                cache_key TEXT,
                target_w INTEGER,
                target_h INTEGER,
                attempt INTEGER DEFAULT 0,
                rejection_reason TEXT,
                canonical_key TEXT,
                detail TEXT,
                heartbeat TEXT,
                updated_at TEXT,
                PRIMARY KEY (run_id, asset_key)
            );
            CREATE TABLE IF NOT EXISTS outputs (
                run_id TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                role TEXT NOT NULL,
                path TEXT,
                sha256 TEXT,
                bytes INTEGER,
                PRIMARY KEY (run_id, asset_key, role)
            );
            """
        )
        self.conn.commit()

    def upsert_asset(self, asset: dict, *, commit: bool = True) -> None:
        existing = self.asset(asset["key"])
        source_sha = asset.get("source_sha256") or (existing or {}).get("source_sha256")
        pixel_sha = asset.get("pixel_sha256") or (existing or {}).get("pixel_sha256")
        metadata = asset.get("source_metadata") or {}
        if not isinstance(metadata, str):
            metadata = json.dumps(metadata, sort_keys=True)
        tags = asset.get("tags") or []
        if not isinstance(tags, str):
            tags = json.dumps(tags)
        self.conn.execute(
            """
            INSERT INTO assets (
                key, rdb_type, rdb_id, source_sha256, pixel_sha256, source_width,
                source_height, source_format, name, kind, family, material_class,
                tags, disposition, family_prompt_version, source_metadata, risk,
                seam, alpha, duplicate_of, canonical_key, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                source_sha256=excluded.source_sha256,
                pixel_sha256=excluded.pixel_sha256,
                source_width=excluded.source_width,
                source_height=excluded.source_height,
                source_format=excluded.source_format,
                name=excluded.name,
                kind=excluded.kind,
                family=excluded.family,
                material_class=excluded.material_class,
                tags=excluded.tags,
                disposition=excluded.disposition,
                family_prompt_version=excluded.family_prompt_version,
                source_metadata=excluded.source_metadata,
                risk=excluded.risk,
                seam=excluded.seam,
                alpha=excluded.alpha,
                duplicate_of=excluded.duplicate_of,
                canonical_key=excluded.canonical_key,
                updated_at=excluded.updated_at
            """,
            (
                asset["key"], int(asset["rdb_type"]), int(asset["rdb_id"]),
                source_sha, pixel_sha, asset.get("source_width"), asset.get("source_height"),
                asset.get("source_format"), asset.get("name"), asset.get("kind"),
                asset.get("family"), asset.get("material_class"), tags,
                asset["disposition"], asset.get("family_prompt_version"), metadata,
                asset.get("risk"), asset.get("seam"), asset.get("alpha"),
                asset.get("duplicate_of"), asset.get("canonical_key"), utc_now(),
            ),
        )
        if commit:
            self.conn.commit()

    def _asset_from_row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        asset = dict(row)
        asset["tags"] = json.loads(asset["tags"] or "[]")
        asset["source_metadata"] = json.loads(asset["source_metadata"] or "{}")
        return asset

    def asset(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM assets WHERE key = ?", (key,)).fetchone()
        return self._asset_from_row(row)

    def all_assets(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM assets ORDER BY rdb_type, rdb_id").fetchall()
        return [self._asset_from_row(row) for row in rows]

    def update_asset_hashes(self, key: str, source_sha256: str, pixel_sha256: str, width: int, height: int, source_format: str) -> None:
        self.conn.execute(
            """
            UPDATE assets
            SET source_sha256=?, pixel_sha256=?, source_width=?, source_height=?,
                source_format=?, updated_at=?
            WHERE key=?
            """,
            (source_sha256, pixel_sha256, width, height, source_format, utc_now(), key),
        )
        self.conn.commit()

    def cohort_assets(self, *, risk: str = "auto", family: str | None = None, types: list[int] | None = None, ids: set[str] | None = None) -> list[dict]:
        query = "SELECT * FROM assets WHERE risk = ?"
        params: list = [risk]
        if family:
            query += " AND family = ?"
            params.append(family)
        if types:
            marks = ",".join("?" for _ in types)
            query += f" AND rdb_type IN ({marks})"
            params.extend(int(item) for item in types)
        query += " ORDER BY rdb_type, rdb_id"
        rows = self.conn.execute(query, params).fetchall()
        assets = [self._asset_from_row(row) for row in rows]
        if ids is not None:
            assets = [asset for asset in assets if asset["key"] in ids]
        return assets

    def aliases_of(self, key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM assets WHERE duplicate_of = ? ORDER BY rdb_type, rdb_id",
            (key,),
        ).fetchall()
        return [self._asset_from_row(row) for row in rows]

    def create_run(self, run_id: str, cohort: str, preset: str, config: dict) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO runs (run_id, cohort, preset, status, created_at, updated_at, config_json, detail)
            VALUES (?, ?, ?, 'running', ?, ?, ?, '{}')
            """,
            (run_id, cohort, preset, now, now, json.dumps({"config_revision": config.get("config_revision")})),
        )
        self.conn.commit()

    def latest_run_id(self, cohort: str | None = None) -> str | None:
        if cohort:
            row = self.conn.execute(
                "SELECT run_id FROM runs WHERE cohort = ? ORDER BY created_at DESC LIMIT 1",
                (cohort,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["run_id"] if row else None

    def run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["detail"] = json.loads(data["detail"] or "{}")
        return data

    def update_run(self, run_id: str, **fields) -> None:
        if "detail" in fields and not isinstance(fields["detail"], str):
            fields["detail"] = json.dumps(fields["detail"])
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        self.conn.execute(
            f"UPDATE runs SET {assignments} WHERE run_id=?",
            (*fields.values(), run_id),
        )
        self.conn.commit()

    def upsert_job(self, job: dict) -> None:
        detail = job.get("detail") or {}
        if not isinstance(detail, str):
            detail = json.dumps(detail)
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO jobs (
                run_id, asset_key, status, stage, seed, prompt, cache_key, target_w,
                target_h, attempt, rejection_reason, canonical_key, detail, heartbeat, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, asset_key) DO UPDATE SET
                status=excluded.status,
                stage=excluded.stage,
                seed=excluded.seed,
                prompt=excluded.prompt,
                cache_key=excluded.cache_key,
                target_w=excluded.target_w,
                target_h=excluded.target_h,
                attempt=excluded.attempt,
                rejection_reason=excluded.rejection_reason,
                canonical_key=excluded.canonical_key,
                detail=excluded.detail,
                heartbeat=excluded.heartbeat,
                updated_at=excluded.updated_at
            """,
            (
                job["run_id"], job["asset_key"], job["status"], job.get("stage"),
                job.get("seed"), job.get("prompt"), job.get("cache_key"),
                job.get("target_w"), job.get("target_h"), int(job.get("attempt") or 0),
                job.get("rejection_reason"), job.get("canonical_key"), detail,
                job.get("heartbeat") or now, now,
            ),
        )
        self.conn.commit()

    def _job_from_row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["detail"] = json.loads(job["detail"] or "{}")
        return job

    def job(self, run_id: str, asset_key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE run_id=? AND asset_key=?",
            (run_id, asset_key),
        ).fetchone()
        return self._job_from_row(row)

    def jobs(self, run_id: str, status: str | None = None) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE run_id=? AND status=? ORDER BY asset_key",
                (run_id, status),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE run_id=? ORDER BY asset_key",
                (run_id,),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def update_job(self, run_id: str, asset_key: str, **fields) -> None:
        if "detail" in fields and not isinstance(fields["detail"], str):
            fields["detail"] = json.dumps(fields["detail"])
        fields["updated_at"] = utc_now()
        if "heartbeat" not in fields:
            fields["heartbeat"] = fields["updated_at"]
        assignments = ", ".join(f"{key}=?" for key in fields)
        self.conn.execute(
            f"UPDATE jobs SET {assignments} WHERE run_id=? AND asset_key=?",
            (*fields.values(), run_id, asset_key),
        )
        self.conn.commit()

    def add_output(self, run_id: str, asset_key: str, role: str, path: Path, sha256: str, size: int) -> None:
        self.conn.execute(
            """
            INSERT INTO outputs (run_id, asset_key, role, path, sha256, bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, asset_key, role) DO UPDATE SET
                path=excluded.path, sha256=excluded.sha256, bytes=excluded.bytes
            """,
            (run_id, asset_key, role, str(path), sha256, int(size)),
        )
        self.conn.commit()

    def outputs(self, run_id: str, asset_key: str | None = None) -> list[dict]:
        if asset_key:
            rows = self.conn.execute(
                "SELECT * FROM outputs WHERE run_id=? AND asset_key=? ORDER BY role",
                (run_id, asset_key),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM outputs WHERE run_id=? ORDER BY asset_key, role",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self, run_id: str) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=? GROUP BY status",
            (run_id,),
        ).fetchall()
        counts = {row["status"]: row["n"] for row in rows}
        total = sum(counts.values())
        return {"run_id": run_id, "total": total, "counts": counts}

    def log_event(self, run_id: str, asset_key: str, event: str, **payload) -> None:
        if self.events_path is None:
            return
        record = {
            "ts": utc_now(),
            "run_id": run_id,
            "asset_key": asset_key,
            "event": event,
            **payload,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
