"""Immutable experiment directories and reproducibility manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
import platform
import re
import subprocess


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def default_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("ews_%Y%m%d_%H%M%S")


def create_run_directory(runs_dir, run_id: str | None = None) -> Path:
    """Create a new output directory and refuse accidental overwrite."""
    runs_dir = Path(runs_dir).resolve()
    run_id = run_id or default_run_id()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    output_dir = runs_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    *,
    root,
    output_dir,
    config,
    data_files,
    code_files,
    status="running",
    extra=None,
):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()

    def file_records(paths):
        records = []
        for value in sorted({Path(path).resolve() for path in paths}):
            if not value.is_file():
                continue
            try:
                relative = value.relative_to(root).as_posix()
            except ValueError:
                relative = str(value)
            records.append(
                {
                    "path": relative,
                    "bytes": value.stat().st_size,
                    "sha256": sha256_file(value),
                }
            )
        return records

    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "yfinance"]
        ),
        "config": config,
        "data_files": file_records(data_files),
        "code_files": file_records(code_files),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(output_dir, manifest) -> Path:
    path = Path(output_dir) / "experiment_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
