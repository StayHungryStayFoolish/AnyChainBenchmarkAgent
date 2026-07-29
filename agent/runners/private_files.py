"""Private filesystem boundary for job-local execution material."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_directory(path: str | Path) -> Path:
    """Create or tighten one job-owned directory."""

    directory = Path(path)
    directory.mkdir(
        mode=PRIVATE_DIRECTORY_MODE,
        parents=True,
        exist_ok=True,
    )
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"private job path is not a directory: {directory}")
    os.chmod(directory, PRIVATE_DIRECTORY_MODE)
    return directory


def atomic_write_private_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write a private file with a fixed owner-only mode."""

    target = Path(path)
    ensure_private_directory(target.parent)
    if target.is_symlink():
        raise ValueError(f"private job file may not be a symlink: {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        PRIVATE_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, PRIVATE_FILE_MODE)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_private_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically write private binary evidence."""

    target = Path(path)
    ensure_private_directory(target.parent)
    if target.is_symlink():
        raise ValueError(f"private job file may not be a symlink: {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        PRIVATE_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, PRIVATE_FILE_MODE)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
