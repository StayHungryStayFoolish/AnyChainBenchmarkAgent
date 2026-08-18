"""Content-sensitive repository identity observed by the running process."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def repository_revision(repo_root: str | Path) -> dict[str, str]:
    """Return commit plus tracked/untracked worktree content identity."""

    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
    diff = _git(root, "diff", "--binary", "HEAD", "--")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    digest = hashlib.sha256()
    digest.update(diff)
    for raw_name in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(raw_name)
        path = root / raw_name.decode("utf-8", errors="surrogateescape")
        if path.is_file():
            digest.update(path.read_bytes())
    return {"commit": commit, "worktree_hash": digest.hexdigest()}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=root, stderr=subprocess.DEVNULL)
