"""Version detection + self-update for umi.

Two modes:

* Dev checkout — there's a ``.git/`` next to the source. ``current_commit``
  asks git; ``pull`` runs ``git pull --ff-only``.
* App bundle — no ``.git/``; ``build_app.sh`` bakes the head commit into
  a ``VERSION`` file inside the bundle's resources. ``pull`` is a no-op
  in this mode and the studio's update button opens the repo page so
  the user can pull + rebuild.

Either way ``latest_remote_commit`` asks the GitHub API for the current
HEAD of ``main`` so we can tell the user whether an update exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Optional

REPO_OWNER = "ibrue"
REPO_NAME = "gripper-test"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
API_LATEST_COMMIT = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/main"
)


def is_dev_checkout(repo_dir: str) -> bool:
    return os.path.isdir(os.path.join(repo_dir, ".git"))


def _version_file(repo_dir: str) -> str:
    return os.path.join(repo_dir, "VERSION")


def current_commit(repo_dir: str) -> Optional[str]:
    """Return the full commit SHA we're currently running, or None.

    Priority:
      1. ``git rev-parse HEAD`` if ``repo_dir`` is a dev checkout.
      2. A baked ``_baked_version.BAKED_SHA`` constant (written by
         build_app.sh before py2app runs — works regardless of where
         the bundling tool drops resource files).
      3. A ``VERSION`` text file in ``repo_dir`` (fallback).
    """
    if is_dev_checkout(repo_dir):
        try:
            out = subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=5,
            )
            return out.decode().strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            pass
    try:
        import _baked_version  # type: ignore
        sha = getattr(_baked_version, "BAKED_SHA", None)
        if isinstance(sha, str) and sha and sha != "unknown":
            return sha
    except ImportError:
        pass
    vf = _version_file(repo_dir)
    if os.path.exists(vf):
        try:
            with open(vf) as f:
                return f.read().strip() or None
        except OSError:
            pass
    return None


def short(sha: Optional[str]) -> str:
    return sha[:7] if sha else "?"


def latest_remote_commit(timeout: float = 5.0) -> Optional[str]:
    req = urllib.request.Request(
        API_LATEST_COMMIT,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "umi-studio",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None
    sha = data.get("sha")
    return sha if isinstance(sha, str) else None


def pull(repo_dir: str) -> tuple[bool, str]:
    """``git pull --ff-only`` in repo_dir. (False, msg) on failure."""
    if not is_dev_checkout(repo_dir):
        return False, "not a git checkout (this is an .app bundle — rebuild instead)"
    try:
        res = subprocess.run(
            ["git", "-C", repo_dir, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "git pull timed out"
    except FileNotFoundError:
        return False, "git isn't installed"
    if res.returncode != 0:
        return False, (res.stderr or res.stdout or "git failed").strip()
    out = (res.stdout or "").strip()
    if not out or "Already up to date" in out:
        return True, "already up to date"
    return True, out
