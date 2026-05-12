#!/usr/bin/env bash
# umi gripper studio — double-click launcher for macOS.
# Creates/activates a venv, pulls latest, installs deps, launches app.
set -e
cd "$(dirname "$0")"

# Pull latest from main
git fetch origin main --quiet 2>/dev/null && git merge --ff-only origin/main --quiet 2>/dev/null || true

# Create venv on first run
if [ ! -d .venv ]; then
    echo "Setting up environment (first run)…"
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q --disable-pip-version-check
exec python3 studio.py
