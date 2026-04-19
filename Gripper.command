#!/bin/bash
# Double-click this file in Finder to launch the gripper GUI.
# One-time prerequisites (run in Terminal):
#   brew install python-tk@3.14
set -e
cd "$(dirname "$0")"

if [ -d .git ]; then
    echo "Checking for updates..."
    git fetch --quiet origin 2>/dev/null || echo "  (offline, skipping)"
    git pull --ff-only --quiet 2>/dev/null || echo "  (no fast-forward update available)"
fi

PY=/opt/homebrew/bin/python3
[ -x "$PY" ] || PY=python3

if [ ! -d .venv ]; then
    echo "Creating virtualenv..."
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt

if ! python -c "import tkinter" >/dev/null 2>&1; then
    cat <<'EOF'

Tkinter isn't available for this Python. Install it once with:
    brew install python-tk@3.14
then rm -rf .venv and double-click this file again.

EOF
    exit 1
fi

exec python gripper.py gui
