#!/bin/bash
# Build umi.app — a self-contained macOS application bundle.
#
# Run on macOS with python3 + python-tk available:
#     brew install python-tk@3.14
#     ./build_app.sh
#
# Output: dist/umi.app  — drag it to /Applications.
set -euo pipefail
cd "$(dirname "$0")"

PY=/opt/homebrew/bin/python3
[ -x "$PY" ] || PY=python3

if [ ! -d .venv-build ]; then
    echo "==> creating build venv"
    "$PY" -m venv .venv-build
fi
# shellcheck disable=SC1091
source .venv-build/bin/activate

echo "==> installing runtime + build deps"
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -r requirements-build.txt
# Optional extras — pulled in if available so the .app has them baked in.
pip install bleak telemetry-parser-py 2>/dev/null || true

if ! python -c "import tkinter" >/dev/null 2>&1; then
    cat <<'EOF'
Tk isn't available for the build interpreter. Install it once with:
    brew install python-tk@3.14
then ``rm -rf .venv-build`` and rerun this script.
EOF
    exit 1
fi

echo "==> cleaning previous build"
rm -rf build dist

echo "==> running py2app"
python setup_app.py py2app

cat <<EOF

Done.  dist/umi.app is the standalone bundle.

  open dist/umi.app          # try it
  cp -R dist/umi.app /Applications/   # install it

First launch on macOS: right-click the .app -> Open, then confirm — this
bypasses Gatekeeper for the unsigned bundle. For a fully signed +
notarised distributable, see the comments at the top of setup_app.py.
EOF
