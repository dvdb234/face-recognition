#!/usr/bin/env bash
# ============================================================
#  One-command launcher (Raspberry Pi / Linux / macOS).
#  Run with:   ./run.sh
#  Add flags after it, e.g.:   ./run.sh --mjpg --sensitivity max
#  First time only, make it executable:   chmod +x run.sh
# ============================================================

# Move to this script's own folder, whatever the current directory is.
cd "$(dirname "$0")" || exit 1

if [ ! -f ".venv/bin/activate" ]; then
  echo
  echo "  Virtual environment not found in this folder."
  echo "  Set it up first - see the README 'Raspberry Pi' section."
  echo
  exit 1
fi

source .venv/bin/activate
python assistant.py "$@"
