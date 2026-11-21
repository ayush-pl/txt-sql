#!/usr/bin/env bash
# Run training with output appended to a log (for launching from a terminal tab).
# Usage (from Windows): wsl -d Ubuntu --cd <repo> -- bash scripts/train_logged.sh <config.yaml> <logfile>
cd "$(dirname "$0")/.."
echo "=== START $(date) ===" >> "$2"
bash scripts/train.sh --config "$1" >> "$2" 2>&1
echo "=== EXIT $? $(date) ===" >> "$2"
