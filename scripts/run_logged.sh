#!/usr/bin/env bash
# Run any repo script with output to a log and an exit marker (for launching from a terminal tab).
# Usage: wsl -d Ubuntu --cd <repo> -- bash scripts/run_logged.sh <logfile> <script.sh> [args...]
cd "$(dirname "$0")/.."
LOG=$1; shift
echo "=== START $(date) ===" >> "$LOG"
bash "$@" >> "$LOG" 2>&1
echo "=== EXIT $? $(date) ===" >> "$LOG"
