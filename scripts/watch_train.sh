#!/usr/bin/env bash
# Training watchdog: prints one line per problem (and per eval result), silent otherwise.
#   SLOW  sec/step over a 10-min window above the limit (windows containing an eval are skipped)
#   MEM   GPU memory above the limit (spill into system RAM is what slowed run 2 and crashed evals)
#   DEAD  out-of-memory, traceback or process exit
# Usage: bash scripts/watch_train.sh <logfile> [max_sec_per_step=45] [max_mem_mib=7500]
LOG=$1; MAX_S=${2:-45}; MAX_MEM=${3:-7500}
since() { tr '\r' '\n' < "$LOG" | awk '/=== START/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}'; }
step() { since | grep -oE '\| *[0-9]+/[0-9]+ \[' | tail -1 | grep -oE '[0-9]+' | head -1; }
t0=$(date +%s); s0=$(step); s0=${s0:-0}; mem_alerted=0
seen_ex=$(since | grep -c "eval_ex =")  # only report evals that happen after (re)arming
while true; do
  sleep 60
  out=$(since)
  if echo "$out" | grep -qiE "out of memory|OutOfMemoryError|Traceback|=== EXIT"; then
    echo "DEAD: $(echo "$out" | grep -iE "out of memory|OutOfMemoryError|Error|=== EXIT" | tail -2 | tr '\n' ' ' | cut -c1-300)"
    exit 0
  fi
  n_ex=$(echo "$out" | grep -c "eval_ex =")
  if [ "$n_ex" -gt "$seen_ex" ]; then echo "EVAL: $(echo "$out" | grep "eval_ex =" | tail -1)"; seen_ex=$n_ex; fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$mem" ] && [ "$mem" -gt "$MAX_MEM" ] && [ $mem_alerted -eq 0 ]; then
    echo "MEM: GPU at ${mem} MiB (limit ${MAX_MEM}) at step $(step)"; mem_alerted=1
  elif [ -n "$mem" ] && [ "$mem" -le "$MAX_MEM" ]; then mem_alerted=0; fi
  now=$(date +%s); s=$(step); s=${s:-0}
  if [ $((now - t0)) -ge 600 ]; then
    if [ "$s" -gt "$s0" ] && [ $((s / 100)) -eq $((s0 / 100)) ]; then
      sps=$(( (now - t0) / (s - s0) ))
      [ "$sps" -gt "$MAX_S" ] && echo "SLOW: ${sps} s/step over steps ${s0}-${s} (limit ${MAX_S}), GPU ${mem} MiB"
    elif [ "$s" -eq "$s0" ] && [ $(( (s + 1) % 100 )) -ne 0 ] && [ $((s % 100)) -ne 0 ]; then
      echo "SLOW: no progress for 10 min at step ${s}, GPU ${mem} MiB"
    fi
    t0=$now; s0=$s
  fi
done
