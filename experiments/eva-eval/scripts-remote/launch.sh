#!/usr/bin/env bash
# Wrap a script so it survives SSH drops:
#   - nohup ignores SIGHUP
#   - setsid puts the process in its own session (no controlling TTY)
#   - </dev/null detaches stdin
#   - log goes to logs/<step>.log
# Usage: ./scripts-remote/launch.sh <step_name>  [args...]
#   where <step_name> matches a script in experiments/eva-eval/scripts-remote/<step>.sh

set -e

step=${1:?usage: launch.sh <step_name> [args...]}
shift || true

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
script="$ROOT/experiments/eva-eval/scripts-remote/${step}.sh"
log="$ROOT/logs/${step}.log"

[ -x "$script" ] || { echo "not found or not executable: $script"; exit 1; }
mkdir -p "$ROOT/logs"

cd "$ROOT"
nohup setsid bash "$script" "$@" > "$log" 2>&1 < /dev/null &
pid=$!
echo "$step pid=$pid"
echo "  tail -f $log"
echo "  pgrep -af $step    # check still running"
echo "  pkill -SIGINT -f $step    # gentle stop"
