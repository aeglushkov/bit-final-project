#!/usr/bin/env bash
# Stop the SI split stack. Identical wire-down logic as the bf16 stack
# (pkill patterns for vllm + lmdeploy are stack-agnostic, same SSH tunnel),
# so we just delegate.
exec "$(dirname "$0")/04_stop_servers_bf16.sh" "$@"
