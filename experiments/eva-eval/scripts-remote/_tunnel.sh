#!/usr/bin/env bash
# Manage the morgen -> neo SSH tunnel that exposes neo's lmdeploy server
# (InternVL2-bf16) as morgen:18001. Eval clients keep talking to
# http://localhost:18001/v1 regardless of where the server actually runs.
#
# Usage:
#   ./_tunnel.sh up      # open if not already up
#   ./_tunnel.sh down    # close all matching tunnels
#   ./_tunnel.sh status  # print pid(s) or "no tunnel running"
#
# ServerAliveInterval keeps the tunnel alive across long eval runs (hours).
set -e
source "$(dirname "$0")/_env.sh"

TUNNEL_TAG="ssh.*-L 18001:127.0.0.1:18001.*${NEO_USER}@${NEO_HOST}"

case "${1:-status}" in
    up)
        if pgrep -af "$TUNNEL_TAG" >/dev/null; then
            echo "tunnel already up: $(pgrep -af "$TUNNEL_TAG")"
            exit 0
        fi
        ssh -fN \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=10 \
            -o ExitOnForwardFailure=yes \
            -L 18001:127.0.0.1:18001 \
            "${NEO_USER}@${NEO_HOST}"
        sleep 1
        if pgrep -af "$TUNNEL_TAG" >/dev/null; then
            echo "tunnel up: $(pgrep -af "$TUNNEL_TAG")"
        else
            echo "ERROR: tunnel failed to start"
            exit 1
        fi
        ;;
    down)
        if pgrep -af "$TUNNEL_TAG" >/dev/null; then
            pkill -f "$TUNNEL_TAG"
            echo "tunnel closed"
        else
            echo "no tunnel was running"
        fi
        ;;
    status)
        if pgrep -af "$TUNNEL_TAG" >/dev/null; then
            pgrep -af "$TUNNEL_TAG"
        else
            echo "no tunnel running"
        fi
        ;;
    *)
        echo "usage: $0 {up|down|status}"
        exit 1
        ;;
esac
