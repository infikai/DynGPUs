#!/bin/bash
# discover_hosts.sh

HOSTFILE="hostfile.txt"

if [ -f "$HOSTFILE" ]; then
  cat "$HOSTFILE"
fi
