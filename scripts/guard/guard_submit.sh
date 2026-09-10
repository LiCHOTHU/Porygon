#!/bin/bash
# guard_submit.sh <name> -- <full sbatch command with __EX__ placeholder for exclude list>
# Submits, records jobid + command in the registry so the watchdog can resubmit
# if the job dies on a broken GPU node.
set -u
NAME="$1"; shift; [ "$1" = "--" ] && shift
REG=~/workspace/imitation/scripts/guard/registry.tsv
EX=$(cat ~/workspace/imitation/scripts/flaky_nodes_exclude.txt)
CMD="$*"
JID=$(eval "${CMD//__EX__/$EX}")
echo -e "${JID}\t${NAME}\t${CMD}" >> $REG
echo "$JID"
