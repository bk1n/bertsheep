#!/usr/bin/env bash
set -euo pipefail

source .env

SRC=$(wslpath -u "$DATA_PATH")

rsync -ah --info=progress2 --stats "$SRC/" ./data/
echo "Data fetched from $DATA_PATH on $(date)"