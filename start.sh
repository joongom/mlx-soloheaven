#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven --memory-budget-gb 50 --gpu-keepalive "$@"
