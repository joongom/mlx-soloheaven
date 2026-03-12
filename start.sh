#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
mlx-soloheaven --gpu-keepalive "$@"
