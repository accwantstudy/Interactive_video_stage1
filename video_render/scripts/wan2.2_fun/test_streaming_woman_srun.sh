#!/usr/bin/env bash
# Same streaming test and Slurm resources, with the woman reference image.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/test_streaming_srun.sh" \
    --ref-image asset/synthetic_woman.png \
    --save-path samples/wan-videos-fun-control-a14b-streaming-woman \
    "$@"
