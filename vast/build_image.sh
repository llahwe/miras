#!/usr/bin/env bash
set -euo pipefail

# Build the Vast docker image locally.
# Usage:
#   ./vast/build_image.sh miras-vast:latest
#
# If you want to use this on Vast, push to a registry you control and set the Vast template image to it.

TAG="${1:-miras-vast:latest}"

docker build -f vast/Dockerfile -t "${TAG}" .
echo "Built ${TAG}"

