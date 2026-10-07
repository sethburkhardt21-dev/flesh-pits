#!/bin/sh
# H9 — fetch chroma at the pinned SHA (donor_matrix D6).
# Usage: ./fetch.sh [dest-dir]
set -e
SHA="f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f"
DEST="${1:-./chroma-d6}"
if [ -e "$DEST" ]; then
  echo "destination exists: $DEST (refusing to overwrite)" >&2
  exit 1
fi
git clone https://github.com/chroma-core/chroma "$DEST"
cd "$DEST"
git checkout --detach "$SHA"
git rev-parse HEAD
echo "chroma pinned at $SHA"
