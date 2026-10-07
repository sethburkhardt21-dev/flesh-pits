#!/bin/sh
# H11 — fetch MAPIE at the pinned SHA (donor_matrix D17).
# Usage: ./fetch.sh [dest-dir]
set -e
SHA="3b84b8212db2bba452ef5a09ae06a0dd545869ae"
DEST="${1:-./mapie-d17}"
if [ -e "$DEST" ]; then
  echo "destination exists: $DEST (refusing to overwrite)" >&2
  exit 1
fi
git clone https://github.com/scikit-learn-contrib/MAPIE "$DEST"
cd "$DEST"
git checkout --detach "$SHA"
git rev-parse HEAD
echo "MAPIE pinned at $SHA"
