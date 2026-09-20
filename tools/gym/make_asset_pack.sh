#!/bin/sh
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Assemble the asset pack the Python package downloads on first use.
#
# The pack is data/ from this checkout plus the subset of stk-assets listed in
# asset-manifest.txt, laid out as a source checkout and its sibling assets:
#
#     <pack>/stk/data/     <- the game data
#     <pack>/stk-assets/   <- the traced subset
#     <pack>/supertuxkart  <- the binary
#
# That nesting is not cosmetic. Run with the working directory at <pack>/stk,
# FileManager finds "./data/" and then resolves the assets as
# "./data/../../stk-assets", which lands on <pack>/stk-assets -- so the pack
# needs no environment variables at all, exactly as a checkout beside
# stk-assets needs none. Setting SUPERTUXKART_DATADIR=<pack>/stk works too and
# resolves to the same place.
#
# Every directory the game looks for is created even when the manifest puts no
# file in it: FileManager aborts with "Not all assets found" if one is missing,
# and an empty music/ is cheaper than the 68 MB of music a race never plays.
#
# Usage: make_asset_pack.sh <stk-assets-dir> <binary> <output-dir>

set -eu

if [ $# -ne 3 ]; then
    sed -n '6,20p' "$0" >&2
    exit 2
fi

ASSETS=$1
BINARY=$2
OUT=$3
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
MANIFEST=$HERE/asset-manifest.txt

[ -d "$ASSETS" ] || { echo "no such assets directory: $ASSETS" >&2; exit 1; }
[ -f "$BINARY" ] || { echo "no such binary: $BINARY" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "no such manifest: $MANIFEST" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/stk-assets" "$OUT/stk"

cp -r "$REPO/data" "$OUT/stk/data"
cp "$BINARY" "$OUT/supertuxkart"
chmod +x "$OUT/supertuxkart"

# The subset, structure preserved.
paths=$(grep -v '^#' "$MANIFEST" | grep -v '^[[:space:]]*$')

printf '%s\n' "$paths" | while IFS= read -r rel; do
    [ -f "$ASSETS/$rel" ] || continue
    mkdir -p "$OUT/stk-assets/$(dirname "$rel")"
    cp "$ASSETS/$rel" "$OUT/stk-assets/$rel"
done

# A manifest path that is not in the checkout means the two have drifted apart,
# and the pack would be missing exactly the kind of file that reports nothing at
# runtime: a texture resolves to a flat colour and the race still exits 0. Fail
# here instead, where it is still visible.
absent=$(printf '%s\n' "$paths" | while IFS= read -r rel; do
             [ -f "$OUT/stk-assets/$rel" ] || echo "$rel"
         done)
if [ -n "$absent" ]; then
    printf '%s\n' "$absent" | head -20 | sed "s|^|missing from $ASSETS: |" >&2
    echo "$(printf '%s\n' "$absent" | wc -l) manifest path(s) missing" >&2
    echo "regenerate it with tools/gym/trace_assets.py against this stk-assets" >&2
    exit 1
fi

for d in tracks karts library models music sfx textures; do
    mkdir -p "$OUT/stk-assets/$d"
done

echo "pack: $OUT"
echo "  files:  $(find "$OUT" -type f | wc -l)"
echo "  size:   $(du -sh "$OUT" | cut -f1)"
