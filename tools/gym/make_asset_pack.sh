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
# file in it: FileManager aborts with "Not all assets found" if one is missing.
#
# Sound is the exception to the manifest: it is taken by name here rather than
# traced. The trace cannot be trusted for it. The game loads a third of its sound
# effects lazily and starts a track's music only once the race is under way, so a
# trace under-reports -- and if the tracing machine has no audio device, OpenAL
# fails to open one, SuperTuxKart writes sfx_on="false" into its config and every
# later run is silent, the trace included. That is how the pack shipped through
# gym-v0.1.2 with no sound at all: both sfx/*.ogg and music/*.ogg pruned, while
# the .music descriptors stayed (the game reads those whether sound works or
# not), and a race logged "SFXBuffer: Could not load sound effect" 68 times.
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

# The sound, by name rather than from the manifest (see the header). All of sfx/
# is 5.5 MB, so it goes in whole -- cheaper than being right about which effects
# a race reaches. Music is 68 MB for 36 tracks of which the pack plays five, so
# only what those five name is taken: a track.xml names a .music descriptor and
# the descriptor names its ogg, plus a second one for the fast variant. About
# 11 MB, and the five tracks are the ones the manifest header already records.
sounds=$(cd "$ASSETS" && ls sfx/*.ogg)
for track in $(sed -n 's/^# tracks: //p' "$MANIFEST"); do
    for desc in $(grep -o '"[^"]*\.music"' "$ASSETS/tracks/$track/track.xml" | tr -d '"'); do
        [ -f "$ASSETS/music/$desc" ] || continue
        for ogg in $(grep -o '"[^"]*\.ogg"' "$ASSETS/music/$desc" | tr -d '"'); do
            sounds=$(printf '%s\nmusic/%s\n' "$sounds" "$ogg")
        done
    done
done
paths=$(printf '%s\n%s\n' "$paths" "$sounds" | grep -v '^[[:space:]]*$' | sort -u)

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
    echo "$(printf '%s\n' "$absent" | wc -l) path(s) missing" >&2
    echo "for a traced path, regenerate the manifest with tools/gym/trace_assets.py" >&2
    echo "against this stk-assets; for a sfx/ or music/ one, the checkout is short" >&2
    echo "of sound files and needs an svn up" >&2
    exit 1
fi

for d in tracks karts library models music sfx textures; do
    mkdir -p "$OUT/stk-assets/$d"
done

echo "pack: $OUT"
echo "  files:  $(find "$OUT" -type f | wc -l)"
echo "  size:   $(du -sh "$OUT" | cut -f1)"
