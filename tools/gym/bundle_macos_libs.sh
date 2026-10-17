#!/bin/sh
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Make a macOS pack's binary self-contained: copy the libraries it loads into
# <pack>/lib and point it there.
#
# The binary links its libraries as @rpath/<name>, and the build sets the rpath
# to the absolute path of dependencies-macosx/lib, which exists only on the
# machine that built it. This copies the closure of those names -- each library
# loads its own neighbours by @rpath too, resolved against the executable's
# rpath -- and replaces the rpath with @executable_path/lib.
#
# Real files only, never the version symlinks the dependency tree is made of:
# the package unpacks the pack with tarfile's "data" filter, or on Pythons
# older than 3.11.4 with a check of its own that refuses every link.
#
# Editing the load commands voids the linker's ad hoc signature, and arm64
# refuses to run an unsigned binary, so it is signed again.
#
# Usage: bundle_macos_libs.sh <dependencies-lib-dir> <pack-dir>
set -eu
if [ $# -ne 2 ]; then
    sed -n '5,20p' "$0" >&2
    exit 2
fi
LIBS=$1
PACK=$2
BINARY=$PACK/supertuxkart
[ -d "$LIBS" ] || { echo "no such library directory: $LIBS" >&2; exit 1; }
[ -f "$BINARY" ] || { echo "no such binary: $BINARY" >&2; exit 1; }

rpath_names() {
    otool -L "$1" | tail -n +2 | awk '$1 ~ /^@rpath\// { sub("@rpath/", "", $1); print $1 }'
}

mkdir -p "$PACK/lib"
todo=$(rpath_names "$BINARY")
while [ -n "$todo" ]; do
    name=$(printf '%s\n' "$todo" | head -n 1)
    todo=$(printf '%s\n' "$todo" | tail -n +2)
    [ -f "$PACK/lib/$name" ] && continue
    [ -e "$LIBS/$name" ] || { echo "$BINARY needs $name, not in $LIBS" >&2; exit 1; }
    cp -L "$LIBS/$name" "$PACK/lib/$name"
    chmod u+w "$PACK/lib/$name"
    todo=$(printf '%s\n%s\n' "$todo" "$(rpath_names "$PACK/lib/$name")" | grep -v '^$' || true)
done

old=$(otool -l "$BINARY" | awk '/LC_RPATH/ { getline; getline; print $2 }')
for path in $old; do
    install_name_tool -delete_rpath "$path" "$BINARY"
done
install_name_tool -add_rpath @executable_path/lib "$BINARY"
codesign --force --sign - "$BINARY"
ls "$PACK/lib"
