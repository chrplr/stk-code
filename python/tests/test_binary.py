# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Locating the binary and the directory to run it from.

These build directory trees rather than starting the game: what is being
checked is which of several plausible roots wins, and that is decided before
anything is launched.
"""

from __future__ import annotations

import pytest

from stk_gym.binary import default_cwd, find_pack


def make_pack(root):
    """A pack as make_asset_pack.sh lays one out."""
    (root / "stk" / "data").mkdir(parents=True)
    (root / "stk-assets" / "tracks").mkdir(parents=True)
    binary = root / "supertuxkart"
    binary.write_bytes(b"")
    return binary


def make_checkout(root):
    """Enough of a stk-code checkout for the marker to be found."""
    (root / "src" / "gym").mkdir(parents=True)
    (root / "src" / "gym" / "gym_server.hpp").write_text("")
    (root / "data").mkdir()
    return root


@pytest.fixture(autouse=True)
def no_datadir(monkeypatch):
    """SUPERTUXKART_DATADIR short-circuits default_cwd, so keep it out."""
    monkeypatch.delenv("SUPERTUXKART_DATADIR", raising=False)


def test_pack_is_served_its_own_stk_directory(tmp_path):
    binary = make_pack(tmp_path / "pack")
    assert find_pack(binary) == tmp_path / "pack"
    assert default_cwd(binary) == str(tmp_path / "pack" / "stk")


def test_checkout_build_is_served_the_checkout(tmp_path):
    repo = make_checkout(tmp_path / "stk-code")
    binary = repo / "build" / "bin" / "supertuxkart"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    assert find_pack(binary) is None
    assert default_cwd(binary) == str(repo)


def test_a_pack_inside_a_checkout_is_still_served_by_the_pack(tmp_path):
    """The pack wins over a checkout that merely encloses it.

    The release workflow assembles the pack inside the checkout it just built,
    so this is the normal arrangement, not a contrived one. Serving the
    checkout instead sends the game looking for assets beside the checkout,
    where there are none, and it dies during the hello handshake with nothing
    on stdout to say why.
    """
    repo = make_checkout(tmp_path / "stk-code")
    binary = make_pack(repo / "pack")
    assert default_cwd(binary) == str(repo / "pack" / "stk")


def test_datadir_in_the_environment_wins(tmp_path, monkeypatch):
    binary = make_pack(tmp_path / "pack")
    monkeypatch.setenv("SUPERTUXKART_DATADIR", str(tmp_path / "elsewhere"))
    assert default_cwd(binary) is None
