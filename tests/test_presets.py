"""Preset name resolution — the --preset-by-name path added this session.

It had no test at all, which an audit caught. It is the mechanism that lets a
tuning session reproduce on hardware (`--preset zmz`), so it is worth pinning:
the browser addresses presets by bare name and the CLI by name OR path, and
those two must not drift apart again.
"""

from __future__ import annotations

import pytest

from midas_hand_retargeter import presets
from midas_hand_retargeter.params import RetargetProfile


def test_a_bare_name_resolves_into_the_preset_directory():
    resolved = presets.resolve("zmz")
    assert resolved.name == "zmz.json"
    assert resolved.parent == presets.DEFAULT_PRESET_DIR


def test_a_path_is_taken_as_given(tmp_path):
    explicit = tmp_path / "somewhere" / "mine.json"
    assert presets.resolve(explicit) == explicit
    # A bare filename ending in .json is a path, not a name in the directory.
    assert presets.resolve("mine.json").name == "mine.json"
    assert presets.resolve("mine.json").parent != presets.DEFAULT_PRESET_DIR


def test_resolve_honours_an_explicit_directory(tmp_path):
    assert presets.resolve("mine", directory=tmp_path) == tmp_path / "mine.json"


def test_a_missing_preset_lists_what_is_available(tmp_path):
    """So the error is actionable rather than just a path."""

    presets.save(tmp_path / "alpha.json", RetargetProfile())
    with pytest.raises(FileNotFoundError) as excinfo:
        presets.load(tmp_path / "nope.json")
    assert "nope.json" in str(excinfo.value)
    assert "available:" in str(excinfo.value)


def test_save_and_load_round_trip_by_name(tmp_path):
    """The property the CLI depends on: what the tuner saved by name is what
    --preset loads by that same name."""

    saved = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.23, "dexpilot.abduction_limit": 0.11}
    )
    presets.save(
        presets.resolve("rt", directory=tmp_path), saved, neutral_offsets={"index_pip_joint": -0.25}
    )

    loaded, neutral = presets.load(presets.resolve("rt", directory=tmp_path))
    assert loaded.dexpilot.scaling_factor == 1.23
    assert loaded.dexpilot.abduction_limit == 0.11
    assert neutral == {"index_pip_joint": -0.25}
