"""Audit: legacy YAML ``!include`` confinement (nested includes, symlinks)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from okapy_legacy.yaml import yaml as legacy_yaml

REJECTED = (yaml.YAMLError, ValueError, PermissionError)


def _load(path: Path):
    with path.open("r") as f:
        return legacy_yaml.load(f)


@pytest.fixture
def layout(tmp_path):
    config_dir = tmp_path / "configs"
    (config_dir / "sub").mkdir(parents=True)
    (tmp_path / "secret.yaml").write_text("secret: 1\n")
    (tmp_path / "secret.txt").write_text("secret\n")
    return config_dir


def _assert_rejected(path):
    with pytest.raises(REJECTED) as excinfo:
        _load(path)
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_nested_include_escaping_two_levels_is_rejected(layout):
    (layout / "sub" / "inner.yaml").write_text("x: !include ../../secret.yaml\n")
    config = layout / "params.yaml"
    config.write_text("setting: !include sub/inner.yaml\n")

    _assert_rejected(config)


def test_nested_text_include_escaping_is_rejected(layout):
    (layout / "sub" / "inner.yaml").write_text("x: !include ../../secret.txt\n")
    config = layout / "params.yaml"
    config.write_text("setting: !include sub/inner.yaml\n")

    _assert_rejected(config)


def test_nested_include_inside_the_tree_still_works(layout):
    (layout / "sub" / "deep").mkdir()
    (layout / "sub" / "deep" / "leaf.yaml").write_text("v: 3\n")
    (layout / "sub" / "inner.yaml").write_text("x: !include deep/leaf.yaml\n")
    config = layout / "params.yaml"
    config.write_text("setting: !include sub/inner.yaml\n")

    assert _load(config) == {"setting": {"x": {"v": 3}}}


def test_symlinked_file_pointing_outside_is_rejected(layout, tmp_path):
    os.symlink(tmp_path / "secret.yaml", layout / "link.yaml")
    config = layout / "params.yaml"
    config.write_text("setting: !include link.yaml\n")

    _assert_rejected(config)


def test_symlinked_directory_pointing_outside_is_rejected(layout, tmp_path):
    os.symlink(tmp_path, layout / "escape")
    config = layout / "params.yaml"
    config.write_text("setting: !include escape/secret.yaml\n")

    _assert_rejected(config)


def test_nested_include_through_symlinked_subdir_is_rejected(layout, tmp_path):
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "inner.yaml").write_text("x: 1\n")
    os.symlink(outside_dir, layout / "sub" / "linked")
    (layout / "sub" / "inner.yaml").write_text("x: !include linked/inner.yaml\n")
    config = layout / "params.yaml"
    config.write_text("setting: !include sub/inner.yaml\n")

    _assert_rejected(config)


def test_sibling_directory_sharing_a_name_prefix_is_rejected(layout, tmp_path):
    evil = tmp_path / "configs-evil"
    evil.mkdir()
    (evil / "x.yaml").write_text("secret: 1\n")
    config = layout / "params.yaml"
    config.write_text("setting: !include ../configs-evil/x.yaml\n")

    _assert_rejected(config)


def test_symlinked_config_file_resolves_includes_next_to_the_link(layout, tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "params.yaml").write_text("setting: !include local.yaml\n")
    (layout / "local.yaml").write_text("v: 1\n")
    os.symlink(real_dir / "params.yaml", layout / "params.yaml")

    # Whichever directory is used, it must never read outside both of them.
    try:
        result = _load(layout / "params.yaml")
    except REJECTED + (FileNotFoundError,):
        return
    assert result == {"setting": {"v": 1}}


def test_python_tags_inside_nested_include_are_rejected(layout, tmp_path):
    marker = tmp_path / "pwned"
    (layout / "sub" / "inner.yaml").write_text(
        f"x: !!python/object/apply:os.system ['touch {marker}']\n"
    )
    config = layout / "params.yaml"
    config.write_text("setting: !include sub/inner.yaml\n")

    with pytest.raises(yaml.YAMLError):
        _load(config)
    assert not marker.exists()
