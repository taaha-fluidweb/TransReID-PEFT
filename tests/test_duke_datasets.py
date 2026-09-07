"""Unit tests for Duke / Occ-Duke dataset helpers (no network)."""

from pathlib import Path

import pytest

from datasets.build_occ_duke import build_occ_duke_from_duke, _has_occ_duke_layout
from datasets.download_duke import _has_duke_layout, _normalize_extracted_layout


def test_has_duke_layout(tmp_path: Path) -> None:
    root = tmp_path / "dukemtmcreid"
    assert not _has_duke_layout(root)
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        (root / name).mkdir(parents=True)
    assert _has_duke_layout(root)


def test_normalize_extracted_layout_nested(tmp_path: Path) -> None:
    extract = tmp_path / "extract"
    nested = extract / "DukeMTMC-reID"
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        (nested / name).mkdir(parents=True)
        (nested / name / "0001_c1.jpg").write_bytes(b"x")

    _normalize_extracted_layout(extract)
    assert _has_duke_layout(extract)


def test_build_occ_duke_from_duke(tmp_path: Path) -> None:
    duke = tmp_path / "dukemtmcreid"
    occ = tmp_path / "Occluded_Duke"
    lists_dir = tmp_path / "lists"

    for folder in ("bounding_box_train", "bounding_box_test", "query"):
        (duke / folder).mkdir(parents=True)

    (duke / "bounding_box_train" / "train_a.jpg").write_bytes(b"a")
    (duke / "bounding_box_test" / "gallery_a.jpg").write_bytes(b"b")
    (duke / "bounding_box_test" / "query_from_gallery.jpg").write_bytes(b"c")

    lists_dir.mkdir()
    (lists_dir / "train.list").write_text("train_a.jpg\n", encoding="utf-8")
    (lists_dir / "gallery.list").write_text("gallery_a.jpg\n", encoding="utf-8")
    (lists_dir / "query.list").write_text("query_from_gallery.jpg\n", encoding="utf-8")

    build_occ_duke_from_duke(duke, occ, lists_dir)

    assert _has_occ_duke_layout(occ)
    assert (occ / "bounding_box_train" / "train_a.jpg").is_file()
    assert (occ / "bounding_box_test" / "gallery_a.jpg").is_file()
    assert (occ / "query" / "query_from_gallery.jpg").is_file()
