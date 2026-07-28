"""Build the Occluded-DukeMTMC dataset from DukeMTMC-reID."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .download_duke import ensure_dukemtmcreid

# Official name lists (ICCV 2019 PGFA paper):
# https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID/tree/master/dataset
OCC_DUKE_LIST_BASE = (
    "https://raw.githubusercontent.com/lightas/"
    "ICCV19_Pose_Guided_Occluded_Person_ReID/master/dataset/Occluded_Duke"
)
OCC_DUKE_SPLITS = (
    ("train", "bounding_box_train"),
    ("gallery", "bounding_box_test"),
    ("query", "query"),
)


def _has_occ_duke_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "bounding_box_train").is_dir()
        and (path / "query").is_dir()
        and (path / "bounding_box_test").is_dir()
    )


def _ensure_list_files(lists_dir: Path) -> None:
    lists_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for split, _ in OCC_DUKE_SPLITS:
        dest = lists_dir / f"{split}.list"
        if dest.is_file():
            continue
        url = f"{OCC_DUKE_LIST_BASE}/{split}.list"
        print(f"Downloading Occ-Duke split list: {split}.list")
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
    if errors:
        raise RuntimeError(
            "Failed to download Occluded-Duke name lists.\n" + "\n".join(errors)
        )


def build_occ_duke_from_duke(duke_dir: Path, occ_dir: Path, lists_dir: Path) -> None:
    """Copy/re-split Duke images into Occluded-Duke folders using official lists."""
    _ensure_list_files(lists_dir)
    occ_dir.mkdir(parents=True, exist_ok=True)

    for split, folder_name in OCC_DUKE_SPLITS:
        target_split = occ_dir / folder_name
        if target_split.exists():
            shutil.rmtree(target_split)
        target_split.mkdir(parents=True)

        source_split = duke_dir / folder_name
        list_path = lists_dir / f"{split}.list"
        with open(list_path, encoding="utf-8") as handle:
            imgs = [line.strip() for line in handle if line.strip()]

        for img in imgs:
            target_path = target_split / img
            source_path = source_split / img
            if not source_path.is_file():
                # Occluded query images are moved from gallery in the official split.
                source_path = duke_dir / "bounding_box_test" / img
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Missing Duke image for Occ-Duke ({split}): {img}"
                )
            shutil.copy2(source_path, target_path)

        print(f"Occ-Duke {folder_name}: {len(imgs)} images")


def ensure_occ_duke(root: str | Path = "data", search_dirs: list[Path] | None = None) -> Path:
    """
    Ensure Occluded-DukeMTMC exists at {root}/Occluded_Duke/.

    Built from DukeMTMC-reID using official split lists from the ICCV 2019 PGFA
    repo (images are not distributed separately due to DukeMTMC privacy).
    """
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "Occluded_Duke"

    if _has_occ_duke_layout(target):
        print("Occluded-Duke already present.")
        return target

    duke_dir = ensure_dukemtmcreid(root=root, search_dirs=search_dirs)
    lists_dir = Path(__file__).resolve().parent / "occluded_duke_lists"

    if target.exists():
        shutil.rmtree(target)
    build_occ_duke_from_duke(duke_dir, target, lists_dir)

    if not _has_occ_duke_layout(target):
        raise RuntimeError(
            f"Failed to build Occluded-Duke at {target}. "
            "Expected bounding_box_train/, query/, bounding_box_test/."
        )

    print(f"Occluded-Duke ready at {target}")
    return target
