"""Download and prepare the DukeMTMC-reID dataset."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Google Drive mirror (layumi / community): https://github.com/layumi/DukeMTMC-reID_evaluation
DUKE_GDRIVE_ID = "1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O"
DUKE_ZIP_NAME = "DukeMTMC-reID.zip"
LOCAL_ZIP_NAMES = ("dukemtmcreid.zip", "duke.zip", "DukeMTMC-reID.zip", DUKE_ZIP_NAME)


def _has_duke_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "bounding_box_train").is_dir()
        and (path / "query").is_dir()
        and (path / "bounding_box_test").is_dir()
    )


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _download_via_gdown(zip_path: Path) -> None:
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={DUKE_GDRIVE_ID}"
    print("Downloading DukeMTMC-reID from Google Drive...")
    gdown.download(url, str(zip_path), quiet=False)


def _find_local_zip(search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for name in LOCAL_ZIP_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _normalize_extracted_layout(tmp_dir: Path) -> None:
    if _has_duke_layout(tmp_dir):
        return
    for child in tmp_dir.iterdir():
        if child.is_dir() and _has_duke_layout(child):
            for item in child.iterdir():
                dest = tmp_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            break


def ensure_dukemtmcreid(root: str | Path = "data", search_dirs: list[Path] | None = None) -> Path:
    """
    Ensure DukeMTMC-reID exists at {root}/dukemtmcreid/.

    Tries, in order:
    1. Existing extracted dataset
    2. Local zip uploaded by user (DukeMTMC-reID.zip, etc.)
    3. Google Drive
    """
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "dukemtmcreid"

    if _has_duke_layout(target):
        print("DukeMTMC-reID already present.")
        return target

    dirs = search_dirs or [data_dir, data_dir.parent, Path.cwd()]
    local_zip = _find_local_zip(dirs)
    zip_path = data_dir / DUKE_ZIP_NAME

    if local_zip and local_zip.resolve() != zip_path.resolve():
        print(f"Using local zip: {local_zip}")
        shutil.copy2(local_zip, zip_path)
    elif not zip_path.is_file():
        try:
            _download_via_gdown(zip_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download DukeMTMC-reID from Google Drive.\n"
                f"Upload {DUKE_ZIP_NAME} to data/ and retry.\n"
                f"Original error: {exc}"
            ) from exc

    tmp_dir = data_dir / "_dukemtmcreid_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _extract_zip(zip_path, tmp_dir)
    _normalize_extracted_layout(tmp_dir)

    if not _has_duke_layout(tmp_dir):
        raise RuntimeError(
            f"Extracted DukeMTMC-reID zip at {tmp_dir} is missing expected folders "
            "(bounding_box_train/, query/, bounding_box_test/)."
        )

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp_dir), str(target))

    print(f"DukeMTMC-reID ready at {target}")
    return target
