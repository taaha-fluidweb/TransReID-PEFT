"""Download and prepare the MSMT17 dataset."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Google Drive ID for MSMT17 v1 / v2 zip
MSMT17_GDRIVE_ID = "1c3k5R_B3cE2N1y-W2Kms9tJ0B89gZ2L0"
MSMT17_ZIP_NAME = "MSMT17_V1.zip"
LOCAL_ZIP_NAMES = ("MSMT17.zip", "msmt17.zip", "MSMT17_V1.zip", "MSMT17_V2.zip")


def _has_msmt17_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train").is_dir()
        and (path / "test").is_dir()
        and (path / "list_train.txt").is_file()
        and (path / "list_query.txt").is_file()
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

    url = f"https://drive.google.com/uc?id={MSMT17_GDRIVE_ID}"
    print("Downloading MSMT17 from Google Drive...")
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
    if _has_msmt17_layout(tmp_dir):
        return
    for child in tmp_dir.iterdir():
        if child.is_dir() and _has_msmt17_layout(child):
            for item in child.iterdir():
                dest = tmp_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            break


def ensure_msmt17(root: str | Path = "data", search_dirs: list[Path] | None = None) -> Path:
    """
    Ensure MSMT17 exists at {root}/MSMT17/ or {root}/msmt17/.

    Tries, in order:
    1. Existing extracted dataset
    2. Local zip uploaded by user (MSMT17.zip, msmt17.zip, etc.)
    3. Google Drive / HTTP mirror download
    """
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)

    target_msmt = data_dir / "MSMT17"
    target_lower = data_dir / "msmt17"

    if _has_msmt17_layout(target_msmt):
        print("MSMT17 already present at data/MSMT17.")
        return target_msmt

    if _has_msmt17_layout(target_lower):
        print("MSMT17 already present at data/msmt17.")
        return target_lower

    dirs = search_dirs or [data_dir, data_dir.parent, Path.cwd()]
    local_zip = _find_local_zip(dirs)
    zip_path = data_dir / MSMT17_ZIP_NAME

    if local_zip and local_zip.resolve() != zip_path.resolve():
        print(f"Using local zip: {local_zip}")
        shutil.copy2(local_zip, zip_path)
    elif not zip_path.is_file():
        try:
            _download_via_gdown(zip_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download MSMT17 from Google Drive.\n"
                f"Please upload MSMT17.zip or MSMT17_V1.zip to data/ and retry.\n"
                f"Original error: {exc}"
            ) from exc

    tmp_dir = data_dir / "_msmt17_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _extract_zip(zip_path, tmp_dir)
    _normalize_extracted_layout(tmp_dir)

    if not _has_msmt17_layout(tmp_dir):
        raise RuntimeError(
            f"Extracted MSMT17 zip at {tmp_dir} is missing expected files "
            "(train/, test/, list_train.txt, list_query.txt)."
        )

    if target_msmt.exists():
        shutil.rmtree(target_msmt)
    shutil.move(str(tmp_dir), str(target_msmt))

    print(f"MSMT17 ready at {target_msmt}")
    return target_msmt
