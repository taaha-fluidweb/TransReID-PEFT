"""Download and prepare the Market-1501 dataset."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Official Google Drive (Zheng Lab): https://zheng-lab-anu.github.io/Project/project_reid.html
MARKET1501_GDRIVE_ID = "0B8-rUzbwVRk0c054eEozWG9COHM"
MARKET1501_ZIP_NAME = "Market-1501-v15.09.15.zip"
MARKET1501_HTTP_MIRRORS = (
    "http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip",
)
LOCAL_ZIP_NAMES = ("market1501.zip", "Market-1501.zip", "Market1501.zip", MARKET1501_ZIP_NAME)


def _has_market_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "bounding_box_train").is_dir()
        and (path / "query").is_dir()
        and (path / "bounding_box_test").is_dir()
    )


def _normalize_to_market1501(data_dir: Path) -> Path:
    """Ensure data/market1501/ exists with the expected folder layout."""
    target = data_dir / "market1501"
    if _has_market_layout(target):
        return target

    search_roots = [data_dir, *data_dir.iterdir()] if data_dir.is_dir() else [data_dir]
    for candidate in search_roots:
        if not candidate.is_dir():
            continue
        if _has_market_layout(candidate):
            if candidate.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    shutil.rmtree(target)
                candidate.rename(target)
            return target

    raise RuntimeError(
        f"Could not find Market-1501 folders under {data_dir}. "
        "Expected bounding_box_train/, query/, bounding_box_test/."
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

    url = f"https://drive.google.com/uc?id={MARKET1501_GDRIVE_ID}"
    print(f"Downloading Market-1501 from Google Drive...")
    gdown.download(url, str(zip_path), quiet=False)


def _download_via_http(url: str, zip_path: Path) -> None:
    print(f"Downloading Market-1501 from {url}")
    urllib.request.urlretrieve(url, zip_path)


def _find_local_zip(search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for name in LOCAL_ZIP_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def ensure_market1501(root: str | Path = "data", search_dirs: list[Path] | None = None) -> Path:
    """
    Ensure Market-1501 exists at {root}/market1501/.

    Tries, in order:
    1. Existing extracted dataset
    2. Local zip uploaded by user (market1501.zip, etc.)
    3. Google Drive (official)
    4. HTTP mirror(s)
    """
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "market1501"

    if _has_market_layout(target):
        print("Market-1501 already present.")
        return target

    dirs = search_dirs or [data_dir, data_dir.parent, Path.cwd()]
    local_zip = _find_local_zip(dirs)
    zip_path = data_dir / MARKET1501_ZIP_NAME

    if local_zip and local_zip.resolve() != zip_path.resolve():
        print(f"Using local zip: {local_zip}")
        shutil.copy2(local_zip, zip_path)
    elif not zip_path.is_file():
        errors: list[str] = []
        try:
            _download_via_gdown(zip_path)
        except Exception as exc:
            errors.append(f"Google Drive: {exc}")
            for url in MARKET1501_HTTP_MIRRORS:
                try:
                    _download_via_http(url, zip_path)
                    break
                except (urllib.error.URLError, OSError, TimeoutError) as mirror_exc:
                    errors.append(f"{url}: {mirror_exc}")
            else:
                raise RuntimeError(
                    "Failed to download Market-1501.\n" + "\n".join(errors)
                ) from exc

    tmp_dir = data_dir / "_market1501_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _extract_zip(zip_path, tmp_dir)

    # Zip may contain folders at root or under one subdirectory
    if not _has_market_layout(tmp_dir):
        for child in tmp_dir.iterdir():
            if child.is_dir() and _has_market_layout(child):
                for item in child.iterdir():
                    dest = tmp_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                break

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp_dir), str(target))

    if not zip_path.exists() or zip_path.stat().st_size > 0:
        # Keep zip for reuse; optional cleanup could delete tmp only
        pass

    print(f"Market-1501 ready at {target}")
    return target
