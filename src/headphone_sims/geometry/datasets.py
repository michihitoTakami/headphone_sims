"""Downloaders for public pinna/head mesh datasets (all CC BY 4.0).

Meshes are fetched on demand into ``data/`` and are not redistributed with
this repository. Cite the sources when publishing results:

- HUTUBS: Brinkmann et al. 2019, "A cross-evaluated database of measured and
  simulated HRTFs...", DOI 10.14279/depositonce-8487 (CC BY 4.0)
- Aachen high-resolution KEMAR: RWTH ITA, DOI 10.18154/RWTH-2020-11307 (CC BY 4.0)
- VIKING: Spagnol et al. 2019, Zenodo record 4160401 (CC BY 4.0)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

HUTUBS_MESH_URL = "https://sofacoustics.org/data/database/hutubs/pp{subject}_3DheadMesh.ply"
KEMAR_STL_URL = (
    "https://sofacoustics.org/data/database/aachen%20(high-resolution%20kemar)/Kemar_3D-Model.stl"
)
VIKING_INDEX_URL = "https://sofacoustics.org/data/database/viking/"

DEFAULT_DATA_DIR = Path("data")

ATTRIBUTIONS = {
    "hutubs": "Brinkmann et al. 2019, DOI 10.14279/depositonce-8487, CC BY 4.0",
    "kemar": "RWTH Aachen ITA, DOI 10.18154/RWTH-2020-11307, CC BY 4.0",
    "viking": "Spagnol et al. 2019, Zenodo 4160401, CC BY 4.0",
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def fetch_hutubs_mesh(subject: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Head+pinna PLY mesh for a HUTUBS subject (1..96; not all have meshes)."""
    url = HUTUBS_MESH_URL.format(subject=subject)
    return _download(url, data_dir / "hutubs" / f"pp{subject}_3DheadMesh.ply")


def fetch_kemar(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Aachen high-resolution KEMAR full-head STL."""
    return _download(KEMAR_STL_URL, data_dir / "kemar" / "Kemar_3D-Model.stl")


def fetch_viking_pinna(subject: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """VIKING left-pinna STL for subject letter A..T (fetched via index scrape)."""
    subject = subject.upper()
    index_html = urllib.request.urlopen(VIKING_INDEX_URL).read().decode("utf-8", errors="replace")
    candidates = [
        part.split('"')[0]
        for part in index_html.split('href="')[1:]
        if part.split('"')[0].lower().endswith(".stl")
    ]
    matches = [c for c in candidates if subject in Path(c).stem.upper()]
    if not matches:
        raise ValueError(
            f"no VIKING STL matching subject {subject!r}; index lists: {candidates[:20]}"
        )
    name = matches[0]
    return _download(VIKING_INDEX_URL + name, data_dir / "viking" / Path(name).name)
