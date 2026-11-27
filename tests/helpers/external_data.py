from pathlib import Path

import yaml


def iter_collections(dataset_root: Path, version: str = "curated-v0"):
    collections_root = dataset_root / version / "collections"

    for config_path in sorted(collections_root.glob("*/collection.yaml")):
        config = yaml.safe_load(config_path.read_text())

        collection_dir = config_path.parent
        collection_id = config["collection_id"]

        yield {
            **config,
            "collection_dir": collection_dir,
            "dicom_dir": collection_dir / "dicom",
            "golden_dir": dataset_root / version / "golden" / collection_id,
        }


def iter_dro_case_dirs(root: Path) -> list[Path]:
    """Return the individual DRO case directories under ``root``.

    Expected layout::

        DRO/
          DRO_0_0/
          DRO_1_0/
          ...

    The root itself is treated as one case only if no subdirectory is a case.
    An empty list is returned when nothing looks like a DRO case.
    """

    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    case_dirs = [p for p in subdirs if _looks_like_dro_case_dir(p)]

    if case_dirs:
        return case_dirs

    if _looks_like_dro_case_dir(root):
        return [root]

    return []


def _looks_like_dro_case_dir(path: Path) -> bool:
    has_nifti_mask = any(path.rglob("*.nii")) or any(path.rglob("*.nii.gz"))
    has_files = any(p.is_file() for p in path.rglob("*"))
    return has_nifti_mask and has_files
