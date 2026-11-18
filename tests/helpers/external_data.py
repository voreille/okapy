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