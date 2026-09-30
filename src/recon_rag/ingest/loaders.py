import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Values a source never sends but that are implied by which feed it is
SOURCE_DEFAULTS: dict[str, dict[str, Any]] = {
    "wh_na": {"region": "NA"},
    "wh_emea": {"region": "EMEA"},
    "wh_apac": {"region": "APAC"},
}


@dataclass
class Batch:
    source: str
    name: str
    rows: list[dict[str, Any]]


def read_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("records", [])
    raise ValueError(f"unsupported file type: {path}")


def discover_batches(data_dir: Path) -> list[Batch]:
    """Every sub-directory of ``data_dir`` is a source; its files are batches in lexical order."""
    batches = []
    for source_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for f in sorted(source_dir.iterdir()):
            if f.suffix in (".csv", ".json"):
                batches.append(Batch(source=source_dir.name, name=f.stem, rows=read_file(f)))
    return batches
