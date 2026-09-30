import os
from pathlib import Path

import pytest

from recon_rag.datagen import generate


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("data") / "raw"
    truth = generate(out, n_orders=600, seed=123)
    return out, truth


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="integration tests need DATABASE_URL (Postgres with pgvector)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
