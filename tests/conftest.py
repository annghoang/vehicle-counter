import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "ROI_DIR", tmp_path / "roi")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    return tmp_path
