import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_snapshot_dir(tmp_path: Path) -> str:
    d = tmp_path / "snapshots"
    d.mkdir()
    return str(d)
