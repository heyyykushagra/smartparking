"""Shared test fixtures."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


@pytest.fixture(autouse=True)
def isolated_position_file(tmp_path, monkeypatch):
    """Never let a test read or overwrite the real CarParkPos.

    EnhancedParkingDetector.save_parking_positions() writes to
    config.POSITION_FILE, so point it at a temp file for every test.
    """
    monkeypatch.setattr(config, "POSITION_FILE", str(tmp_path / "CarParkPos"))
