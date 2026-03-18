"""Tests for data ingestion."""

import numpy as np
import pytest


class TestCSVIngester:
    def test_load_basic_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,c\n1,2,3\n4,5,6\n")
        pytest.skip("CSVIngester not yet implemented")

    def test_auto_type_detection(self, tmp_path):
        pytest.skip("CSVIngester not yet implemented")


class TestNumpyIngester:
    def test_load_ndarray(self):
        pytest.skip("NumpyIngester not yet implemented")

    def test_load_dataframe(self):
        pytest.skip("NumpyIngester not yet implemented")
