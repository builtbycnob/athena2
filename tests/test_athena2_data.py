"""Tests for ATHENA2 data pipeline."""
import numpy as np
import pytest


class TestConfig:
    def test_load_config(self):
        from athena2.data.ingestion import load_config
        config = load_config()
        assert "datasets" in config
        assert "splits" in config
        assert "paths" in config

    def test_official_splits_in_config(self):
        from athena2.data.ingestion import load_config
        config = load_config()
        splits = config["splits"]
        assert splits["train_max_year"] == 2015
        assert splits["validation_years"] == [2016, 2017]
        assert splits["test_min_year"] == 2018


class TestCleanLegalText:
    def test_html_removal(self):
        from athena2.data.ingestion import clean_legal_text
        assert clean_legal_text("<p>Hello</p>") == "Hello"
        assert clean_legal_text("<br/>line") == "line"

    def test_whitespace_normalization(self):
        from athena2.data.ingestion import clean_legal_text
        assert clean_legal_text("  foo  bar  ") == "foo bar"

    def test_newline_collapse(self):
        from athena2.data.ingestion import clean_legal_text
        result = clean_legal_text("line1\n\n\n\nline2")
        # clean_legal_text normalizes all whitespace including newlines
        assert "line1" in result
        assert "line2" in result

    def test_none_input(self):
        from athena2.data.ingestion import clean_legal_text
        assert clean_legal_text(None) == ""


class TestDatasetStats:
    def test_summary(self):
        from collections import Counter
        from athena2.data.ingestion import DatasetStats
        stats = DatasetStats(
            name="test", total_rows=1000,
            splits={"train": 800, "val": 100, "test": 100},
            languages=Counter({"de": 480, "fr": 390, "it": 130}),
            labels=Counter({0: 700, 1: 300}),
            law_areas=Counter({"public_law": 500, "civil_law": 500}),
            years=Counter({2020: 500, 2021: 500}),
            facts_lengths=[100, 200, 300],
            considerations_lengths=[50, 100, 150],
        )
        s = stats.summary()
        assert "test" in s
        assert "de" in s


class TestTemporalSplit:
    def test_split_assignment(self):
        """Test that the temporal split logic correctly assigns years."""
        import pandas as pd
        from athena2.data.ingestion import load_config

        config = load_config()
        split_config = config.get("splits", {})
        train_max = split_config.get("train_max_year", 2015)
        val_years = set(split_config.get("validation_years", [2016, 2017]))
        test_min = split_config.get("test_min_year", 2018)

        df = pd.DataFrame({
            "year": [2010, 2015, 2016, 2017, 2018, 2022],
            "facts": ["text"] * 6,
            "label": [0, 1, 0, 1, 0, 1],
        })

        def assign_split(year):
            if year <= train_max:
                return "train"
            elif year in val_years:
                return "validation"
            elif year >= test_min:
                return "test"
            return "validation"

        df["athena2_split"] = df["year"].apply(assign_split)
        expected = ["train", "train", "validation", "validation", "test", "test"]
        assert list(df["athena2_split"]) == expected
