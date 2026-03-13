# tests/test_cli.py
"""Tests for the ATHENA CLI entry point."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
import yaml

from athena.cli import _parse_args, main


class TestParseArgs:
    """Test argparse configuration."""

    def test_run_command_parses(self):
        args = _parse_args(["run", "--case", "c.yaml", "--simulation", "s.yaml", "--output", "out/"])
        assert args.command == "run"
        assert args.case == "c.yaml"
        assert args.simulation == "s.yaml"
        assert args.output == "out/"

    def test_missing_command_exits(self):
        """No subcommand should result in command=None."""
        args = _parse_args([])
        assert args.command is None

    def test_missing_required_arg_exits(self):
        with pytest.raises(SystemExit):
            _parse_args(["run", "--case", "c.yaml"])

    def test_concurrency_flag(self):
        args = _parse_args(["run", "--case", "c.yaml", "--simulation", "s.yaml", "--output", "out/", "--concurrency", "2"])
        assert args.concurrency == 2

    def test_concurrency_flag_default_none(self):
        args = _parse_args(["run", "--case", "c.yaml", "--simulation", "s.yaml", "--output", "out/"])
        assert args.concurrency is None


class TestMainPipeline:
    """Test the full pipeline with mocked orchestrator and LLM."""

    CASE_DATA = {
        "case": {
            "id": "test-001",
            "type": "tax",
            "jurisdiction": "IT",
        },
        "parties": [],
        "arguments": [],
    }

    SIM_DATA = {
        "case_ref": "test-001",
        "language": "it",
        "judge_profiles": [
            {"id": "formalist", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
        ],
        "appellant_profiles": [
            {"id": "aggressive", "style": "aggressive"},
        ],
        "temperature": {"appellant": 0.7, "respondent": 0.3, "judge": 0.2},
        "runs_per_combination": 1,
    }

    MOCK_RESULT = {
        "run_id": "formalist__aggressive__000",
        "judge_profile": "formalist",
        "appellant_profile": "aggressive",
        "appellant_brief": {"text": "test brief"},
        "respondent_brief": {"text": "test response"},
        "judge_decision": {
            "verdict": {
                "qualification_correct": True,
            },
            "argument_evaluation": [
                {
                    "argument_id": "arg1",
                    "persuasiveness": 7,
                    "determinative": True,
                },
            ],
            "precedent_analysis": {
                "prec1": {"followed": True, "distinguished": False},
            },
        },
        "validation_warnings": {
            "appellant": [],
            "respondent": [],
            "judge": [],
        },
    }

    def test_full_pipeline(self, tmp_path):
        """Full pipeline runs with mocked orchestrator, produces all outputs."""
        case_file = tmp_path / "case.yaml"
        sim_file = tmp_path / "sim.yaml"
        output_dir = tmp_path / "output"

        case_file.write_text(yaml.dump(self.CASE_DATA))
        sim_file.write_text(yaml.dump(self.SIM_DATA))

        with patch("athena.simulation.orchestrator.run_monte_carlo", return_value=[self.MOCK_RESULT]) as mock_mc, \
             patch("athena.output.memo.generate_strategic_memo", return_value="# Memo\n\nTest memo.") as mock_memo:

            main(["run", "--case", str(case_file), "--simulation", str(sim_file), "--output", str(output_dir)])

            mock_mc.assert_called_once()
            mock_memo.assert_called_once()

        # Verify all output files exist
        assert (output_dir / "probability_table.md").exists()
        assert (output_dir / "decision_tree.txt").exists()
        assert (output_dir / "strategic_memo.md").exists()
        assert (output_dir / "raw_results.json").exists()

        # Verify raw results content
        with open(output_dir / "raw_results.json") as f:
            raw = json.load(f)
        assert len(raw) == 1
        assert raw[0]["run_id"] == "formalist__aggressive__000"

        # Verify table and tree have content
        table = (output_dir / "probability_table.md").read_text()
        assert "formalist" in table.lower() or "Profilo" in table

        tree = (output_dir / "decision_tree.txt").read_text()
        assert "formalist" in tree

    def test_memo_failure_does_not_crash(self, tmp_path):
        """If LLM memo generation fails, CLI saves placeholder and continues."""
        case_file = tmp_path / "case.yaml"
        sim_file = tmp_path / "sim.yaml"
        output_dir = tmp_path / "output"

        case_file.write_text(yaml.dump(self.CASE_DATA))
        sim_file.write_text(yaml.dump(self.SIM_DATA))

        with patch("athena.simulation.orchestrator.run_monte_carlo", return_value=[self.MOCK_RESULT]), \
             patch("athena.output.memo.generate_strategic_memo", side_effect=RuntimeError("No LLM available")):

            main(["run", "--case", str(case_file), "--simulation", str(sim_file), "--output", str(output_dir)])

        memo = (output_dir / "strategic_memo.md").read_text()
        assert "failed" in memo.lower()
        # Other outputs should still exist
        assert (output_dir / "probability_table.md").exists()
        assert (output_dir / "decision_tree.txt").exists()
        assert (output_dir / "raw_results.json").exists()


class TestImports:
    """Verify CLI module imports correctly."""

    def test_module_imports(self):
        import athena.cli
        assert hasattr(athena.cli, "main")
        assert hasattr(athena.cli, "_parse_args")
