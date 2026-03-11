# tests/test_game_theory.py
"""Known-answer tests for the game theory module."""

import pytest

from athena.game_theory.schemas import (
    GameTheoryAnalysis,
    OutcomeValuation,
    BATNA,
    SettlementRange,
    SensitivityResult,
)
from athena.game_theory.valuation import compute_outcome_values, compute_status_quo
from athena.game_theory.equilibrium import (
    compute_weighted_probabilities,
    compute_batna,
    compute_ev_by_strategy,
    compute_settlement_range,
)
from athena.game_theory.sensitivity import (
    find_threshold,
    sensitivity_litigation_cost,
    sensitivity_rejection_probability,
    sensitivity_fine_amount,
    run_all_sensitivity,
)
from athena.game_theory import analyze
from athena.output.game_theory_summary import format_game_theory_summary


# ---- Fixtures ----

@pytest.fixture
def stakes():
    """Standard stakes from gdp-milano case."""
    return {
        "current_sanction": {
            "norm": "art. 143 CdS",
            "fine_range": [170, 680],
            "points_deducted": 4,
        },
        "alternative_sanction": {
            "norm": "artt. 6-7 CdS",
            "fine_range": [42, 173],
            "points_deducted": 0,
        },
        "litigation_cost_estimate": 1500,
    }


@pytest.fixture
def simple_prob_table():
    """Probability table with known values for hand calculation."""
    return {
        ("formalista", "aggressivo"): {
            "n_runs": 10,
            "p_rejection": 0.5,
            "p_annulment": 0.3,
            "p_reclassification": 0.2,
            "ci_rejection": (0.3, 0.7),
            "ci_annulment": (0.15, 0.5),
            "ci_reclassification": (0.07, 0.42),
        },
    }


@pytest.fixture
def multi_prob_table():
    """Probability table with 2 judges x 2 strategies."""
    return {
        ("formalista", "aggressivo"): {
            "n_runs": 5,
            "p_rejection": 0.6,
            "p_annulment": 0.2,
            "p_reclassification": 0.2,
            "ci_rejection": (0.3, 0.85),
            "ci_annulment": (0.05, 0.5),
            "ci_reclassification": (0.05, 0.5),
        },
        ("formalista", "tecnico"): {
            "n_runs": 5,
            "p_rejection": 0.4,
            "p_annulment": 0.4,
            "p_reclassification": 0.2,
            "ci_rejection": (0.15, 0.7),
            "ci_annulment": (0.15, 0.7),
            "ci_reclassification": (0.05, 0.5),
        },
        ("garantista", "aggressivo"): {
            "n_runs": 5,
            "p_rejection": 0.2,
            "p_annulment": 0.6,
            "p_reclassification": 0.2,
            "ci_rejection": (0.05, 0.5),
            "ci_annulment": (0.3, 0.85),
            "ci_reclassification": (0.05, 0.5),
        },
        ("garantista", "tecnico"): {
            "n_runs": 5,
            "p_rejection": 0.2,
            "p_annulment": 0.4,
            "p_reclassification": 0.4,
            "ci_rejection": (0.05, 0.5),
            "ci_annulment": (0.15, 0.7),
            "ci_reclassification": (0.15, 0.7),
        },
    }


# ---- Task 1: Schema tests ----

class TestSchemas:
    def test_import(self):
        from athena.game_theory.schemas import GameTheoryAnalysis
        assert GameTheoryAnalysis is not None

    def test_outcome_valuation_roundtrip(self):
        ov = OutcomeValuation(
            outcome="rejection",
            description="test",
            fine=425.0,
            fine_range=(170.0, 680.0),
            points=4,
            net_value=-1925.0,
        )
        assert ov.model_dump()["net_value"] == -1925.0

    def test_settlement_range_no_zopa(self):
        sr = SettlementRange(
            zopa=None, nash_solution=None, surplus=100.0, settlement_exists=False,
        )
        assert sr.settlement_exists is False
        assert sr.zopa is None


# ---- Task 2: Valuation tests ----

class TestOutcomeValuation:
    def test_appellant_rejection(self, stakes):
        vals = compute_outcome_values(stakes, "appellant")
        # rejection: -(425 + 1500) = -1925
        assert vals["rejection"].net_value == pytest.approx(-1925.0)
        assert vals["rejection"].fine == pytest.approx(425.0)
        assert vals["rejection"].points == 4

    def test_appellant_annulment(self, stakes):
        vals = compute_outcome_values(stakes, "appellant")
        # annulment: -1500 (only litigation cost)
        assert vals["annulment"].net_value == pytest.approx(-1500.0)
        assert vals["annulment"].fine == 0.0

    def test_appellant_reclassification(self, stakes):
        vals = compute_outcome_values(stakes, "appellant")
        # reclassification: -(107.5 + 1500) = -1607.5
        assert vals["reclassification"].net_value == pytest.approx(-1607.5)
        assert vals["reclassification"].fine == pytest.approx(107.5)
        assert vals["reclassification"].points == 0

    def test_respondent_rejection(self, stakes):
        vals = compute_outcome_values(stakes, "respondent")
        # respondent wins: +425 - 1500 = -1075
        assert vals["rejection"].net_value == pytest.approx(-1075.0)

    def test_respondent_annulment(self, stakes):
        vals = compute_outcome_values(stakes, "respondent")
        # respondent loses: 0 - 1500 = -1500
        assert vals["annulment"].net_value == pytest.approx(-1500.0)

    def test_respondent_reclassification(self, stakes):
        vals = compute_outcome_values(stakes, "respondent")
        # respondent partial: 107.5 - 1500 = -1392.5
        assert vals["reclassification"].net_value == pytest.approx(-1392.5)

    def test_status_quo_appellant(self, stakes):
        sq = compute_status_quo(stakes, "appellant")
        # -midpoint(170, 680) = -425
        assert sq == pytest.approx(-425.0)

    def test_status_quo_respondent(self, stakes):
        sq = compute_status_quo(stakes, "respondent")
        # +425
        assert sq == pytest.approx(425.0)

    def test_custom_litigation_cost(self, stakes):
        vals = compute_outcome_values(stakes, "appellant", litigation_cost=2000)
        assert vals["rejection"].net_value == pytest.approx(-(425 + 2000))
        assert vals["annulment"].net_value == pytest.approx(-2000)


# ---- Task 3: Equilibrium tests ----

class TestWeightedProbabilities:
    def test_single_entry(self, simple_prob_table):
        probs = compute_weighted_probabilities(simple_prob_table)
        assert probs["p_rejection"] == pytest.approx(0.5)
        assert probs["p_annulment"] == pytest.approx(0.3)
        assert probs["p_reclassification"] == pytest.approx(0.2)

    def test_multi_judge_uniform(self, multi_prob_table):
        probs = compute_weighted_probabilities(multi_prob_table, strategy="aggressivo")
        # formalista: 0.6, garantista: 0.2 → avg = 0.4
        assert probs["p_rejection"] == pytest.approx(0.4)
        # formalista: 0.2, garantista: 0.6 → avg = 0.4
        assert probs["p_annulment"] == pytest.approx(0.4)

    def test_custom_weights(self, multi_prob_table):
        weights = {"formalista": 0.8, "garantista": 0.2}
        probs = compute_weighted_probabilities(
            multi_prob_table, strategy="aggressivo", judge_weights=weights,
        )
        # 0.8*0.6 + 0.2*0.2 = 0.52
        assert probs["p_rejection"] == pytest.approx(0.52)

    def test_empty_table(self):
        probs = compute_weighted_probabilities({})
        assert probs["p_rejection"] == 0.0


class TestBATNA:
    def test_known_ev(self, simple_prob_table, stakes):
        """P={rej:0.5, ann:0.3, recl:0.2} → EV = 0.5×(-1925) + 0.3×(-1500) + 0.2×(-1607.5) = -1734.0"""
        batna = compute_batna(simple_prob_table, stakes, "appellant")
        expected_ev = 0.5 * (-1925) + 0.3 * (-1500) + 0.2 * (-1607.5)
        assert expected_ev == pytest.approx(-1734.0)
        assert batna.expected_value == pytest.approx(expected_ev)
        assert batna.party_id == "appellant"

    def test_ev_range_bounded(self, simple_prob_table, stakes):
        batna = compute_batna(simple_prob_table, stakes, "appellant")
        assert batna.expected_value_range[0] <= batna.expected_value
        assert batna.expected_value_range[1] >= batna.expected_value

    def test_best_strategy_single(self, simple_prob_table, stakes):
        batna = compute_batna(simple_prob_table, stakes, "appellant")
        assert batna.best_strategy == "aggressivo"

    def test_respondent_batna(self, simple_prob_table, stakes):
        batna = compute_batna(simple_prob_table, stakes, "respondent")
        # 0.5×(-1075) + 0.3×(-1500) + 0.2×(-1392.5) = -1265.0
        expected = 0.5 * (-1075) + 0.3 * (-1500) + 0.2 * (-1392.5)
        assert batna.expected_value == pytest.approx(expected)


class TestEVByStrategy:
    def test_multi_strategy(self, multi_prob_table, stakes):
        evs = compute_ev_by_strategy(multi_prob_table, stakes, "appellant")
        assert "aggressivo" in evs
        assert "tecnico" in evs
        # tecnico has more annulments on average → better for appellant
        assert evs["tecnico"] > evs["aggressivo"] or evs["aggressivo"] > evs["tecnico"]


class TestSettlement:
    def test_zopa_exists(self, simple_prob_table, stakes):
        batna_app = compute_batna(simple_prob_table, stakes, "appellant")
        batna_resp = compute_batna(simple_prob_table, stakes, "respondent")
        sq_app = compute_status_quo(stakes, "appellant")
        sq_resp = compute_status_quo(stakes, "respondent")

        settlement = compute_settlement_range(batna_app, batna_resp, sq_app, sq_resp)
        # Appellant reservation = max(BATNA=-1733, SQ=-425) = -425
        # Respondent max payment = -min(BATNA=-1265, SQ=425) = -(-1265) = 1265
        # Wait — respondent SQ = +425 (they collect fine without litigating)
        # min(-1265, 425) = -1265 → -min = 1265
        # ZOPA: [-425, 1265] — this exists
        assert settlement.settlement_exists is True
        assert settlement.zopa is not None
        assert settlement.nash_solution is not None
        # Actual respondent BATNA = 0.5×(-1075) + 0.3×(-1500) + 0.2×(-1392.5) = -1266.0
        # Respondent max payment = -min(-1266, 425) = 1266
        # Nash = midpoint(-425, 1266) = 420.5
        assert settlement.nash_solution == pytest.approx(420.5)

    def test_no_zopa_when_litigation_clearly_better(self):
        """Construct scenario where no ZOPA exists."""
        # Appellant BATNA very favorable (EV = +1000, which beats any settlement)
        batna_app = BATNA(
            party_id="appellant",
            expected_value=1000.0,
            expected_value_range=(800.0, 1200.0),
            best_strategy="aggressive",
            outcome_probabilities={"rejection": 0.0, "annulment": 1.0, "reclassification": 0.0},
        )
        # Respondent also has favorable BATNA
        batna_resp = BATNA(
            party_id="respondent",
            expected_value=500.0,
            expected_value_range=(300.0, 700.0),
            best_strategy="defensive",
            outcome_probabilities={"rejection": 1.0, "annulment": 0.0, "reclassification": 0.0},
        )
        settlement = compute_settlement_range(batna_app, batna_resp, 0, 0)
        # Appellant reservation = max(1000, 0) = 1000
        # Respondent max payment = -min(500, 0) = 0
        # 1000 > 0 → no ZOPA
        assert settlement.settlement_exists is False
        assert settlement.zopa is None


# ---- Task 4: Sensitivity tests ----

class TestFindThreshold:
    def test_basic_interpolation(self):
        sweep = [0.0, 1.0, 2.0, 3.0]
        evs = [100.0, 50.0, 0.0, -50.0]
        # Where does EV cross 25? Between 0 and 1: t = (25-100)/(50-100) = 1.5
        threshold = find_threshold(sweep, evs, 25.0)
        assert threshold == pytest.approx(1.5)

    def test_exact_zero_crossing(self):
        sweep = [0.0, 1.0, 2.0]
        evs = [10.0, 0.0, -10.0]
        threshold = find_threshold(sweep, evs, 0.0)
        # Crosses at 1.0 (first crossing found between index 0 and 1)
        assert threshold == pytest.approx(1.0)

    def test_no_crossing(self):
        sweep = [0.0, 1.0, 2.0]
        evs = [10.0, 20.0, 30.0]
        threshold = find_threshold(sweep, evs, -5.0)
        assert threshold is None


class TestSensitivityLitigationCost:
    def test_ev_decreases_with_cost(self, simple_prob_table, stakes):
        result = sensitivity_litigation_cost(simple_prob_table, stakes)
        # EV should monotonically decrease as litigation cost increases
        for i in range(len(result.ev_at_each) - 1):
            assert result.ev_at_each[i] >= result.ev_at_each[i + 1]
        assert result.impact > 0

    def test_threshold_exists(self, simple_prob_table, stakes):
        result = sensitivity_litigation_cost(simple_prob_table, stakes)
        # With default range (0, 5000), there should be a threshold
        # where litigation EV crosses status quo
        # (at low cost, litigation may be better than status quo)
        assert result.parameter == "litigation_cost"


class TestSensitivityFineAmount:
    def test_impact_positive(self, simple_prob_table, stakes):
        result = sensitivity_fine_amount(simple_prob_table, stakes)
        assert result.impact > 0
        assert result.parameter == "current_fine"


class TestSensitivityRejectionProb:
    def test_ev_decreases_with_rejection(self, simple_prob_table, stakes):
        result = sensitivity_rejection_probability(
            simple_prob_table, stakes, "aggressivo",
        )
        # Higher rejection probability → worse for appellant
        for i in range(len(result.ev_at_each) - 1):
            assert result.ev_at_each[i] >= result.ev_at_each[i + 1]


class TestTornadoRanking:
    def test_sorted_by_impact(self, multi_prob_table, stakes):
        results = run_all_sensitivity(multi_prob_table, stakes)
        assert len(results) >= 2
        # Verify descending impact order
        for i in range(len(results) - 1):
            assert results[i].impact >= results[i + 1].impact


# ---- Task 5: Integration ----

class TestAnalyze:
    def test_full_analysis(self, multi_prob_table, stakes):
        aggregated = {
            "probability_table": multi_prob_table,
            "total_runs": 20,
            "failed_runs": 0,
            "dominated_strategies": [],
            "argument_effectiveness": {},
            "precedent_analysis": {},
        }
        case_data = {"stakes": stakes}

        result = analyze(aggregated, case_data)

        assert isinstance(result, GameTheoryAnalysis)
        assert "appellant" in result.party_valuations
        assert "respondent" in result.party_valuations
        assert "appellant" in result.batna
        assert "respondent" in result.batna
        assert isinstance(result.settlement, SettlementRange)
        assert len(result.sensitivity) > 0
        assert "aggressivo" in result.expected_value_by_strategy
        assert "tecnico" in result.expected_value_by_strategy
        assert result.recommended_strategy in ("aggressivo", "tecnico")
        assert result.analysis_metadata["total_runs"] == 20

    def test_dominated_strategy_excluded(self, multi_prob_table, stakes):
        aggregated = {
            "probability_table": multi_prob_table,
            "total_runs": 20,
            "failed_runs": 0,
            "dominated_strategies": ["aggressivo"],
            "argument_effectiveness": {},
            "precedent_analysis": {},
        }
        case_data = {"stakes": stakes}
        result = analyze(aggregated, case_data)
        # aggressivo is dominated, so tecnico should be recommended
        assert result.recommended_strategy == "tecnico"

    def test_json_roundtrip(self, multi_prob_table, stakes):
        aggregated = {
            "probability_table": multi_prob_table,
            "total_runs": 20,
            "failed_runs": 0,
            "dominated_strategies": [],
            "argument_effectiveness": {},
            "precedent_analysis": {},
        }
        case_data = {"stakes": stakes}
        result = analyze(aggregated, case_data)
        # Verify JSON serializable
        data = result.model_dump()
        reconstructed = GameTheoryAnalysis(**data)
        assert reconstructed.recommended_strategy == result.recommended_strategy


# ---- Task 6: Output formatting ----

class TestGameTheorySummary:
    def test_markdown_output(self, multi_prob_table, stakes):
        aggregated = {
            "probability_table": multi_prob_table,
            "total_runs": 20,
            "failed_runs": 0,
            "dominated_strategies": [],
            "argument_effectiveness": {},
            "precedent_analysis": {},
        }
        case_data = {"stakes": stakes}
        result = analyze(aggregated, case_data)
        md = format_game_theory_summary(result)

        assert "# Analisi di Teoria dei Giochi" in md
        assert "## Expected Value per Strategia" in md
        assert "## BATNA" in md
        assert "## Analisi Transattiva" in md
        assert "## Analisi di Sensibilità" in md
        assert "aggressivo" in md or "tecnico" in md
