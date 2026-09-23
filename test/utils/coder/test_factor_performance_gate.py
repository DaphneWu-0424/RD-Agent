import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from rdagent.components.coder.CoSTEER import CoSTEER
import rdagent.components.coder.CoSTEER.evaluators as costeer_evaluators
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.factor_coder import FactorCoSTEER
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import (
    FactorGateDecision,
    FactorPerformanceEvaluator,
    get_factor_gate_decision,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
from rdagent.components.coder.factor_coder.profile_data import build_profile_dataset
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.developer.utils import _build_execute_calls


@pytest.fixture
def profile_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    metadata = {
        "profile_rows": 100,
        "full_rows": 1000,
        "scale_ratio": 10,
        "instrument_count": 10,
        "date_count": 10,
    }
    (tmp_path / "profile_meta.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_profile", str(tmp_path))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "performance_gate_enabled", True)
    return tmp_path


def test_data_type_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", "debug-data")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_profile", "profile-data")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", "all-data")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "file_based_execution_timeout", 11)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "profile_execution_timeout", 22)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "full_execution_timeout", 33)

    assert FactorFBWorkspace._data_folder_and_timeout("Debug") == (Path("debug-data"), 11)
    assert FactorFBWorkspace._data_folder_and_timeout("Profile") == (Path("profile-data"), 22)
    assert FactorFBWorkspace._data_folder_and_timeout("All") == (Path("all-data"), 33)
    with pytest.raises(ValueError, match="Unknown factor execution data_type"):
        FactorFBWorkspace._data_folder_and_timeout("unknown")


def test_fast_implementation_passes(
    profile_folder: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "projected_full_runtime_budget", 1000.0)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "profile_execution_timeout", 1)
    implementation = SimpleNamespace(execute_profile=lambda: ("ok", object()))
    task = SimpleNamespace(factor_name="example", name="example")

    feedback = FactorPerformanceEvaluator(scen=object()).evaluate(task, implementation)

    assert feedback.final_decision is True
    assert "projected_full_runtime_seconds=" in feedback.code_feedback
    assert feedback.source_feedback == {"performance": True}


def test_slow_implementation_times_out(
    profile_folder: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "profile_execution_timeout", 0.01)

    def slow_execute():
        time.sleep(0.02)
        return "ok", object()

    implementation = SimpleNamespace(execute_profile=slow_execute)
    task = SimpleNamespace(factor_name="slow", name="slow")
    feedback = FactorPerformanceEvaluator(scen=object()).evaluate(task, implementation)

    assert feedback.final_decision is False
    assert "failure_type=PERFORMANCE_TIMEOUT" in feedback.execution_feedback
    assert "stage=PROFILE" in feedback.execution_feedback


def test_correctness_and_performance_feedback_merge_to_failure() -> None:
    correctness = CoSTEERSingleFeedback(
        execution="correctness passed",
        return_checking="schema passed",
        code="no correctness issue",
        final_decision=True,
        source_feedback={"correctness": True},
    )
    performance = CoSTEERSingleFeedback(
        execution="performance failed",
        return_checking=None,
        code="optimize repeated work",
        final_decision=False,
        source_feedback={"performance": False},
    )

    merged = CoSTEERSingleFeedback.merge([correctness, performance])

    assert merged.final_decision is False
    assert "performance failed" in merged.execution
    assert "optimize repeated work" in merged.code
    assert merged.source_feedback["performance"] is False
    # The merged feedback is what the existing evolving trace/RAG path supplies to the next coder round.
    assert "performance failed" in str(merged)


def _gate_feedback(performance: bool, message: str = "performance passed") -> CoSTEERSingleFeedback:
    return CoSTEERSingleFeedback(
        execution=message,
        return_checking="schema passed",
        code=message,
        final_decision=performance,
        source_feedback={"correctness": True, "schema": True, "performance": performance},
    )


def test_projected_runtime_failure_grants_smoke_only() -> None:
    feedback = _gate_feedback(
        False,
        "Performance evaluation failed. failure_type=PROJECTED_RUNTIME_EXCEEDED stage=PROJECTION",
    )

    assert get_factor_gate_decision(feedback) is FactorGateDecision.SMOKE_ONLY


def test_functional_or_non_scalability_failure_is_rejected() -> None:
    functional_failure = _gate_feedback(False, "failure_type=PROJECTED_RUNTIME_EXCEEDED")
    functional_failure.source_feedback["correctness"] = False
    execution_failure = _gate_feedback(False, "failure_type=PERFORMANCE_EXECUTION_ERROR")

    assert get_factor_gate_decision(functional_failure) is FactorGateDecision.REJECT
    assert get_factor_gate_decision(execution_failure) is FactorGateDecision.REJECT
    assert get_factor_gate_decision(_gate_feedback(True)) is FactorGateDecision.FULL_READY


def test_smoke_only_factor_executes_on_profile_but_not_full_sample() -> None:
    class Workspace:
        def execute(self, data_type):
            return data_type

    workspace = Workspace()
    feedback = _gate_feedback(
        False,
        "Performance evaluation failed. failure_type=PROJECTED_RUNTIME_EXCEEDED stage=PROJECTION",
    )
    exp = SimpleNamespace(
        sub_tasks=[object()],
        sub_workspace_list=[workspace],
        prop_dev_feedback=CoSTEERMultiFeedback([feedback]),
    )

    smoke_calls = _build_execute_calls(exp, [], "Profile")
    full_calls = _build_execute_calls(exp, [], "All")

    assert len(smoke_calls) == 1
    assert smoke_calls[0][1] == ("Profile",)
    assert full_calls == []


def test_smoke_only_gate_state_marks_original_task_implemented() -> None:
    task = SimpleNamespace(factor_name="smoke_factor", factor_implementation=False)
    workspace = object()
    feedback = CoSTEERMultiFeedback(
        [
            _gate_feedback(
                False,
                "Performance evaluation failed. failure_type=PROJECTED_RUNTIME_EXCEEDED stage=PROJECTION",
            ),
        ],
    )
    exp = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[workspace])
    coder = object.__new__(FactorCoSTEER)

    decisions = coder._apply_gate_state(exp, feedback)

    assert decisions == [FactorGateDecision.SMOKE_ONLY]
    assert exp.research_mode == "smoke"
    assert exp.factor_gate_decisions == ["SMOKE_ONLY"]
    assert exp.sub_workspace_list == [workspace]
    assert task.factor_implementation is True


def test_develop_preserves_falsey_selected_smoke_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    task = SimpleNamespace(factor_name="smoke_factor", factor_implementation=False)
    workspace = object()
    selected_feedback = CoSTEERMultiFeedback(
        [
            _gate_feedback(
                False,
                "Performance evaluation failed. failure_type=PROJECTED_RUNTIME_EXCEEDED stage=PROJECTION",
            ),
        ],
    )
    latest_feedback = CoSTEERMultiFeedback(
        [_gate_feedback(False, "failure_type=PERFORMANCE_EXECUTION_ERROR")],
    )
    exp = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[workspace])
    coder = object.__new__(FactorCoSTEER)
    coder.selected_feedback = selected_feedback
    coder.evolve_agent = SimpleNamespace(evolving_trace=[SimpleNamespace(feedback=latest_feedback)])
    monkeypatch.setattr(CoSTEER, "develop", lambda self, value: value)

    result = coder.develop(exp)

    assert bool(selected_feedback) is False
    assert result.prop_dev_feedback is selected_feedback
    assert result.research_mode == "smoke"
    assert task.factor_implementation is True


def test_rejected_gate_state_does_not_mark_task_implemented() -> None:
    task = SimpleNamespace(factor_name="broken_factor", factor_implementation=True)
    feedback = _gate_feedback(False, "failure_type=PERFORMANCE_EXECUTION_ERROR")
    exp = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[object()])
    coder = object.__new__(FactorCoSTEER)

    decisions = coder._apply_gate_state(exp, CoSTEERMultiFeedback([feedback]))

    assert decisions == [FactorGateDecision.REJECT]
    assert exp.research_mode == "rejected"
    assert task.factor_implementation is False


def test_smoke_date_env_uses_profile_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=20, name="datetime")
    instruments = pd.Index(["asset_a", "asset_b"], name="instrument")
    index = pd.MultiIndex.from_product([dates, instruments])
    pd.DataFrame({"close": 1.0}, index=index).to_hdf(tmp_path / "daily_pv.h5", key="data")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_profile", str(tmp_path))

    date_env = QlibFactorRunner._smoke_date_env()

    assert date_env == {
        "train_start": "2020-01-01",
        "train_end": "2020-01-16",
        "valid_start": "2020-01-17",
        "valid_end": "2020-01-22",
        "test_start": "2020-01-23",
        "test_end": "2020-01-28",
    }


def test_multi_evaluator_runs_full_chain_in_one_evolve_step(monkeypatch: pytest.MonkeyPatch) -> None:
    class StaticEvaluator:
        def __init__(self, decision: bool, message: str) -> None:
            self.decision = decision
            self.message = message

        def evaluate(self, *args):
            return CoSTEERSingleFeedback(
                execution=self.message,
                return_checking=None,
                code=self.message,
                final_decision=self.decision,
            )

    def run_inline(calls, n):
        return [function(*args) for function, args in calls]

    monkeypatch.setattr(costeer_evaluators, "multiprocessing_wrapper", run_inline)
    task = SimpleNamespace(factor_implementation=False)
    evo = SimpleNamespace(
        sub_tasks=[task],
        sub_workspace_list=[object()],
        sub_gt_implementations=None,
    )
    evaluator = CoSTEERMultiEvaluator(
        [StaticEvaluator(True, "correctness passed"), StaticEvaluator(False, "performance failed")],
        scen=object(),
        evaluate_all_at_once=True,
    )
    iterator = evaluator.evaluate_iter()
    next(iterator)

    partial = iterator.send(evo)
    assert partial[0].final_decision is False
    assert "performance failed" in partial[0].execution
    with pytest.raises(StopIteration) as stopped:
        iterator.send(evo)
    assert stopped.value.value[0].final_decision is False
    assert task.factor_implementation is False


def test_profile_builder_preserves_continuous_window_and_schema(tmp_path: Path) -> None:
    source = tmp_path / "full"
    output = tmp_path / "profile"
    source.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=300, name="datetime")
    instruments = pd.Index([f"asset_{index:03d}" for index in range(60)], name="instrument")
    index = pd.MultiIndex.from_product([dates, instruments])
    frame = pd.DataFrame({"close": range(len(index)), "volume": 1.0}, index=index)
    frame.to_hdf(source / "daily_pv.h5", key="data")
    (source / "README.md").write_text("auxiliary")

    metadata = build_profile_dataset(source, output, instrument_count=40, date_count=256)
    profiled = pd.read_hdf(output / "daily_pv.h5", key="data")

    assert metadata["full_rows"] == len(frame)
    assert metadata["profile_rows"] == len(profiled) == 40 * 256
    assert profiled.index.names == ["datetime", "instrument"]
    assert profiled.columns.equals(frame.columns)
    assert list(profiled.index.get_level_values("datetime").unique()) == list(dates[-256:])
    assert profiled.index.get_level_values("instrument").nunique() == 40
    assert (output / "README.md").read_text() == "auxiliary"
    assert json.loads((output / "profile_meta.json").read_text()) == metadata

    repeated = build_profile_dataset(source, output, instrument_count=40, date_count=256)
    assert repeated == metadata
    pd.testing.assert_frame_equal(profiled, pd.read_hdf(output / "daily_pv.h5", key="data"))
