import json
import re
import time
from enum import Enum
from pathlib import Path

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorFinalDecisionEvaluator,
    FactorValueEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace
from rdagent.log import rdagent_logger as logger

FactorSingleFeedback = CoSTEERSingleFeedbackDeprecated


class FactorGateDecision(str, Enum):
    """Research permission granted by the implementation gates."""

    REJECT = "REJECT"
    SMOKE_ONLY = "SMOKE_ONLY"
    FULL_READY = "FULL_READY"


def get_factor_gate_decision(feedback: FactorSingleFeedback | None) -> FactorGateDecision:
    """Translate merged functional/performance feedback into a research permission."""
    if feedback is None:
        return FactorGateDecision.REJECT
    sources = feedback.source_feedback
    if sources.get("correctness") is not True or sources.get("schema") is not True:
        return FactorGateDecision.REJECT
    if sources.get("performance") is True:
        return FactorGateDecision.FULL_READY
    performance_feedback = "\n".join(
        str(value or "")
        for value in (
            feedback.execution,
            feedback.return_checking,
            feedback.code,
            getattr(feedback, "final_feedback", None),
        )
    )
    if sources.get("performance") is False and "failure_type=PROJECTED_RUNTIME_EXCEEDED" in performance_feedback:
        return FactorGateDecision.SMOKE_ONLY
    return FactorGateDecision.REJECT


class FactorEvaluatorForCoder(CoSTEEREvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        if queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            feedback = FactorSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
            feedback.source_feedback.update({"correctness": False, "schema": False})
            return feedback
        factor_feedback = FactorSingleFeedback()

        # 1. Get factor execution feedback to generated implementation and remove the long list of numbers in execution feedback
        (
            execution_feedback,
            gen_df,
        ) = implementation.execute()

        execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
        factor_feedback.execution_feedback = "\n".join(
            [line for line in execution_feedback.split("\n") if "warning" not in line.lower()],
        )

        # 2. Get factor value feedback
        if gen_df is None:
            factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
            factor_feedback.value_generated_flag = False
            decision_from_value_check = None
        else:
            factor_feedback.value_generated_flag = True
            (
                factor_feedback.value_feedback,
                decision_from_value_check,
            ) = self.value_evaluator.evaluate(
                implementation=implementation, gt_implementation=gt_implementation, version=target_task.version,
            )

        factor_feedback.final_decision_based_on_gt = gt_implementation is not None

        if decision_from_value_check is not None and decision_from_value_check is True:
            # To avoid confusion, when same_value_or_high_correlation is True, we do not need code feedback
            factor_feedback.code_feedback = "Final decision is True and there are no code critics."
            factor_feedback.final_decision = decision_from_value_check
            factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
        elif decision_from_value_check is not None and decision_from_value_check is False:
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=gt_implementation,
            )
            factor_feedback.final_decision = decision_from_value_check
            factor_feedback.final_feedback = "Value evaluation failed, skip final decision evaluation."
        else:
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=gt_implementation,
            )
            (
                factor_feedback.final_decision,
                factor_feedback.final_feedback,
            ) = self.final_decision_evaluator.evaluate(
                target_task=target_task,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                code_feedback=factor_feedback.code_feedback,
            )
        factor_feedback.source_feedback.update(
            {
                "correctness": factor_feedback.final_decision is True,
                "schema": gen_df is not None and decision_from_value_check is not False,
            },
        )
        return factor_feedback


class FactorPerformanceEvaluator(CoSTEEREvaluator):
    """Deterministic wall-clock performance gate for factor implementations."""

    source_tag = "performance"

    @staticmethod
    def _feedback(message: str, decision: bool, value_generated: bool = False) -> FactorSingleFeedback:
        return FactorSingleFeedback(
            execution_feedback=message,
            value_generated_flag=value_generated,
            code_feedback=message,
            value_feedback=message,
            final_decision=decision,
            final_feedback=message,
            final_decision_based_on_gt=False,
            source_feedback={FactorPerformanceEvaluator.source_tag: decision},
        )

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return self._feedback(
                "Performance evaluation failed. failure_type=PERFORMANCE_NO_IMPLEMENTATION stage=PROFILE",
                False,
            )

        if not FACTOR_COSTEER_SETTINGS.performance_gate_enabled:
            return self._feedback("Performance evaluation PASS (performance gate disabled).", True)

        profile_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_profile)
        metadata_path = profile_folder / "profile_meta.json"
        if not profile_folder.is_dir() or not metadata_path.is_file():
            return self._feedback(
                "Performance evaluation failed. failure_type=PROFILE_DATA_MISSING stage=PROFILE. "
                "Profile dataset is missing; run the profile data builder.",
                False,
            )

        started_at = time.perf_counter()
        try:
            execution_feedback, generated = implementation.execute_profile()
        except Exception as exc:
            runtime = time.perf_counter() - started_at
            message = (
                "Performance evaluation failed. failure_type=PERFORMANCE_EXECUTION_ERROR stage=PROFILE "
                f"runtime={runtime:.2f}s error={exc}"
            )
            return self._feedback(message, False)
        runtime = time.perf_counter() - started_at

        if (
            runtime > FACTOR_COSTEER_SETTINGS.profile_execution_timeout
            or "failure_type=PERFORMANCE_TIMEOUT" in execution_feedback
        ):
            message = (
                "Performance evaluation failed.\n\n"
                "failure_type=PERFORMANCE_TIMEOUT\n"
                "stage=PROFILE\n"
                f"runtime={runtime:.2f}s\n"
                f"timeout={FACTOR_COSTEER_SETTINGS.profile_execution_timeout}s\n\n"
                f"Execution feedback: {execution_feedback}"
            )
            return self._feedback(message, False)
        if generated is None:
            message = (
                "Performance evaluation failed. failure_type=PERFORMANCE_EXECUTION_ERROR stage=PROFILE\n"
                f"Measured profile runtime: {runtime:.2f} seconds\n"
                f"Execution feedback: {execution_feedback}"
            )
            return self._feedback(message, False)

        try:
            metadata = json.loads(metadata_path.read_text())
            profile_rows = int(metadata["profile_rows"])
            full_rows = int(metadata["full_rows"])
            if profile_rows <= 0 or full_rows <= 0:
                raise ValueError("row counts must be positive")
            scale_ratio = full_rows / profile_rows
            instrument_count = int(metadata["instrument_count"])
            date_count = int(metadata["date_count"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return self._feedback(
                "Performance evaluation failed. failure_type=PROFILE_METADATA_INVALID stage=PROFILE "
                f"error={exc}",
                False,
            )

        projected_runtime = runtime * scale_ratio
        budget = FACTOR_COSTEER_SETTINGS.projected_full_runtime_budget
        decision = projected_runtime <= budget
        factor_name = getattr(target_task, "factor_name", target_task.name)
        logger.info(
            f"[FactorRuntime] stage=PROFILE factor={factor_name} projected_full={projected_runtime:.2f}s "
            f"{'PASS' if decision else 'FAIL'}",
        )
        outcome = "PASS" if decision else "failed"
        guidance = ""
        if not decision:
            guidance = (
                "\n\nThe implementation appears functionally valid but is too expensive for full-sample execution. "
                "Preserve the factor definition and output schema, but optimize the implementation. "
                "Reduce Python-level repeated work, repeated object construction, unnecessary I/O, "
                "or avoidable per-window overhead where possible. Do not change the mathematical "
                "definition merely to pass the performance test."
            )
        failure_type = "" if decision else "failure_type=PROJECTED_RUNTIME_EXCEEDED\n"
        message = (
            f"Performance evaluation {outcome}.\n\n"
            f"{failure_type}"
            "stage=PROFILE\n"
            "estimated=true\n"
            f"profile_runtime_seconds={runtime:.2f}\n"
            f"profile_rows={profile_rows}\n"
            f"full_rows={full_rows}\n"
            f"scale_ratio={scale_ratio:.6f}\n"
            f"instrument_count={instrument_count}\n"
            f"date_count={date_count}\n"
            f"projected_full_runtime_seconds={projected_runtime:.2f}\n"
            f"projected_full_runtime_budget_seconds={budget:.2f}\n\n"
            "The full-runtime projection is a linear heuristic."
            f"{guidance}"
        )
        return self._feedback(message, decision, value_generated=True)


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
