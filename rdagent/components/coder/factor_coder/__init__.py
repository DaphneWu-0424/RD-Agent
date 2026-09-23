from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import (
    FactorEvaluatorForCoder,
    FactorGateDecision,
    FactorPerformanceEvaluator,
    get_factor_gate_decision,
)
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(
            [FactorEvaluatorForCoder(scen=scen), FactorPerformanceEvaluator(scen=scen)],
            scen=scen,
            evaluate_all_at_once=True,
        )
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    @staticmethod
    def _gate_decisions(feedback):
        return [get_factor_gate_decision(single_feedback) for single_feedback in feedback]

    def should_use_new_evo(self, base_fb, new_fb) -> bool:
        """Prefer full-ready solutions and retain the latest smoke-capable fallback."""
        new_decisions = self._gate_decisions(new_fb)
        new_rank = (
            sum(decision is not FactorGateDecision.REJECT for decision in new_decisions),
            sum(decision is FactorGateDecision.FULL_READY for decision in new_decisions),
        )
        if new_rank[0] == 0:
            return False
        if base_fb is None:
            return True
        base_decisions = self._gate_decisions(base_fb)
        base_rank = (
            sum(decision is not FactorGateDecision.REJECT for decision in base_decisions),
            sum(decision is FactorGateDecision.FULL_READY for decision in base_decisions),
        )
        return new_rank >= base_rank

    def _exp_postprocess_by_feedback(self, evo: Experiment, feedback):
        decisions = self._gate_decisions(feedback)
        if all(decision is FactorGateDecision.REJECT for decision in decisions):
            return super()._exp_postprocess_by_feedback(evo, feedback)
        for task, decision in zip(evo.sub_tasks, decisions):
            task.factor_implementation = decision is not FactorGateDecision.REJECT
        return evo

    def _apply_gate_state(self, exp: Experiment, feedback):
        """Propagate the selected fallback's gate state to the caller's experiment."""
        decisions = self._gate_decisions(feedback)
        exp.factor_gate_decisions = [decision.value for decision in decisions]
        if any(decision is FactorGateDecision.SMOKE_ONLY for decision in decisions):
            exp.research_mode = "smoke"
        elif any(decision is FactorGateDecision.FULL_READY for decision in decisions):
            exp.research_mode = "full"
        else:
            exp.research_mode = "rejected"

        for task, workspace, single_feedback, decision in zip(
            exp.sub_tasks,
            exp.sub_workspace_list,
            feedback,
            decisions,
        ):
            workspace_preserved = workspace is not None
            task.factor_implementation = decision is not FactorGateDecision.REJECT and workspace_preserved
            sources = single_feedback.source_feedback if single_feedback is not None else {}
            logger.info(
                f"[FactorGate] factor={task.factor_name} "
                f"functional_pass={sources.get('correctness') is True and sources.get('schema') is True} "
                f"performance_pass={sources.get('performance') is True} "
                f"decision={decision.value} workspace_preserved={workspace_preserved}",
            )
        return decisions

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                exp.prop_dev_feedback = (
                    self.selected_feedback
                    if self.selected_feedback is not None
                    else self.evolve_agent.evolving_trace[-1].feedback
                )
                self._apply_gate_state(exp, exp.prop_dev_feedback)
        return exp
