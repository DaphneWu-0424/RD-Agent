import json
from types import SimpleNamespace

import pytest

import rdagent.oai.backend.litellm as litellm_backend
import rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader as pdf_loader


@pytest.mark.offline
def test_factor_only_prompt_and_repeated_extraction_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_report = """
    We define simple_return_factor = close_t / close_t_minus_5 - 1 as a baseline stock-selection factor.
    The prediction probability emitted by the trained classifier is retained as model_probability_factor.
    A transformer encoder is used to train the classifier, but the encoder itself is only a model architecture.
    """
    expected = {
        "simple_return_factor": "Five-day close return used as a baseline stock-selection factor.",
        "model_probability_factor": "Classifier probability used directly as a stock-selection factor.",
    }
    captured_system_prompts = []
    captured_call_kwargs = []

    class FakeSession:
        def __init__(self, system_prompt: str) -> None:
            captured_system_prompts.append(system_prompt)
            self.responses = iter([json.dumps({"factors": expected}), json.dumps({"factors": {}})])

        def build_chat_completion(self, *, user_prompt: str, **kwargs) -> str:
            captured_call_kwargs.append(kwargs)
            return next(self.responses)

    class FakeBackend:
        def build_chat_session(self, *, session_system_prompt: str) -> FakeSession:
            return FakeSession(session_system_prompt)

    monkeypatch.setattr(pdf_loader, "APIBackend", lambda: FakeBackend())
    extraction = getattr(pdf_loader, "__extract_factors_name_and_desc_from_content")

    results = [extraction(synthetic_report) for _ in range(3)]

    assert results == [expected, expected, expected]
    assert all("transformer_encoder" not in result for result in results)
    assert len(captured_call_kwargs) == 6
    assert all(kwargs["temperature"] == 0.0 for kwargs in captured_call_kwargs)
    assert all(kwargs["json_mode"] is True for kwargs in captured_call_kwargs)
    prompt = captured_system_prompts[0]
    assert "不要只抽取报告最终采用的主因子" in prompt
    assert "简单手工构造因子" in prompt
    assert "模型输出形成的因子" in prompt
    assert "纯算法、模型架构" in prompt
    assert "保留被计算的核心统计量" in prompt
    assert "difference、ratio、change、rank" in prompt
    assert '"summary":' not in prompt
    assert '"models": {' not in prompt


@pytest.mark.offline
def test_litellm_per_call_temperature_overrides_global_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"factors": {}}'), finish_reason="stop")],
        )

    monkeypatch.setattr(litellm_backend, "completion", fake_completion)
    monkeypatch.setattr(litellm_backend, "completion_cost", lambda **kwargs: 0.0)
    monkeypatch.setattr(litellm_backend, "token_counter", lambda **kwargs: 0)
    monkeypatch.setattr(litellm_backend.LITELLM_SETTINGS, "chat_stream", False)
    monkeypatch.setattr(litellm_backend.LITELLM_SETTINGS, "log_llm_chat_content", False)
    monkeypatch.setattr(litellm_backend.LITELLM_SETTINGS, "chat_temperature", 0.73)
    litellm_backend.LiteLLMAPIBackend._has_logged_settings = True
    backend = litellm_backend.LiteLLMAPIBackend()

    response, finish_reason = backend._create_chat_completion_inner_function(
        messages=[{"role": "user", "content": "extract factors"}],
        temperature=0.0,
    )

    assert response == '{"factors": {}}'
    assert finish_reason == "stop"
    assert captured["temperature"] == 0.0
