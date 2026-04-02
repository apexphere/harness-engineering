import pytest

from src.response import Response
from src.stages import (
    register_layer,
    clear_layers,
    get_layer_names,
    run_pipeline,
    register_default_layers,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear layer registry before each test."""
    clear_layers()
    yield
    clear_layers()


def _make_layer(suffix: str):
    """Create a test layer that appends to response text."""
    def layer(query: str, response: Response, config: dict) -> Response:
        return response.with_text(response.text + f" [{suffix}]")
    return layer


def test_register_and_list_layers():
    register_layer("a", _make_layer("a"))
    register_layer("b", _make_layer("b"))
    assert get_layer_names() == ["a", "b"]


def test_run_pipeline_stage_1():
    register_layer("first", _make_layer("first"))
    register_layer("second", _make_layer("second"))
    result = run_pipeline("test", stage=1)
    assert "[first]" in result.text
    assert "[second]" not in result.text


def test_run_pipeline_stage_2():
    register_layer("first", _make_layer("first"))
    register_layer("second", _make_layer("second"))
    result = run_pipeline("test", stage=2)
    assert "[first]" in result.text
    assert "[second]" in result.text


def test_run_pipeline_invalid_stage_low():
    register_layer("a", _make_layer("a"))
    with pytest.raises(ValueError):
        run_pipeline("test", stage=0)


def test_run_pipeline_invalid_stage_high():
    register_layer("a", _make_layer("a"))
    with pytest.raises(ValueError):
        run_pipeline("test", stage=2)


def test_pipeline_passes_config():
    """Layers receive the config dict."""
    def config_layer(query: str, response: Response, config: dict) -> Response:
        return response.with_text(config.get("test_value", "missing"))

    register_layer("config_test", config_layer)
    result = run_pipeline("test", stage=1, config={"test_value": "found"})
    assert result.text == "found"


def test_pipeline_immutability():
    """Each layer gets a fresh Response, not a mutated one."""
    register_layer("a", _make_layer("a"))
    register_layer("b", _make_layer("b"))

    result_1 = run_pipeline("test", stage=1)
    result_2 = run_pipeline("test", stage=2)

    # Stage 1 result should not be affected by stage 2 run
    assert "[b]" not in result_1.text
    assert "[b]" in result_2.text


def test_register_default_layers():
    register_default_layers()
    names = get_layer_names()
    assert names == ["prompts", "tools", "parsing", "guardrails", "evals", "orchestration"]


def test_default_stubs_run_without_error():
    register_default_layers()
    result = run_pipeline("hello", stage=6)
    assert isinstance(result, Response)
    assert "hello" in result.text  # prompt stub includes the query


def test_verbose_mode(capsys):
    register_layer("test_layer", _make_layer("verbose"))
    run_pipeline("q", stage=1, config={"verbose": True})
    captured = capsys.readouterr()
    assert "test_layer" in captured.out
