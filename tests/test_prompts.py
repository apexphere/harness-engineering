from src.prompts import build_research_prompt, build_summary_prompt, Prompt


def test_prompt_renders_template():
    prompt = Prompt(system="sys", user_template="Hello {name}, ask about {topic}")
    rendered = prompt.render_user(name="Alice", topic="physics")
    assert rendered == "Hello Alice, ask about physics"


def test_research_prompt_includes_context():
    prompt = build_research_prompt("test?", context="prior info here")
    assert "prior info here" in prompt.system


def test_research_prompt_without_context():
    prompt = build_research_prompt("test?")
    assert "Prior context" not in prompt.system


def test_summary_prompt_template():
    prompt = build_summary_prompt()
    rendered = prompt.render_user(text="some text")
    assert rendered == "Summarize: some text"


def test_prompt_is_immutable():
    prompt = Prompt(system="sys", user_template="{q}")
    # frozen dataclass — should not be assignable
    try:
        prompt.system = "new"  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass
