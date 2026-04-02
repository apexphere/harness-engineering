from src.tools import execute_tool, execute_search, execute_calculate


def test_search_finds_aurora():
    result = execute_search("what causes aurora borealis")
    assert "solar wind" in result.output.lower()
    assert not result.is_error


def test_search_returns_no_results():
    result = execute_search("quantum entanglement spoons")
    assert "No results" in result.output


def test_calculate_basic_math():
    result = execute_calculate("2 + 2")
    assert result.output == "4"
    assert not result.is_error


def test_calculate_rejects_dangerous_input():
    result = execute_calculate("__import__('os').system('ls')")
    assert result.is_error


def test_calculate_division():
    result = execute_calculate("144 / 12")
    assert result.output == "12.0"


def test_execute_tool_routes_correctly():
    result = execute_tool("search", {"query": "gravity"})
    assert "spacetime" in result.output.lower()


def test_execute_tool_unknown():
    result = execute_tool("fly", {"destination": "moon"})
    assert result.is_error
    assert "Unknown tool" in result.output
