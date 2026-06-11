import pytest
from refactored_plugins.skill_index import SkillAwareContextPlugin


@pytest.fixture
def mock_tool_registry():
    return {
        "math": {
            "type": "function",
            "function": {"name": "math_tool", "description": "Performs mathematical calculations"},
        },
        "weather": {
            "type": "function",
            "function": {"name": "weather_tool", "description": "Gets the current weather"},
        },
        "database": {"type": "function", "function": {"name": "db_tool", "description": "Queries the database"}},
    }


@pytest.mark.asyncio
async def test_skill_aware_context_plugin(mock_tool_registry):
    # Initialize the plugin
    plugin = SkillAwareContextPlugin(tool_registry=mock_tool_registry)

    # Pass a mock OpenAI request with "_required_skills": ["math"]
    mock_request = {"messages": [{"role": "user", "content": "What is 2 + 2?"}], "_required_skills": ["math"]}

    # Process the request
    modified_request = await plugin.process(mock_request)

    # Asserts that the returned request contains exactly 1 tool in its "tools" array
    assert "tools" in modified_request
    assert len(modified_request["tools"]) == 1

    # Asserts that it is the correct "math" schema
    assert modified_request["tools"][0]["function"]["name"] == "math_tool"

    # Asserts that the telemetry correctly tracks that 2 tools were filtered out.
    metrics = plugin.get_plugin_metrics()
    assert metrics["total_tools_filtered"] == 2.0
