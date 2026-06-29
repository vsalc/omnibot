"""Tests for the AIGenerator tool-calling flow (backend/ai_generator.py)."""

from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator


@pytest.fixture
def patched_client():
    """Patch the Anthropic SDK so no client is constructed / no network used."""
    with patch("ai_generator.anthropic.Anthropic") as anthropic_cls:
        client = MagicMock()
        anthropic_cls.return_value = client
        yield client


def test_direct_response_without_tools(patched_client, make_anthropic_response):
    patched_client.messages.create.return_value = make_anthropic_response(text="42")
    gen = AIGenerator(api_key="k", model="m")

    out = gen.generate_response(query="2+2 doubled?")

    assert out == "42"
    assert patched_client.messages.create.call_count == 1


def test_first_call_advertises_tools(patched_client, make_anthropic_response):
    patched_client.messages.create.return_value = make_anthropic_response(text="ok")
    gen = AIGenerator(api_key="k", model="m")
    tools = [{"name": "search_course_content"}]

    gen.generate_response(query="hi", tools=tools, tool_manager=MagicMock())

    params = patched_client.messages.create.call_args.kwargs
    assert params["tools"] == tools
    assert params["tool_choice"] == {"type": "auto"}


def test_conversation_history_injected_into_system(patched_client, make_anthropic_response):
    patched_client.messages.create.return_value = make_anthropic_response(text="ok")
    gen = AIGenerator(api_key="k", model="m")

    gen.generate_response(query="hi", conversation_history="USER: earlier\nAI: reply")

    params = patched_client.messages.create.call_args.kwargs
    assert "Previous conversation:" in params["system"]
    assert "USER: earlier" in params["system"]


def test_tool_use_triggers_execution_and_second_call(patched_client, make_anthropic_response):
    # First call asks for a tool; second call returns the synthesized answer.
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("search_course_content", {"query": "mcp"})),
        make_anthropic_response(text="Final synthesized answer."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "TOOL RESULT TEXT"

    gen = AIGenerator(api_key="k", model="m")
    out = gen.generate_response(
        query="what is mcp",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert out == "Final synthesized answer."
    assert patched_client.messages.create.call_count == 2
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="mcp"
    )


def test_second_call_has_tool_result_and_no_tools(patched_client, make_anthropic_response):
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("search_course_content", {"query": "mcp"})),
        make_anthropic_response(text="done"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "TOOL RESULT TEXT"

    gen = AIGenerator(api_key="k", model="m")
    gen.generate_response(
        query="what is mcp",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second = patched_client.messages.create.call_args_list[1].kwargs
    # Single-round design: the follow-up call must NOT advertise tools.
    assert "tools" not in second

    messages = second["messages"]
    assert messages[0]["role"] == "user"          # original query
    assert messages[1]["role"] == "assistant"     # tool_use turn
    assert messages[2]["role"] == "user"          # tool_result turn

    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool_1"
    assert tool_result["content"] == "TOOL RESULT TEXT"
