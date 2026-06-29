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


def test_second_round_readvertises_tools_with_tool_result(patched_client, make_anthropic_response):
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
    # Sequential design: the follow-up round still advertises tools so Claude can
    # issue another tool call after seeing the first round's results.
    assert "tools" in second
    assert second["tool_choice"] == {"type": "auto"}

    messages = second["messages"]
    assert messages[0]["role"] == "user"          # original query
    assert messages[1]["role"] == "assistant"     # tool_use turn
    assert messages[2]["role"] == "user"          # tool_result turn

    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool_1"
    assert tool_result["content"] == "TOOL RESULT TEXT"


def test_two_rounds_then_synthesis_without_tools(patched_client, make_anthropic_response):
    # Round 1 outline -> round 2 search -> tools-stripped synthesis call.
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("get_course_outline", {"course_name": "X"}), block_id="tool_1"),
        make_anthropic_response(tool_use=("search_course_content", {"query": "topic"}), block_id="tool_2"),
        make_anthropic_response(text="final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "R"

    gen = AIGenerator(api_key="k", model="m")
    out = gen.generate_response(
        query="find a course like lesson 4 of X",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert out == "final answer"
    assert patched_client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2

    # The final (synthesis) call must not advertise tools — this enforces the cap.
    third = patched_client.messages.create.call_args_list[2].kwargs
    assert "tools" not in third


def test_round_two_tool_is_executed(patched_client, make_anthropic_response):
    # The cap is on rounds, not on executing round 2's tool.
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("get_course_outline", {"course_name": "X"}), block_id="tool_1"),
        make_anthropic_response(tool_use=("search_course_content", {"query": "topic"}), block_id="tool_2"),
        make_anthropic_response(text="final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "R"

    gen = AIGenerator(api_key="k", model="m")
    gen.generate_response(
        query="find a course like lesson 4 of X",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_call = tool_manager.execute_tool.call_args_list[1]
    assert second_call.args[0] == "search_course_content"
    assert second_call.kwargs == {"query": "topic"}


def test_tool_failure_terminates_with_graceful_synthesis(patched_client, make_anthropic_response):
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("search_course_content", {"query": "mcp"})),
        make_anthropic_response(text="Sorry, the search failed."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("boom")

    gen = AIGenerator(api_key="k", model="m")
    out = gen.generate_response(
        query="what is mcp",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    # One round attempted, then a synthesis call — no second tool round.
    assert out == "Sorry, the search failed."
    assert tool_manager.execute_tool.call_count == 1
    assert patched_client.messages.create.call_count == 2

    synthesis = patched_client.messages.create.call_args_list[1].kwargs
    assert "tools" not in synthesis

    # The error is surfaced to Claude (not the user) via an is_error tool_result.
    tool_result = synthesis["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "boom" in tool_result["content"]


def test_context_preserved_between_rounds(patched_client, make_anthropic_response):
    patched_client.messages.create.side_effect = [
        make_anthropic_response(tool_use=("get_course_outline", {"course_name": "X"}), block_id="tool_1"),
        make_anthropic_response(tool_use=("search_course_content", {"query": "topic"}), block_id="tool_2"),
        make_anthropic_response(text="final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "R"

    gen = AIGenerator(api_key="k", model="m")
    gen.generate_response(
        query="find a course like lesson 4 of X",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    # The synthesis call sees the full ordered history of both rounds.
    messages = patched_client.messages.create.call_args_list[2].kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"][0]["tool_use_id"] == "tool_1"
    assert messages[4]["content"][0]["tool_use_id"] == "tool_2"
