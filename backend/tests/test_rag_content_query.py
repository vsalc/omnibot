"""Tests for how RAGSystem handles content-query questions (backend/rag_system.py).

Two layers:
  * Orchestration  — RAGSystem.query wiring, with AIGenerator/VectorStore mocked.
  * Integration    — a real VectorStore + real CourseSearchTool, with only the
                     Anthropic client mocked to drive the tool call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag_system import RAGSystem


def _config(tmp_path):
    return SimpleNamespace(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH=str(tmp_path / "chroma"),
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="test-model",
    )


# --------------------------------------------------------------------------- #
# Orchestration (mocked AIGenerator + VectorStore)
# --------------------------------------------------------------------------- #
@pytest.fixture
def orchestration_rag(tmp_path):
    """RAGSystem with heavy collaborators patched out at construction time."""
    with patch("rag_system.VectorStore"), patch("rag_system.AIGenerator"):
        rag = RAGSystem(_config(tmp_path))
    return rag


def test_query_wires_prompt_history_and_tools(orchestration_rag):
    rag = orchestration_rag
    rag.ai_generator.generate_response.return_value = "the answer"

    rag.query("What is MCP?")

    kwargs = rag.ai_generator.generate_response.call_args.kwargs
    assert "What is MCP?" in kwargs["query"]
    assert kwargs["tool_manager"] is rag.tool_manager
    assert kwargs["tools"] == rag.tool_manager.get_tool_definitions()
    assert kwargs["conversation_history"] is None  # no session given


def test_query_collects_and_resets_sources(orchestration_rag):
    rag = orchestration_rag
    rag.ai_generator.generate_response.return_value = "the answer"
    # Simulate a search having stashed sources during the call.
    rag.search_tool.last_sources = [{"text": "MCP Course - Lesson 1", "link": "u"}]

    answer, sources = rag.query("What is MCP?")

    assert answer == "the answer"
    assert sources == [{"text": "MCP Course - Lesson 1", "link": "u"}]
    # Sources are reset afterward so the next query starts clean.
    assert rag.tool_manager.get_last_sources() == []


def test_query_persists_session_history(orchestration_rag):
    rag = orchestration_rag
    rag.ai_generator.generate_response.return_value = "the answer"
    session_id = rag.session_manager.create_session()

    rag.query("What is MCP?", session_id=session_id)
    history = rag.session_manager.get_conversation_history(session_id)

    assert "What is MCP?" in history
    assert "the answer" in history


# --------------------------------------------------------------------------- #
# Integration (real vector store + real search tool, mocked LLM only)
# --------------------------------------------------------------------------- #
def _seed(rag):
    from models import Course, Lesson, CourseChunk

    rag.vector_store.add_course_metadata(
        Course(
            title="MCP Course",
            course_link="https://example.com/mcp",
            instructor="Ada",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Intro",
                    lesson_link="https://example.com/mcp/1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Servers",
                    lesson_link="https://example.com/mcp/2",
                ),
            ],
        )
    )
    rag.vector_store.add_course_content(
        [
            CourseChunk(
                content="The Model Context Protocol lets clients call tools over a server.",
                course_title="MCP Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="An MCP server exposes resources and tools to a host application.",
                course_title="MCP Course",
                lesson_number=2,
                chunk_index=1,
            ),
        ]
    )


def test_content_query_runs_real_search_end_to_end(tmp_path, make_anthropic_response):
    """A content question drives a real search_course_content tool call."""
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        make_anthropic_response(
            tool_use=("search_course_content", {"query": "model context protocol"})
        ),
        make_anthropic_response(text="MCP lets clients call tools over a server."),
    ]

    with patch("ai_generator.anthropic.Anthropic", return_value=fake_client):
        rag = RAGSystem(_config(tmp_path))
        _seed(rag)

        answer, sources = rag.query("What is the model context protocol?")

    # Final synthesized answer comes from the (mocked) second LLM call.
    assert answer == "MCP lets clients call tools over a server."

    # The real CourseSearchTool ran against the seeded store and produced sources.
    assert sources, "expected sources from the real search"
    assert any("MCP Course" in s["text"] for s in sources)

    # The tool_result fed back to the LLM contains real retrieved content.
    second_call = fake_client.messages.create.call_args_list[1].kwargs
    tool_result = second_call["messages"][2]["content"][0]
    assert (
        "MCP" in tool_result["content"]
        or "Model Context Protocol" in tool_result["content"]
    )
