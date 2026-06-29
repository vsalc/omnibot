"""Shared pytest fixtures for the backend test suite.

The backend uses flat imports (``from vector_store import ...``); pytest is
configured with ``pythonpath = ["backend"]`` in ``pyproject.toml`` so those
imports resolve when the suite runs from the repo root.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


# --------------------------------------------------------------------------- #
# CourseSearchTool / VectorStore fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_vector_store():
    """A VectorStore double. Tests set ``.search.return_value`` per case."""
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = "https://example.com/lesson"
    return store


@pytest.fixture
def sample_search_results():
    """Two content hits with course/lesson metadata."""
    return SearchResults(
        documents=["Chunk about MCP basics.", "Chunk about MCP servers."],
        metadata=[
            {"course_title": "MCP Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "MCP Course", "lesson_number": 2, "chunk_index": 1},
        ],
        distances=[0.1, 0.2],
    )


# --------------------------------------------------------------------------- #
# Anthropic response stubs (no real SDK / network)
# --------------------------------------------------------------------------- #
def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, tool_input, block_id="tool_1"):
    return SimpleNamespace(
        type="tool_use", name=name, input=tool_input, id=block_id
    )


@pytest.fixture
def make_anthropic_response():
    """Factory building stub Anthropic responses.

    Usage:
        make_anthropic_response(text="hi")
        make_anthropic_response(tool_use=("search_course_content", {"query": "x"}))
    """

    def _make(text=None, tool_use=None, block_id="tool_1"):
        if tool_use is not None:
            name, tool_input = tool_use
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[_tool_use_block(name, tool_input, block_id)],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[_text_block(text if text is not None else "")],
        )

    return _make


# --------------------------------------------------------------------------- #
# Real (temp) vector store for the integration test
# --------------------------------------------------------------------------- #
@pytest.fixture
def seeded_vector_store(tmp_path):
    """A real VectorStore on a temp Chroma dir, seeded with one tiny course.

    Reuses the already-downloaded embedding model. Slower than the mocked
    fixtures; used only by the end-to-end content-query test.
    """
    store = VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )

    course = Course(
        title="MCP Course",
        course_link="https://example.com/mcp",
        instructor="Ada",
        lessons=[
            Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/mcp/1"),
            Lesson(lesson_number=2, title="Servers", lesson_link="https://example.com/mcp/2"),
        ],
    )
    chunks = [
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
    store.add_course_metadata(course)
    store.add_course_content(chunks)
    return store
