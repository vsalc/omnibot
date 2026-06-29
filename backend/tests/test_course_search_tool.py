"""Tests for CourseSearchTool.execute (backend/search_tools.py)."""

from search_tools import CourseSearchTool
from vector_store import SearchResults


def test_returns_formatted_results_with_headers(
    mock_vector_store, sample_search_results
):
    mock_vector_store.search.return_value = sample_search_results
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="what is mcp")

    assert "[MCP Course - Lesson 1]" in out
    assert "[MCP Course - Lesson 2]" in out
    assert "Chunk about MCP basics." in out
    assert "Chunk about MCP servers." in out


def test_forwards_course_and_lesson_filters(mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="topic", course_name="MCP", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="topic", course_name="MCP", lesson_number=2
    )


def test_returns_error_string_verbatim(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty("boom: db unreachable")
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="anything")

    assert out == "boom: db unreachable"


def test_empty_results_message_includes_filters(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="nope", course_name="MCP", lesson_number=3)

    assert "No relevant content found" in out
    assert "in course 'MCP'" in out
    assert "in lesson 3" in out


def test_populates_last_sources_with_links(mock_vector_store, sample_search_results):
    mock_vector_store.search.return_value = sample_search_results
    mock_vector_store.get_lesson_link.return_value = "https://example.com/mcp/1"
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="what is mcp")

    assert len(tool.last_sources) == 2
    first = tool.last_sources[0]
    assert first["text"] == "MCP Course - Lesson 1"
    assert first["link"] == "https://example.com/mcp/1"
    # Lesson link resolution was requested for each lesson-scoped hit.
    mock_vector_store.get_lesson_link.assert_any_call("MCP Course", 1)
    mock_vector_store.get_lesson_link.assert_any_call("MCP Course", 2)


def test_source_without_lesson_has_no_link(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["General course blurb."],
        metadata=[{"course_title": "MCP Course", "chunk_index": 0}],
        distances=[0.1],
    )
    tool = CourseSearchTool(mock_vector_store)

    out = tool.execute(query="overview")

    assert "[MCP Course]" in out
    assert tool.last_sources == [{"text": "MCP Course", "link": None}]


def test_tool_definition_shape(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    defn = tool.get_tool_definition()

    assert defn["name"] == "search_course_content"
    assert defn["input_schema"]["required"] == ["query"]
    assert "query" in defn["input_schema"]["properties"]
