import anthropic
from typing import List, Optional, Dict, Any, Tuple


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool-calling rounds per query. Each round is a
    # separate API request in which Claude can reason over the previous round's
    # results before deciding whether to call another tool.
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content** — use **only** for questions about specific course content or detailed educational materials.
- **get_course_outline** — use for questions about a course's outline, structure, syllabus, or which lessons it contains.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Sequential tool use:
- You may use tools across multiple steps. After a tool returns, review its
  results and decide whether another tool call is needed to fully answer.
- Typical multi-step pattern: use get_course_outline to find a lesson title or
  structure, then use search_course_content with that information.
- Prefer the fewest steps that answer the question. If one tool call suffices,
  stop and answer. Once you have enough information, answer directly.

Outline Responses:
- When answering an outline query, always include the **course title**, the **course link**, and for **each lesson its number and title**.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Bounded agentic loop: up to MAX_TOOL_ROUNDS tool-using rounds, each a
        # separate API request that re-advertises the tools so Claude can issue a
        # follow-up tool call after seeing the previous round's results.
        for _ in range(self.MAX_TOOL_ROUNDS):
            params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**params)

            # Termination (b): Claude requested no tools (or we have no way to run
            # them) — its response is the answer.
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks or not tool_manager:
                return self._extract_text(response)

            # Record the tool-use turn and run the requested tools.
            messages.append({"role": "assistant", "content": response.content})
            tool_results, had_error = self._execute_tool_calls(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            # Termination (c): a tool failed — stop looping and let Claude explain.
            if had_error:
                break

        # Reached only via (a) the round cap or (c) a tool error. The message list
        # ends with a valid tool_result turn; make a final tools-stripped call so
        # Claude synthesizes a natural-language answer instead of another tool call.
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content,
        )
        return self._extract_text(final_response)

    def _execute_tool_calls(
        self, response, tool_manager
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool_use block in a response and collect their results.

        Each tool_use block gets exactly one matching tool_result (success or
        error) — the Messages API rejects a follow-up turn that omits a result for
        any tool_use id. A failed tool sets the error flag (which stops further
        rounds) but does not skip its sibling tools.

        Returns:
            (tool_result_blocks, had_error)
        """
        tool_results = []
        had_error = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                content = tool_manager.execute_tool(block.name, **block.input)
                result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                }
            except Exception as exc:
                had_error = True
                result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Tool '{block.name}' failed: {exc}",
                    "is_error": True,
                }
            tool_results.append(result)
        return tool_results, had_error

    @staticmethod
    def _extract_text(response) -> str:
        """Return the first text block's text (the first content block may be a
        thinking or tool_use block, so indexing content[0] is unsafe)."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
