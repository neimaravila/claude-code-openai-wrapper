#!/usr/bin/env python3
"""
Unit tests for tool support across models, claude_cli, message_adapter, and tool_bridge.

Tests cover:
- New Pydantic models (ToolUseBlock, ToolResultBlock, ToolCall, etc.)
- ParsedMessage with tool_use_blocks and tool_result_blocks
- parse_claude_message() detecting tool blocks from SDK output
- MessageAdapter converters (openai_tools_to_anthropic, anthropic_tool_use_to_openai)
- MessageAdapter.filter_content(preserve_tools=True)
- MessageAdapter.messages_to_prompt() with role="tool" messages
- AnthropicMessage content validation accepting tool_use/tool_result blocks
- AnthropicMessagesRequest.get_tool_result_context()
- tool_bridge.create_tools_system_prompt() and parse_tool_calls_from_text()
- Endpoint response formatting with tool_calls (OpenAI) and tool_use blocks (Anthropic)
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace

from src.models import (
    Message,
    Choice,
    StreamChoice,
    ChatCompletionRequest,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicToolInputSchema,
    AnthropicToolDefinition,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
    FunctionDefinition,
    ToolDefinition,
    FunctionCall,
    ToolCall,
)
from src.claude_cli import (
    ClaudeCodeCLI,
    ParsedMessage,
    ToolUsePart,
    ToolResultPart,
    ThinkingPart,
)
from src.message_adapter import MessageAdapter


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestAnthropicToolUseBlock:
    """Test AnthropicToolUseBlock model."""

    def test_create_tool_use_block(self):
        block = AnthropicToolUseBlock(id="toolu_123", name="get_weather", input={"city": "Tokyo"})
        assert block.type == "tool_use"
        assert block.id == "toolu_123"
        assert block.name == "get_weather"
        assert block.input == {"city": "Tokyo"}

    def test_serialize_tool_use_block(self):
        block = AnthropicToolUseBlock(
            id="toolu_123", name="read_file", input={"path": "/etc/hostname"}
        )
        data = block.model_dump()
        assert data["type"] == "tool_use"
        assert data["id"] == "toolu_123"
        assert data["name"] == "read_file"
        assert data["input"]["path"] == "/etc/hostname"


class TestAnthropicToolResultBlock:
    """Test AnthropicToolResultBlock model."""

    def test_create_tool_result_block(self):
        block = AnthropicToolResultBlock(tool_use_id="toolu_123", content="file content here")
        assert block.type == "tool_result"
        assert block.tool_use_id == "toolu_123"
        assert block.content == "file content here"
        assert block.is_error is None

    def test_tool_result_with_error(self):
        block = AnthropicToolResultBlock(
            tool_use_id="toolu_456",
            content="File not found",
            is_error=True,
        )
        assert block.is_error is True

    def test_tool_result_with_list_content(self):
        block = AnthropicToolResultBlock(
            tool_use_id="toolu_789",
            content=[{"type": "text", "text": "result text"}],
        )
        assert isinstance(block.content, list)


class TestAnthropicToolDefinition:
    """Test AnthropicToolDefinition model."""

    def test_create_tool_definition(self):
        schema = AnthropicToolInputSchema(
            properties={"city": {"type": "string"}},
            required=["city"],
        )
        tool_def = AnthropicToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            input_schema=schema,
        )
        assert tool_def.name == "get_weather"
        assert tool_def.description == "Get weather for a city"
        assert tool_def.input_schema.properties == {"city": {"type": "string"}}

    def test_tool_definition_minimal(self):
        schema = AnthropicToolInputSchema()
        tool_def = AnthropicToolDefinition(name="noop", input_schema=schema)
        assert tool_def.name == "noop"
        assert tool_def.description is None
        assert tool_def.input_schema.properties == {}
        assert tool_def.is_custom_tool is True

    def test_builtin_tool_web_search(self):
        """Built-in tools like web_search have no input_schema."""
        tool_def = AnthropicToolDefinition(
            type="web_search_20250305",
            name="web_search",
            max_uses=8,
        )
        assert tool_def.name == "web_search"
        assert tool_def.type == "web_search_20250305"
        assert tool_def.input_schema is None
        assert tool_def.is_custom_tool is False

    def test_builtin_tool_text_editor(self):
        tool_def = AnthropicToolDefinition(
            type="text_editor_20250124",
            name="str_replace_editor",
        )
        assert tool_def.is_custom_tool is False


class TestOpenAIToolModels:
    """Test OpenAI tool/function calling models."""

    def test_function_definition(self):
        func = FunctionDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert func.name == "get_weather"

    def test_tool_definition(self):
        tool = ToolDefinition(
            function=FunctionDefinition(name="search", description="Search the web")
        )
        assert tool.type == "function"
        assert tool.function.name == "search"

    def test_function_call(self):
        call = FunctionCall(name="get_weather", arguments='{"city": "Tokyo"}')
        assert call.name == "get_weather"
        assert json.loads(call.arguments) == {"city": "Tokyo"}

    def test_tool_call(self):
        tc = ToolCall(
            id="call_abc123",
            function=FunctionCall(name="search", arguments='{"query": "python"}'),
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "search"


class TestMessageWithTools:
    """Test Message model with tool support."""

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="I'll check the weather.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get_weather", arguments='{"city": "Tokyo"}'),
                )
            ],
        )
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_tool_role_message(self):
        msg = Message(
            role="tool",
            content="Sunny, 25C",
            tool_call_id="call_1",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"
        assert msg.content == "Sunny, 25C"

    def test_assistant_message_without_content(self):
        msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="search", arguments='{"q": "test"}'),
                )
            ],
        )
        assert msg.content is None
        assert msg.tool_calls is not None


class TestChoiceFinishReason:
    """Test that Choice and StreamChoice accept 'tool_calls' finish_reason."""

    def test_choice_tool_calls_finish_reason(self):
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="test"),
            finish_reason="tool_calls",
        )
        assert choice.finish_reason == "tool_calls"

    def test_stream_choice_tool_calls_finish_reason(self):
        sc = StreamChoice(index=0, delta={"content": ""}, finish_reason="tool_calls")
        assert sc.finish_reason == "tool_calls"


class TestAnthropicMessageValidation:
    """Test AnthropicMessage content validation with tool blocks."""

    def test_accepts_tool_use_block_in_content(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "Let me search that."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"query": "test"},
                },
            ],
        )
        assert len(msg.content) == 2

    def test_accepts_tool_result_block_in_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Search results here",
                },
            ],
        )
        assert len(msg.content) == 1

    def test_filters_thinking_blocks(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Result"},
            ],
        )
        # Thinking blocks should be filtered out
        assert len(msg.content) == 1

    def test_mixed_text_and_tool_blocks(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "I'll check that."},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
                {"type": "thinking", "thinking": "hmm"},
            ],
        )
        # text + tool_use kept, thinking filtered
        assert len(msg.content) == 2


class TestAnthropicMessagesRequestTools:
    """Test AnthropicMessagesRequest with tools."""

    def test_request_with_tools(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[AnthropicMessage(role="user", content="What's the weather?")],
            max_tokens=1024,
            tools=[
                AnthropicToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    input_schema=AnthropicToolInputSchema(
                        properties={"city": {"type": "string"}},
                        required=["city"],
                    ),
                )
            ],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "get_weather"

    def test_request_with_tool_choice(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=1024,
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_get_tool_result_context(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "Sunny, 25C",
                        }
                    ],
                )
            ],
            max_tokens=1024,
        )
        ctx = req.get_tool_result_context()
        assert "toolu_123" in ctx
        assert "Sunny, 25C" in ctx

    def test_get_tool_result_context_with_error(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": "Not found",
                            "is_error": True,
                        }
                    ],
                )
            ],
            max_tokens=1024,
        )
        ctx = req.get_tool_result_context()
        assert "[ERROR]" in ctx

    def test_get_tool_result_context_none_when_no_results(self):
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[AnthropicMessage(role="user", content="Just text")],
            max_tokens=1024,
        )
        assert req.get_tool_result_context() is None


class TestAnthropicMessagesResponseToolUse:
    """Test AnthropicMessagesResponse with tool_use blocks."""

    def test_response_with_tool_use(self):
        resp = AnthropicMessagesResponse(
            model="claude-sonnet-4-5-20250929",
            content=[
                AnthropicTextBlock(text="I'll check the weather."),
                AnthropicToolUseBlock(
                    id="toolu_123",
                    name="get_weather",
                    input={"city": "Tokyo"},
                ),
            ],
            stop_reason="tool_use",
            usage=AnthropicUsage(input_tokens=100, output_tokens=50),
        )
        assert resp.stop_reason == "tool_use"
        assert len(resp.content) == 2
        data = resp.model_dump()
        assert data["content"][1]["type"] == "tool_use"
        assert data["content"][1]["name"] == "get_weather"

    def test_response_tool_use_only(self):
        resp = AnthropicMessagesResponse(
            model="claude-sonnet-4-5-20250929",
            content=[
                AnthropicToolUseBlock(
                    id="toolu_789",
                    name="search",
                    input={"query": "python"},
                ),
            ],
            stop_reason="tool_use",
            usage=AnthropicUsage(input_tokens=50, output_tokens=30),
        )
        assert len(resp.content) == 1


class TestChatCompletionRequestTools:
    """Test ChatCompletionRequest with tools."""

    def test_request_with_tools(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="Hello")],
            tools=[
                ToolDefinition(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    )
                )
            ],
        )
        assert len(req.tools) == 1
        assert req.tools[0].function.name == "get_weather"

    def test_request_with_tool_choice(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="Hello")],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"


# ===========================================================================
# ParsedMessage and parse_claude_message tests
# ===========================================================================


class TestParsedMessageWithTools:
    """Test ParsedMessage dataclass with tool fields."""

    def test_parsed_message_defaults(self):
        pm = ParsedMessage()
        assert pm.tool_use_blocks is None
        assert pm.tool_result_blocks is None

    def test_parsed_message_with_tool_use(self):
        pm = ParsedMessage(
            text="Let me search.",
            tool_use_blocks=[ToolUsePart(id="toolu_1", name="search", input={"q": "test"})],
        )
        assert len(pm.tool_use_blocks) == 1
        assert pm.tool_use_blocks[0].name == "search"

    def test_parsed_message_with_tool_result(self):
        pm = ParsedMessage(
            text="Result",
            tool_result_blocks=[ToolResultPart(tool_use_id="toolu_1", content="Found it")],
        )
        assert len(pm.tool_result_blocks) == 1


class TestParseClaudeMessageToolBlocks:
    """Test parse_claude_message() detecting tool blocks."""

    @pytest.fixture
    def cli_class(self):
        return ClaudeCodeCLI

    def test_parse_tool_use_block_object(self, cli_class):
        """Detects ToolUseBlock objects in SDK output."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        # Mock ToolUseBlock - use SimpleNamespace for controlled attributes
        tool_block = SimpleNamespace(
            id="toolu_abc",
            name="get_weather",
            input={"city": "Tokyo"},
        )

        text_block = SimpleNamespace(text="I'll check the weather.")

        messages = [{"content": [text_block, tool_block]}]
        result = cli.parse_claude_message(messages)

        assert result.text == "I'll check the weather."
        assert result.tool_use_blocks is not None
        assert len(result.tool_use_blocks) == 1
        assert result.tool_use_blocks[0].name == "get_weather"
        assert result.tool_use_blocks[0].input == {"city": "Tokyo"}
        assert result.tool_use_blocks[0].id == "toolu_abc"

    def test_parse_tool_use_block_dict(self, cli_class):
        """Detects tool_use blocks in dict format."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "content": [
                    {"type": "text", "text": "Checking..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_xyz",
                        "name": "read_file",
                        "input": {"path": "/tmp/test.txt"},
                    },
                ]
            }
        ]
        result = cli.parse_claude_message(messages)

        assert result.text == "Checking..."
        assert result.tool_use_blocks is not None
        assert len(result.tool_use_blocks) == 1
        assert result.tool_use_blocks[0].name == "read_file"

    def test_parse_tool_result_block_dict(self, cli_class):
        """Detects tool_result blocks in dict format."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_xyz",
                        "content": "file contents",
                    },
                    {"type": "text", "text": "Done."},
                ]
            }
        ]
        result = cli.parse_claude_message(messages)

        assert result.text == "Done."
        assert result.tool_result_blocks is not None
        assert len(result.tool_result_blocks) == 1
        assert result.tool_result_blocks[0].tool_use_id == "toolu_xyz"

    def test_parse_tool_result_block_object(self, cli_class):
        """Detects ToolResultBlock objects with real string tool_use_id."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        result_block = SimpleNamespace(
            tool_use_id="toolu_abc",
            content="result text",
            is_error=False,
        )
        text_block = SimpleNamespace(text="The file says: ...")

        messages = [{"content": [result_block, text_block]}]
        result = cli.parse_claude_message(messages)

        assert result.text == "The file says: ..."
        assert result.tool_result_blocks is not None
        assert len(result.tool_result_blocks) == 1

    def test_parse_no_tool_blocks(self, cli_class):
        """Returns None for tool fields when no tool blocks present."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [{"content": [{"type": "text", "text": "Just text"}]}]
        result = cli.parse_claude_message(messages)

        assert result.text == "Just text"
        assert result.tool_use_blocks is None
        assert result.tool_result_blocks is None

    def test_parse_mixed_thinking_and_tool_blocks(self, cli_class):
        """Handles mix of thinking, text, and tool blocks."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Answer"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "search",
                        "input": {"q": "test"},
                    },
                ]
            }
        ]
        result = cli.parse_claude_message(messages)

        assert result.text == "Answer"
        assert result.reasoning is not None
        assert result.tool_use_blocks is not None
        assert len(result.tool_use_blocks) == 1

    def test_parse_tool_use_in_old_format(self, cli_class):
        """Detects tool_use blocks in old SDK message format."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Reading file..."},
                        {
                            "type": "tool_use",
                            "id": "toolu_old",
                            "name": "read",
                            "input": {"path": "/tmp/file"},
                        },
                    ]
                },
            }
        ]
        result = cli.parse_claude_message(messages)

        assert result.text == "Reading file..."
        assert result.tool_use_blocks is not None
        assert result.tool_use_blocks[0].name == "read"


# ===========================================================================
# MessageAdapter tests
# ===========================================================================


class TestFilterContentPreserveTools:
    """Test MessageAdapter.filter_content with preserve_tools parameter."""

    def test_preserve_tools_keeps_tool_xml(self):
        content = "Text before <read_file>/tmp/test</read_file> text after"
        result = MessageAdapter.filter_content(content, preserve_tools=True)
        assert "<read_file>" in result

    def test_preserve_tools_false_removes_tool_xml(self):
        content = "Text before <read_file>/tmp/test</read_file> text after"
        result = MessageAdapter.filter_content(content, preserve_tools=False)
        assert "<read_file>" not in result

    def test_preserve_tools_keeps_bash_xml(self):
        content = "Running: <bash>ls -la</bash>"
        result = MessageAdapter.filter_content(content, preserve_tools=True)
        assert "<bash>" in result

    def test_preserve_tools_still_removes_thinking_when_needed(self):
        content = "<thinking>Let me think</thinking> Result"
        result = MessageAdapter.filter_content(
            content, preserve_tools=True, preserve_thinking=False
        )
        assert "<thinking>" not in result
        assert "Result" in result


class TestMessagesToPromptWithTools:
    """Test MessageAdapter.messages_to_prompt with tool role messages."""

    def test_tool_role_message(self):
        messages = [
            Message(role="user", content="What's the weather?"),
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city": "Tokyo"}',
                        ),
                    )
                ],
            ),
            Message(
                role="tool",
                content="Sunny, 25C",
                tool_call_id="call_1",
                name="get_weather",
            ),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert "get_weather" in prompt
        assert "Sunny, 25C" in prompt
        assert "call_1" in prompt

    def test_assistant_tool_calls_in_prompt(self):
        messages = [
            Message(role="user", content="Search for python"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        function=FunctionCall(
                            name="search",
                            arguments='{"query": "python"}',
                        ),
                    )
                ],
            ),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)
        assert "search" in prompt
        assert "python" in prompt


class TestOpenAIToolsToAnthropic:
    """Test MessageAdapter.openai_tools_to_anthropic converter."""

    def test_convert_single_tool(self):
        tools = [
            ToolDefinition(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            )
        ]
        result = MessageAdapter.openai_tools_to_anthropic(tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"
        assert result[0]["input_schema"]["properties"]["city"]["type"] == "string"

    def test_convert_tool_without_parameters(self):
        tools = [
            ToolDefinition(
                function=FunctionDefinition(name="no_params", description="A tool with no params")
            )
        ]
        result = MessageAdapter.openai_tools_to_anthropic(tools)

        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_convert_multiple_tools(self):
        tools = [
            ToolDefinition(function=FunctionDefinition(name="tool_a", description="Tool A")),
            ToolDefinition(function=FunctionDefinition(name="tool_b", description="Tool B")),
        ]
        result = MessageAdapter.openai_tools_to_anthropic(tools)
        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"


class TestAnthropicToolUseToOpenAI:
    """Test MessageAdapter.anthropic_tool_use_to_openai converter."""

    def test_convert_single_tool_use(self):
        blocks = [ToolUsePart(id="toolu_abc", name="get_weather", input={"city": "Tokyo"})]
        result = MessageAdapter.anthropic_tool_use_to_openai(blocks)

        assert len(result) == 1
        assert result[0]["id"] == "toolu_abc"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"city": "Tokyo"}

    def test_convert_multiple_tool_uses(self):
        blocks = [
            ToolUsePart(id="t1", name="tool_a", input={"x": 1}),
            ToolUsePart(id="t2", name="tool_b", input={"y": 2}),
        ]
        result = MessageAdapter.anthropic_tool_use_to_openai(blocks)
        assert len(result) == 2

    def test_convert_empty_input(self):
        blocks = [ToolUsePart(id="t1", name="noop", input={})]
        result = MessageAdapter.anthropic_tool_use_to_openai(blocks)
        assert json.loads(result[0]["function"]["arguments"]) == {}


# ===========================================================================
# tool_bridge tests
# ===========================================================================


class TestToolBridge:
    """Test tool_bridge module."""

    def test_anthropic_tool_defs_to_bridge_format(self):
        from src.tool_bridge import anthropic_tool_defs_to_bridge_format

        tool_defs = [
            AnthropicToolDefinition(
                name="get_weather",
                description="Get weather",
                input_schema=AnthropicToolInputSchema(
                    properties={"city": {"type": "string"}},
                    required=["city"],
                ),
            )
        ]
        result = anthropic_tool_defs_to_bridge_format(tool_defs)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert "properties" in result[0]["input_schema"]

    def test_anthropic_tool_defs_skips_builtin_tools(self):
        """Built-in tools without input_schema are skipped."""
        from src.tool_bridge import anthropic_tool_defs_to_bridge_format

        tool_defs = [
            AnthropicToolDefinition(
                type="web_search_20250305",
                name="web_search",
                max_uses=8,
            ),
            AnthropicToolDefinition(
                name="custom_tool",
                description="A custom tool",
                input_schema=AnthropicToolInputSchema(
                    properties={"q": {"type": "string"}},
                ),
            ),
        ]
        result = anthropic_tool_defs_to_bridge_format(tool_defs)

        assert len(result) == 1
        assert result[0]["name"] == "custom_tool"

    def test_anthropic_tool_defs_all_builtin(self):
        """When all tools are built-in, result is empty."""
        from src.tool_bridge import anthropic_tool_defs_to_bridge_format

        tool_defs = [
            AnthropicToolDefinition(
                type="web_search_20250305",
                name="web_search",
            ),
        ]
        result = anthropic_tool_defs_to_bridge_format(tool_defs)
        assert len(result) == 0

    def test_create_tools_system_prompt(self):
        from src.tool_bridge import create_tools_system_prompt

        tool_defs = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            }
        ]
        prompt = create_tools_system_prompt(tool_defs)
        assert "get_weather" in prompt
        assert "Get weather for a city" in prompt
        assert "city" in prompt
        assert "(required)" in prompt
        assert "<tool_call>" in prompt

    def test_create_tools_system_prompt_required_none(self):
        """Regression: model_dump() produces required=None, must not crash."""
        from src.tool_bridge import create_tools_system_prompt

        tool_defs = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": None,
                },
            }
        ]
        prompt = create_tools_system_prompt(tool_defs)
        assert "test_tool" in prompt
        assert "(required)" not in prompt

    def test_create_tools_system_prompt_from_model_dump(self):
        """End-to-end: AnthropicToolInputSchema.model_dump() -> create_tools_system_prompt."""
        from src.tool_bridge import anthropic_tool_defs_to_bridge_format, create_tools_system_prompt

        tool_defs = [
            AnthropicToolDefinition(
                name="noop",
                input_schema=AnthropicToolInputSchema(
                    properties={"arg": {"type": "string"}},
                ),
            )
        ]
        bridge_defs = anthropic_tool_defs_to_bridge_format(tool_defs)
        # This should not crash even though required is None in model_dump()
        prompt = create_tools_system_prompt(bridge_defs)
        assert "noop" in prompt

    def test_create_tools_system_prompt_multiple_tools(self):
        from src.tool_bridge import create_tools_system_prompt

        tool_defs = [
            {
                "name": "tool_a",
                "description": "Tool A",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "tool_b",
                "description": "Tool B",
                "input_schema": {"type": "object", "properties": {"x": {"type": "int"}}},
            },
        ]
        prompt = create_tools_system_prompt(tool_defs)
        assert "tool_a" in prompt
        assert "tool_b" in prompt
        assert "Tool A" in prompt
        assert "Tool B" in prompt

    def test_create_tools_system_prompt_no_properties(self):
        from src.tool_bridge import create_tools_system_prompt

        tool_defs = [
            {
                "name": "noop",
                "description": "Does nothing",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        prompt = create_tools_system_prompt(tool_defs)
        assert "noop" in prompt
        assert "Parameters: none" in prompt

    def test_parse_tool_calls_from_text_single(self):
        from src.tool_bridge import parse_tool_calls_from_text

        text = 'I\'ll check the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        calls, remaining = parse_tool_calls_from_text(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["input"] == {"city": "Tokyo"}
        assert calls[0]["id"].startswith("toolu_")
        assert "<tool_call>" not in remaining
        assert "check the weather" in remaining

    def test_parse_tool_calls_from_text_multiple(self):
        from src.tool_bridge import parse_tool_calls_from_text

        text = (
            '<tool_call>\n{"name": "tool_a", "arguments": {"x": 1}}\n</tool_call>\n'
            "Some text\n"
            '<tool_call>\n{"name": "tool_b", "arguments": {"y": 2}}\n</tool_call>'
        )
        calls, remaining = parse_tool_calls_from_text(text)

        assert len(calls) == 2
        assert calls[0]["name"] == "tool_a"
        assert calls[1]["name"] == "tool_b"

    def test_parse_tool_calls_from_text_none(self):
        from src.tool_bridge import parse_tool_calls_from_text

        text = "Just regular text, no tool calls."
        calls, remaining = parse_tool_calls_from_text(text)

        assert len(calls) == 0
        assert remaining == text

    def test_parse_tool_calls_from_text_invalid_json(self):
        from src.tool_bridge import parse_tool_calls_from_text

        text = "<tool_call>\n{invalid json}\n</tool_call>"
        calls, remaining = parse_tool_calls_from_text(text)

        assert len(calls) == 0

    def test_parse_tool_calls_input_key(self):
        from src.tool_bridge import parse_tool_calls_from_text

        text = '<tool_call>\n{"name": "test", "input": {"key": "val"}}\n</tool_call>'
        calls, remaining = parse_tool_calls_from_text(text)

        assert len(calls) == 1
        assert calls[0]["input"] == {"key": "val"}


# ===========================================================================
# Endpoint integration tests (using TestClient mock)
# ===========================================================================


class TestEndpointToolResponse:
    """Test endpoint response formatting with tools (mocked SDK)."""

    @pytest.fixture
    def mock_cli(self):
        """Create a mock ClaudeCodeCLI that returns tool_use blocks."""
        with patch("src.main.claude_cli") as mock:

            async def mock_run(*args, **kwargs):
                yield {
                    "content": [
                        SimpleNamespace(text="Let me check."),
                        SimpleNamespace(
                            id="toolu_test",
                            name="get_weather",
                            input={"city": "Tokyo"},
                        ),
                    ]
                }

            mock.run_completion = mock_run

            # parse_claude_message needs to work properly
            mock.parse_claude_message = ClaudeCodeCLI.parse_claude_message.__get__(
                mock, ClaudeCodeCLI
            )
            mock.estimate_token_usage = ClaudeCodeCLI.estimate_token_usage.__get__(
                mock, ClaudeCodeCLI
            )
            yield mock

    @pytest.fixture
    def client(self, mock_cli):
        from fastapi.testclient import TestClient
        from src.main import app

        return TestClient(app, raise_server_exceptions=False)

    def test_anthropic_non_streaming_tool_use(self, client):
        """Anthropic /v1/messages returns tool_use blocks and stop_reason=tool_use."""
        with patch("src.main.validate_claude_code_auth", return_value=(True, {"method": "test"})):
            with patch("src.main.verify_api_key", new_callable=AsyncMock):
                response = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-5-20250929",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "What's the weather?"}],
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["stop_reason"] == "tool_use"

        # Find tool_use block
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) >= 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "Tokyo"}

    def test_openai_non_streaming_tool_calls(self, client):
        """OpenAI /v1/chat/completions returns tool_calls and finish_reason=tool_calls."""
        with patch("src.main.validate_claude_code_auth", return_value=(True, {"method": "test"})):
            with patch("src.main.verify_api_key", new_callable=AsyncMock):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "claude-sonnet-4-5-20250929",
                        "messages": [{"role": "user", "content": "What's the weather?"}],
                        "enable_tools": True,
                    },
                )

        assert response.status_code == 200
        data = response.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert "tool_calls" in choice["message"]
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}
