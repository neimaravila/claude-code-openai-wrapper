"""Bridge between client-defined tools and the Claude Agent SDK.

Since the SDK runs Claude as a subprocess, we cannot use in-process MCP servers.
Instead, we describe tools in the system prompt and parse Claude's response
for tool call patterns, converting them into proper ToolUseBlock format.

The flow:
1. Client sends tool definitions in the request
2. We append tool descriptions to the system prompt
3. Claude responds with tool call JSON (it naturally does this for described tools)
4. We parse the JSON from the response text
5. Return proper ToolUsePart objects to the caller
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# System prompt generation
# ============================================================================

TOOL_SYSTEM_PROMPT_HEADER = """You have access to the following tools. When you want to use a tool, you MUST respond with a JSON tool call block in this exact format (one block per tool call):

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

You may include text before or after tool call blocks. You may make multiple tool calls.
Do NOT describe what you would do - actually make the tool call using the format above.

Available tools:
"""


def create_tools_system_prompt(tool_definitions: List[Dict[str, Any]]) -> str:
    """Generate a system prompt section describing available tools.

    Args:
        tool_definitions: List of tool defs, each with:
            - name: str
            - description: Optional[str]
            - input_schema: dict with type/properties/required

    Returns:
        System prompt text describing the tools.
    """
    parts = [TOOL_SYSTEM_PROMPT_HEADER]

    for tool_def in tool_definitions:
        name = tool_def["name"]
        description = tool_def.get("description", "No description provided")
        input_schema = tool_def.get("input_schema", {"type": "object", "properties": {}})

        parts.append(f"### {name}")
        parts.append(f"Description: {description}")

        properties = input_schema.get("properties", {})
        required = input_schema.get("required") or []

        if properties:
            parts.append("Parameters:")
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "any")
                prop_desc = prop_schema.get("description", "")
                req_marker = " (required)" if prop_name in required else ""
                parts.append(f"  - {prop_name}: {prop_type}{req_marker} - {prop_desc}")
        else:
            parts.append("Parameters: none")

        parts.append("")

    return "\n".join(parts)


# ============================================================================
# Tool call parsing from Claude's text response
# ============================================================================

# Pattern to match <tool_call>...</tool_call> blocks
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls_from_text(
    text: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Parse tool call blocks from Claude's text response.

    Extracts <tool_call>{"name": ..., "arguments": ...}</tool_call> patterns
    from the text.

    Args:
        text: The raw text response from Claude.

    Returns:
        Tuple of (tool_calls, remaining_text):
        - tool_calls: List of dicts with id, name, input keys
        - remaining_text: Text with tool_call blocks removed
    """
    tool_calls = []
    remaining_text = text

    for match in TOOL_CALL_PATTERN.finditer(text):
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", parsed.get("input", {}))

            tool_call = {
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": arguments if isinstance(arguments, dict) else {},
            }
            tool_calls.append(tool_call)
            logger.debug(f"Parsed tool call: {name}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")

    # Remove tool_call blocks from text
    if tool_calls:
        remaining_text = TOOL_CALL_PATTERN.sub("", text).strip()
        # Clean up extra whitespace
        remaining_text = re.sub(r"\n\s*\n\s*\n", "\n\n", remaining_text).strip()

    return tool_calls, remaining_text


# ============================================================================
# Format converters
# ============================================================================


def anthropic_tool_defs_to_bridge_format(
    tool_defs: List[Any],
) -> List[Dict[str, Any]]:
    """Convert AnthropicToolDefinition Pydantic models to plain dicts.

    Args:
        tool_defs: List of AnthropicToolDefinition models.

    Returns:
        List of dicts with name, description, input_schema.
    """
    result = []
    for td in tool_defs:
        # Skip built-in tools (web_search, text_editor, etc.) - they have no input_schema
        if hasattr(td, "is_custom_tool") and not td.is_custom_tool:
            logger.debug(f"Skipping built-in tool: {td.name}")
            continue
        if not hasattr(td, "input_schema") or td.input_schema is None:
            logger.debug(f"Skipping tool without input_schema: {td.name}")
            continue
        entry = {
            "name": td.name,
            "description": td.description or f"Client tool: {td.name}",
            "input_schema": (
                td.input_schema.model_dump()
                if hasattr(td.input_schema, "model_dump")
                else td.input_schema
            ),
        }
        result.append(entry)
    return result
