"""Bridge between client-defined tools and the Claude Agent SDK MCP server.

Uses create_sdk_mcp_server() to register client tools as in-process MCP tools.
Each tool has a stub handler that returns a sentinel value, because with max_turns=1
the SDK stops after generating a ToolUseBlock and returns it to the client.
The client then executes the tool and sends the result back in the next request.
"""

import logging
from typing import Any, Dict, List, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool

logger = logging.getLogger(__name__)

# Sentinel value returned by stub handlers
PENDING_CLIENT_EXECUTION = "[PENDING_CLIENT_EXECUTION]"


def create_client_tools_mcp_server(
    tool_definitions: List[Dict[str, Any]],
    server_name: str = "client_tools",
) -> Any:
    """Create an MCP server from client-defined tool definitions.

    Args:
        tool_definitions: List of tool defs in Anthropic format, each with:
            - name: str
            - description: Optional[str]
            - input_schema: dict with type/properties/required
        server_name: Name for the MCP server instance.

    Returns:
        McpSdkServerConfig ready to pass to ClaudeAgentOptions.mcp_servers.
    """
    sdk_tools = []

    for tool_def in tool_definitions:
        name = tool_def["name"]
        description = tool_def.get("description", f"Client tool: {name}")
        input_schema = tool_def.get("input_schema", {"type": "object", "properties": {}})

        # Build a proper JSON Schema dict for the SDK
        schema = _build_schema(input_schema)

        # Create the tool with a stub handler
        sdk_tool = _create_stub_tool(name, description, schema)
        sdk_tools.append(sdk_tool)
        logger.debug(f"Registered client tool: {name}")

    server = create_sdk_mcp_server(
        name=server_name,
        tools=sdk_tools if sdk_tools else None,
    )

    logger.info(f"Created MCP server '{server_name}' with {len(sdk_tools)} client tools")
    return server


def _build_schema(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON Schema dict from an Anthropic input_schema."""
    schema = {
        "type": input_schema.get("type", "object"),
        "properties": input_schema.get("properties", {}),
    }
    if "required" in input_schema:
        schema["required"] = input_schema["required"]
    return schema


def _create_stub_tool(name: str, description: str, schema: Dict[str, Any]) -> Any:
    """Create a stub SDK tool that returns a sentinel value.

    The stub handler is never actually called in practice because we use
    max_turns=1, causing the SDK to stop after generating the ToolUseBlock.
    """

    @tool(name, description, schema)
    async def stub_handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": PENDING_CLIENT_EXECUTION,
                }
            ]
        }

    return stub_handler


def anthropic_tool_defs_to_bridge_format(
    tool_defs: List[Any],
) -> List[Dict[str, Any]]:
    """Convert AnthropicToolDefinition Pydantic models to plain dicts for the bridge.

    Args:
        tool_defs: List of AnthropicToolDefinition models.

    Returns:
        List of dicts with name, description, input_schema.
    """
    result = []
    for td in tool_defs:
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
