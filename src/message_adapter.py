from typing import List, Optional, Dict, Any
from src.models import Message
import json
import re


class MessageAdapter:
    """Converts between OpenAI message format and Claude Code prompts."""

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude Code prompt format.
        Returns (prompt, system_prompt)
        """
        system_prompt = None
        conversation_parts = []

        for message in messages:
            if message.role == "system":
                # Use the last system message as the system prompt
                system_prompt = message.content if message.content else ""
            elif message.role == "user":
                conversation_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                content = message.content or ""
                # Include tool_calls info in the conversation context
                if message.tool_calls:
                    tool_info = []
                    for tc in message.tool_calls:
                        tool_info.append(
                            f"[Called tool '{tc.function.name}' "
                            f"with args: {tc.function.arguments}]"
                        )
                    if content:
                        content += "\n" + "\n".join(tool_info)
                    else:
                        content = "\n".join(tool_info)
                conversation_parts.append(f"Assistant: {content}")
            elif message.role == "tool":
                # Convert tool result message to context
                tool_id = message.tool_call_id or "unknown"
                tool_name = message.name or "unknown"
                result_content = message.content or ""
                conversation_parts.append(
                    f"Tool result for '{tool_name}' (id={tool_id}): {result_content}"
                )

        # Join conversation parts
        prompt = "\n\n".join(conversation_parts)

        # If the last message wasn't from the user, add a prompt for assistant
        if messages and messages[-1].role != "user":
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def filter_content(
        content: str, preserve_thinking: bool = False, preserve_tools: bool = False
    ) -> str:
        """
        Filter content for unsupported features and tool usage.
        Remove thinking blocks, tool calls, and image references.

        Args:
            content: The content to filter.
            preserve_thinking: If True, skip removal of <thinking> blocks
                (used when reasoning_effort is active so thinking content
                 can be captured separately).
            preserve_tools: If True, skip removal of tool XML patterns
                (used when tools are enabled so tool calls are preserved).
        """
        if not content:
            return content

        # Remove thinking blocks (common when tools are disabled but Claude tries to think)
        if not preserve_thinking:
            thinking_pattern = r"<thinking>.*?</thinking>"
            content = re.sub(thinking_pattern, "", content, flags=re.DOTALL)

        if not preserve_tools:
            # Extract content from attempt_completion blocks (these contain the actual user response)
            attempt_completion_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
            attempt_matches = re.findall(attempt_completion_pattern, content, flags=re.DOTALL)
            if attempt_matches:
                # Use the content from the attempt_completion block
                extracted_content = attempt_matches[0].strip()

                # If there's a <result> tag inside, extract from that
                result_pattern = r"<result>(.*?)</result>"
                result_matches = re.findall(result_pattern, extracted_content, flags=re.DOTALL)
                if result_matches:
                    extracted_content = result_matches[0].strip()

                if extracted_content:
                    content = extracted_content
            else:
                # Remove other tool usage blocks (when tools are disabled)
                tool_patterns = [
                    r"<read_file>.*?</read_file>",
                    r"<write_file>.*?</write_file>",
                    r"<bash>.*?</bash>",
                    r"<search_files>.*?</search_files>",
                    r"<str_replace_editor>.*?</str_replace_editor>",
                    r"<args>.*?</args>",
                    r"<ask_followup_question>.*?</ask_followup_question>",
                    r"<attempt_completion>.*?</attempt_completion>",
                    r"<question>.*?</question>",
                    r"<follow_up>.*?</follow_up>",
                    r"<suggest>.*?</suggest>",
                ]

                for pattern in tool_patterns:
                    content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Pattern to match image references or base64 data
        image_pattern = r"\[Image:.*?\]|data:image/.*?;base64,.*?(?=\s|$)"

        def replace_image(match):
            return "[Image: Content not supported by Claude Code]"

        content = re.sub(image_pattern, replace_image, content)

        # Clean up extra whitespace and newlines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Multiple newlines to double
        content = content.strip()

        # If content is now empty or only whitespace, provide a fallback
        if not content or content.isspace():
            return "I understand you're testing the system. How can I help you today?"

        return content

    @staticmethod
    def format_claude_response(
        content: str, model: str, finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Format Claude response for OpenAI compatibility."""
        return {
            "role": "assistant",
            "content": content,
            "finish_reason": finish_reason,
            "model": model,
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count.
        OpenAI's rule of thumb: ~4 characters per token for English text.
        """
        return len(text) // 4

    @staticmethod
    def openai_tools_to_anthropic(tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert OpenAI ToolDefinition list to Anthropic tool definition format.

        Args:
            tools: List of ToolDefinition Pydantic models (OpenAI format).

        Returns:
            List of dicts in Anthropic tool definition format.
        """
        result = []
        for tool_def in tools:
            func = tool_def.function
            anthropic_tool = {
                "name": func.name,
                "description": func.description or f"Tool: {func.name}",
                "input_schema": func.parameters or {"type": "object", "properties": {}},
            }
            result.append(anthropic_tool)
        return result

    @staticmethod
    def anthropic_tool_use_to_openai(tool_use_blocks: List[Any]) -> List[Dict[str, Any]]:
        """Convert ToolUsePart blocks to OpenAI ToolCall format.

        Args:
            tool_use_blocks: List of ToolUsePart dataclass instances.

        Returns:
            List of dicts in OpenAI tool_calls format.
        """
        result = []
        for block in tool_use_blocks:
            result.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        return result
