#!/usr/bin/env python3
"""
Integration tests for structured outputs (response_format) support.

Tests the system prompt injection flow for json_object and json_schema modes.
These are unit-level integration tests that don't require a running server.
"""

import pytest

from src.models import (
    Message,
    ChatCompletionRequest,
    ResponseFormat,
    ResponseFormatJsonSchema,
)
from src.message_adapter import MessageAdapter


class TestStructuredOutputSystemPromptInjection:
    """Test that structured output instructions are correctly injected into system prompts."""

    def _build_system_prompt(self, request: ChatCompletionRequest) -> str:
        """Simulate the system prompt building logic from main.py endpoints."""
        all_messages = request.messages
        prompt, system_prompt = MessageAdapter.messages_to_prompt(all_messages)

        sampling_instructions = request.get_sampling_instructions()
        if sampling_instructions:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{sampling_instructions}"
            else:
                system_prompt = sampling_instructions

        structured_instructions = request.get_structured_output_instructions()
        if structured_instructions:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{structured_instructions}"
            else:
                system_prompt = structured_instructions

        return system_prompt or ""

    def test_no_response_format_no_injection(self):
        """Without response_format, no structured instructions are injected."""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="Hello")],
        )
        system_prompt = self._build_system_prompt(request)
        assert "valid JSON" not in system_prompt

    def test_json_object_injects_json_instruction(self):
        """json_object mode injects JSON-only instruction into system prompt."""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="List colors as json")],
            response_format=ResponseFormat(type="json_object"),
        )
        system_prompt = self._build_system_prompt(request)
        assert "valid JSON" in system_prompt
        assert "single valid JSON object" in system_prompt

    def test_json_schema_injects_schema_in_system_prompt(self):
        """json_schema mode injects the full schema into system prompt."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population": {"type": "integer"},
            },
            "required": ["name", "population"],
        }
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="Extract info about Tokyo")],
            response_format=ResponseFormat(
                type="json_schema",
                json_schema=ResponseFormatJsonSchema(
                    name="city_info",
                    **{"schema": schema},
                ),
            ),
        )
        system_prompt = self._build_system_prompt(request)
        assert '"city_info"' in system_prompt
        assert '"population"' in system_prompt
        assert "required" in system_prompt

    def test_structured_output_appended_after_user_system_prompt(self):
        """Structured output instructions append after the user's system prompt."""
        request = ChatCompletionRequest(
            messages=[
                Message(role="system", content="You are a helpful assistant"),
                Message(role="user", content="List colors as json"),
            ],
            response_format=ResponseFormat(type="json_object"),
        )
        system_prompt = self._build_system_prompt(request)
        # User system prompt comes first
        assert system_prompt.index("helpful assistant") < system_prompt.index("valid JSON")

    def test_text_format_no_injection(self):
        """type='text' does not inject any instructions."""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="Hello")],
            response_format=ResponseFormat(type="text"),
        )
        system_prompt = self._build_system_prompt(request)
        assert "valid JSON" not in system_prompt
