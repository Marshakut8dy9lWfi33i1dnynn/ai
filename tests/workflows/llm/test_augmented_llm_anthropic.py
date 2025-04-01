from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from mnemo_agent.workflows.llm.augmented_llm_anthropic import (
    AnthropicAugmentedLLM,
    RequestParams,
)


class TestAnthropicAugmentedLLM:
    """
    Tests for the AnthropicAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self):
        """
        Creates a mock LLM instance with common mocks set up.
        """
        # Setup mock objects
        mock_context = MagicMock()
        mock_context.config.anthropic = MagicMock()
        mock_context.config.anthropic.default_model = "claude-3-7-sonnet-latest"
        mock_context.config.anthropic.api_key = "test_key"

        # Create LLM instance
        llm = AnthropicAugmentedLLM(name="test", context=mock_context)

        # Setup common mocks
        llm.aggregator = MagicMock()
        llm.aggregator.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value="claude-3-7-sonnet-latest")
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()

        return llm

    @pytest.fixture
    def default_usage(self):
        """
        Returns a default usage object for testing.
        """
        return Usage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=2789,
            output_tokens=89,
        )

    @staticmethod
    def create_tool_use_message(call_count, usage):
        """
        Creates a tool use message for testing.
        """
        return Message(
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    name="search_tool",
                    input={"query": "test query"},
                    id=f"tool_{call_count}",
                )
            ],
            model="claude-3-7-sonnet-latest",
            stop_reason="tool_use",
            id=f"resp_{call_count}",
            type="message",
            usage=usage,
        )

    @staticmethod
    def create_text_message(text, usage, role="assistant"):
        """
        Creates a text message for testing.
        """
        return Message(
            role=role,
            content=[TextBlock(type="text", text=text)],
            model="claude-3-7-sonnet-latest",
            stop_reason="end_turn",
            id="final_response",
            type="message",
            usage=usage,
        )

    @staticmethod
    def check_final_iteration_prompt_in_messages(messages):
        """
        Checks if there's a final iteration prompt in the given messages.
        """
        for msg in messages:
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and "please stop using tools" in msg.get("content", "").lower()
            ):
                return True
        return False

    def create_tool_use_side_effect(self, max_iterations, default_usage):
        """
        Creates a side effect function for tool use testing.
        """
        call_count = 0

        async def side_effect(fn, **kwargs):
            nonlocal call_count
            call_count += 1

            messages = kwargs.get("messages", [])
            has_final_iteration_prompt = self.check_final_iteration_prompt_in_messages(
                messages
            )

            if has_final_iteration_prompt:
                return [
                    self.create_text_message(
                        "Here is my final answer based on all the tool results gathered so far...",
                        default_usage,
                    )
                ]
            else:
                return [self.create_tool_use_message(call_count, default_usage)]

        return side_effect

    @pytest.mark.asyncio
    async def test_final_response_after_max_iterations_with_tool_use(
        self, mock_llm, default_usage
    ):
        """
        Tests whether we get a final text response when reaching max_iterations with tool_use.
        """
        # Setup executor with side effect
        mock_llm.executor = MagicMock()
        mock_llm.executor.execute = AsyncMock(
            side_effect=self.create_tool_use_side_effect(3, default_usage)
        )

        # Setup tool call mock
        mock_llm.call_tool = AsyncMock(
            return_value=MagicMock(content="Tool result", isError=False)
        )

        # Call LLM with max_iterations=3
        request_params = RequestParams(
            model="claude-3-7-sonnet-latest",
            maxTokens=1000,
            max_iterations=3,
            use_history=True,
        )

        responses = await mock_llm.generate("Test query", request_params)

        # Assertions
        # 1. Verify the last response is a text response
        assert responses[-1].stop_reason == "end_turn"
        assert responses[-1].content[0].type == "text"
        assert "final answer" in responses[-1].content[0].text.lower()

        # 2. Verify execute was called the expected number of times
        assert mock_llm.executor.execute.call_count == request_params.max_iterations

        # 3. Verify final prompt was added before the last request
        calls = mock_llm.executor.execute.call_args_list
        final_call_args = calls[-1][1]  # Arguments of the last call
        messages = final_call_args["messages"]

        # Check for the presence of the final answer request message
        assert self.check_final_iteration_prompt_in_messages(messages), (
            "No message requesting to stop using tools was found"
        )
