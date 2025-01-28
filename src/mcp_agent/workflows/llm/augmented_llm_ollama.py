from typing import Type

from openai import OpenAI

from mnemo_agent.workflows.llm.augmented_llm import (
    ModelT,
    RequestParams,
)
from mnemo_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class OllamaAugmentedLLM(OpenAIAugmentedLLM):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses Ollama's OpenAI-compatible ChatCompletion API.
    """

    def __init__(self, *args, **kwargs):
        # Create a copy of kwargs to avoid modifying the original
        updated_kwargs = kwargs.copy()

        # Only set default_model if it's not already in kwargs
        if "default_model" not in updated_kwargs:
            updated_kwargs["default_model"] = "llama3.2:3b"

        super().__init__(*args, **updated_kwargs)

        self.provider = "Ollama"

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        # First we invoke the LLM to generate a string response
        # We need to do this in a two-step process because Instructor doesn't
        # know how to invoke MCP tools via call_tool, so we'll handle all the
        # processing first and then pass the final response through Instructor
        import instructor

        response = await self.generate_str(
            message=message,
            request_params=request_params,
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_openai(
            OpenAI(
                api_key=self.context.config.openai.api_key,
                base_url=self.context.config.openai.base_url,
            ),
            mode=instructor.Mode.JSON,
        )

        params = self.get_request_params(request_params)
        model = await self.select_model(params)

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model or "llama3.2:3b",
            response_model=response_model,
            messages=[
                {"role": "user", "content": response},
            ],
        )

        return structured_response
