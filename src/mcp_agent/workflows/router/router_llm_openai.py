from typing import Callable, List, Optional, TYPE_CHECKING

from mnemo_agent.agents.agent import Agent
from mnemo_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mnemo_agent.workflows.router.router_llm import LLMRouter

if TYPE_CHECKING:
    from mnemo_agent.context import Context

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).
You will be provided with a request and a list of categories to choose from.
You can choose one or more categories, or choose none if no category is appropriate.
"""


class OpenAILLMRouter(LLMRouter):
    """
    An LLM router that uses an OpenAI model to make routing decisions.
    """

    def __init__(
        self,
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        openai_llm = OpenAIAugmentedLLM(
            instruction=ROUTING_SYSTEM_INSTRUCTION, context=context
        )

        super().__init__(
            llm=openai_llm,
            server_names=server_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
            **kwargs,
        )

    @classmethod
    async def create(
        cls,
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
    ) -> "OpenAILLMRouter":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            server_names=server_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
        )
        await instance.initialize()
        return instance
