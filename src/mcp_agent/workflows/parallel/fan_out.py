import contextlib
import functools
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, TYPE_CHECKING

from mnemo_agent.agents.agent import Agent
from mnemo_agent.context_dependent import ContextDependent
from mnemo_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)
from mnemo_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mnemo_agent.context import Context

logger = get_logger(__name__)


class FanOut(ContextDependent):
    """
    Distribute work to multiple parallel tasks.

    This is a building block of the Parallel workflow, which can be used to fan out
    work to multiple agents or other parallel tasks, and then aggregate the results.
    """

    def __init__(
        self,
        agents: List[Agent | AugmentedLLM[MessageParamT, MessageT]] | None = None,
        functions: List[Callable[[MessageParamT], List[MessageT]]] | None = None,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]] = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        """
        Initialize the FanOut with a list of agents, functions, or LLMs.
        If agents are provided, they will be wrapped in an AugmentedLLM using llm_factory if not already done so.
        If functions are provided, they will be invoked in parallel directly.
        """
        super().__init__(context=context, **kwargs)
        self.executor = self.context.executor
        self.llm_factory = llm_factory
        self.agents = agents or []
        self.functions: List[Callable[[MessageParamT], MessageT]] = functions or []

        if not self.agents and not self.functions:
            raise ValueError(
                "At least one agent or function must be provided for fan-out to work"
            )

        if not self.llm_factory:
            for agent in self.agents:
                if not isinstance(agent, AugmentedLLM):
                    raise ValueError("llm_factory is required when using an Agent")

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> Dict[str, List[MessageT]]:
        """
        Request fan-out agent/function generations, and return the results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks: List[
            Callable[..., List[MessageT]] | Coroutine[Any, Any, List[MessageT]]
        ] = []
        task_names: List[str] = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate(
                        message=message,
                        request_params=request_params,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(function, message))
                task_names.append(function.__name__ or id(function))

            # Wait for all tasks to complete
            logger.debug("Running fan-out tasks:", data=task_names)
            task_results = await self.executor.execute(*tasks)

        logger.debug(
            "Fan-out tasks completed:", data=dict(zip(task_names, task_results))
        )
        return dict(zip(task_names, task_results))

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> Dict[str, str]:
        """
        Request fan-out agent/function generations and return the string results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """

        def fn_result_to_string(fn, message):
            return str(fn(message))

        tasks: List[Callable[..., str] | Coroutine[Any, Any, str]] = []
        task_names: List[str] = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate_str(
                        message=message,
                        request_params=request_params,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(fn_result_to_string, function, message))
                task_names.append(function.__name__ or id(function))

            task_results = await self.executor.execute(*tasks)

        return dict(zip(task_names, task_results))

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Dict[str, ModelT]:
        """
        Request a structured fan-out agent/function generation and return the result as a Pydantic model.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks = []
        task_names = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate_structured(
                        message=message,
                        response_model=response_model,
                        request_params=request_params,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(function, message))
                task_names.append(function.__name__ or id(function))

            task_results = await self.executor.execute(*tasks)

        return dict(zip(task_names, task_results))
