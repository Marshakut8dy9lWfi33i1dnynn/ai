"""
Keep track of all workflow decorator overloads indexed by executor backend.
Different executors may have different ways of configuring workflows.
"""

from typing import Callable, Dict, Type, TypeVar

R = TypeVar("R")


class DecoratorRegistry:
    """Centralized decorator management with validation and metadata."""

    def __init__(self):
        self._workflow_defn_decorators: Dict[str, Callable[[Type], Type]] = {}
        self._workflow_run_decorators: Dict[
            str, Callable[[Callable[..., R]], Callable[..., R]]
        ] = {}

    def register_workflow_defn_decorator(
        self,
        executor_name: str,
        decorator: Callable[[Type], Type],
    ):
        """
        Registers a workflow definition decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :param decorator: The decorator to register.
        """
        if executor_name in self._workflow_defn_decorators:
            print(
                "Workflow definition decorator already registered for '%s'. Overwriting.",
                executor_name,
            )
        self._workflow_defn_decorators[executor_name] = decorator

    def get_workflow_defn_decorator(self, executor_name: str) -> Callable[[Type], Type]:
        """
        Retrieves a workflow definition decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :return: The decorator function.
        """
        return self._workflow_defn_decorators.get(executor_name)

    def register_workflow_run_decorator(
        self,
        executor_name: str,
        decorator: Callable[[Callable[..., R]], Callable[..., R]],
    ):
        """
        Registers a workflow run decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :param decorator: The decorator to register.
        """
        if executor_name in self._workflow_run_decorators:
            print(
                "Workflow run decorator already registered for '%s'. Overwriting.",
                executor_name,
            )
        self._workflow_run_decorators[executor_name] = decorator

    def get_workflow_run_decorator(
        self, executor_name: str
    ) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """
        Retrieves a workflow run decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :return: The decorator function.
        """
        return self._workflow_run_decorators.get(executor_name)


def default_workflow_defn(cls: Type, *args, **kwargs) -> Type:
    """Default no-op workflow definition decorator."""
    return cls


def default_workflow_run(fn: Callable[..., R]) -> Callable[..., R]:
    """Default no-op workflow run decorator."""

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def register_asyncio_decorators(decorator_registry: DecoratorRegistry):
    """Registers default asyncio decorators."""
    executor_name = "asyncio"
    decorator_registry.register_workflow_defn_decorator(
        executor_name, default_workflow_defn
    )
    decorator_registry.register_workflow_run_decorator(
        executor_name, default_workflow_run
    )


def register_temporal_decorators(decorator_registry: DecoratorRegistry):
    """Registers Temporal decorators if Temporal SDK is available."""
    try:
        import temporalio.workflow as temporal_workflow

        TEMPORAL_AVAILABLE = True
    except ImportError:
        TEMPORAL_AVAILABLE = False

    if not TEMPORAL_AVAILABLE:
        return

    executor_name = "temporal"
    decorator_registry.register_workflow_defn_decorator(
        executor_name, temporal_workflow.defn
    )
    decorator_registry.register_workflow_run_decorator(
        executor_name, temporal_workflow.run
    )
