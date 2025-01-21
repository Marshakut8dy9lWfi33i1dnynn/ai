"""
Temporal based orchestrator for the MCP Agent.
Temporal provides durable execution and robust workflow orchestration,
as well as dynamic control flow, making it a good choice for an AI agent orchestrator.
Read more: https://docs.temporal.io/develop/python/core-application
"""

import asyncio
import functools
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from pydantic import ConfigDict
from temporalio import activity, workflow, exceptions
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker

from mnemo_agent.config import TemporalSettings
from mnemo_agent.executor.executor import Executor, ExecutorConfig, R
from mnemo_agent.executor.workflow_signal import (
    BaseSignalHandler,
    Signal,
    SignalHandler,
    SignalRegistration,
    SignalValueT,
)

if TYPE_CHECKING:
    from mnemo_agent.context import Context


class TemporalSignalHandler(BaseSignalHandler[SignalValueT]):
    """Temporal-based signal handling using workflow signals"""

    async def wait_for_signal(self, signal, timeout_seconds=None) -> SignalValueT:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalSignalHandler.wait_for_signal must be called from within a workflow"
            )

        unique_signal_name = f"{signal.name}_{uuid.uuid4()}"
        registration = SignalRegistration(
            signal_name=signal.name,
            unique_name=unique_signal_name,
            workflow_id=workflow.info().workflow_id,
        )

        # Container for signal value
        container = {"value": None, "completed": False}

        # Define the signal handler for this specific registration
        @workflow.signal(name=unique_signal_name)
        def signal_handler(value: SignalValueT):
            container["value"] = value
            container["completed"] = True

        async with self._lock:
            # Register both the signal registration and handler atomically
            self._pending_signals.setdefault(signal.name, []).append(registration)
            self._handlers.setdefault(signal.name, []).append(
                (unique_signal_name, signal_handler)
            )

        try:
            # Wait for signal with optional timeout
            await workflow.wait_condition(
                lambda: container["completed"], timeout=timeout_seconds
            )

            return container["value"]
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timeout waiting for signal {signal.name}") from exc
        finally:
            async with self._lock:
                # Remove ourselves from _pending_signals
                if signal.name in self._pending_signals:
                    self._pending_signals[signal.name] = [
                        sr
                        for sr in self._pending_signals[signal.name]
                        if sr.unique_name != unique_signal_name
                    ]
                    if not self._pending_signals[signal.name]:
                        del self._pending_signals[signal.name]

                # Remove ourselves from _handlers
                if signal.name in self._handlers:
                    self._handlers[signal.name] = [
                        h
                        for h in self._handlers[signal.name]
                        if h[0] != unique_signal_name
                    ]
                    if not self._handlers[signal.name]:
                        del self._handlers[signal.name]

    def on_signal(self, signal_name):
        """Decorator to register a signal handler."""

        def decorator(func: Callable) -> Callable:
            # Create unique signal name for this handler
            unique_signal_name = f"{signal_name}_{uuid.uuid4()}"

            # Create the actual handler that will be registered with Temporal
            @workflow.signal(name=unique_signal_name)
            async def wrapped(signal_value: SignalValueT):
                # Create a signal object to pass to the handler
                signal = Signal(
                    name=signal_name,
                    payload=signal_value,
                    workflow_id=workflow.info().workflow_id,
                )
                if asyncio.iscoroutinefunction(func):
                    await func(signal)
                else:
                    func(signal)

            # Register the handler under the original signal name
            self._handlers.setdefault(signal_name, []).append(
                (unique_signal_name, wrapped)
            )
            return func

        return decorator

    async def signal(self, signal):
        self.validate_signal(signal)

        workflow_handle = workflow.get_external_workflow_handle(
            workflow_id=signal.workflow_id
        )

        # Send the signal to all registrations of this signal
        async with self._lock:
            signal_tasks = []

            if signal.name in self._pending_signals:
                for pending_signal in self._pending_signals[signal.name]:
                    registration = pending_signal.registration
                    if registration.workflow_id == signal.workflow_id:
                        # Only signal for registrations of that workflow
                        signal_tasks.append(
                            workflow_handle.signal(
                                registration.unique_name, signal.payload
                            )
                        )
                    else:
                        continue

            # Notify any registered handler functions
            if signal.name in self._handlers:
                for unique_name, _ in self._handlers[signal.name]:
                    signal_tasks.append(
                        workflow_handle.signal(unique_name, signal.payload)
                    )

        await asyncio.gather(*signal_tasks, return_exceptions=True)

    def validate_signal(self, signal):
        super().validate_signal(signal)
        # Add TemporalSignalHandler-specific validation
        if signal.workflow_id is None:
            raise ValueError(
                "No workflow_id provided on Signal. That is required for Temporal signals"
            )


class TemporalExecutorConfig(ExecutorConfig, TemporalSettings):
    """Configuration for Temporal executors."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class TemporalExecutor(Executor):
    """Executor that runs @workflows as Temporal workflows, with @workflow_tasks as Temporal activities"""

    def __init__(
        self,
        config: TemporalExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
        client: TemporalClient | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        signal_bus = signal_bus or TemporalSignalHandler()
        super().__init__(
            engine="temporal",
            config=config,
            signal_bus=signal_bus,
            context=context,
            **kwargs,
        )
        self.config: TemporalExecutorConfig = (
            config or self.context.config.temporal or TemporalExecutorConfig()
        )
        self.client = client
        self._worker = None
        self._activity_semaphore = None

        if config.max_concurrent_activities is not None:
            self._activity_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_activities
            )

    @staticmethod
    def wrap_as_activity(
        activity_name: str,
        func: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> Coroutine[Any, Any, R]:
        """
        Convert a function into a Temporal activity and return its info.
        """

        @activity.defn(name=activity_name)
        async def wrapped_activity(*args, **local_kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **local_kwargs)
                elif asyncio.iscoroutine(func):
                    return await func
                else:
                    return func(*args, **local_kwargs)
            except Exception as e:
                # Handle exceptions gracefully
                raise e

        return wrapped_activity

    async def _execute_task_as_async(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], **kwargs: Any
    ) -> R | BaseException:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            try:
                if asyncio.iscoroutine(task):
                    return await task
                elif asyncio.iscoroutinefunction(task):
                    return await task(**kwargs)
                else:
                    # Execute the callable and await if it returns a coroutine
                    loop = asyncio.get_running_loop()

                    # If kwargs are provided, wrap the function with partial
                    if kwargs:
                        wrapped_task = functools.partial(task, **kwargs)
                        result = await loop.run_in_executor(None, wrapped_task)
                    else:
                        result = await loop.run_in_executor(None, task)

                    # Handle case where the sync function returns a coroutine
                    if asyncio.iscoroutine(result):
                        return await result

                    return result
            except Exception as e:
                # TODO: saqadri - adding logging or other error handling here
                return e

        if self._activity_semaphore:
            async with self._activity_semaphore:
                return await run_task(task)
        else:
            return await run_task(task)

    async def _execute_task(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], **kwargs: Any
    ) -> R | BaseException:
        func = task.func if isinstance(task, functools.partial) else task
        is_workflow_task = getattr(func, "is_workflow_task", False)
        if not is_workflow_task:
            return await asyncio.create_task(
                self._execute_task_as_async(task, **kwargs)
            )

        execution_metadata: Dict[str, Any] = getattr(func, "execution_metadata", {})

        # Derive stable activity name, e.g. module + qualname
        activity_name = execution_metadata.get("activity_name")
        if not activity_name:
            activity_name = f"{func.__module__}.{func.__qualname__}"

        schedule_to_close = execution_metadata.get(
            "schedule_to_close_timeout", self.config.timeout_seconds
        )

        retry_policy = execution_metadata.get("retry_policy", None)

        _task_activity = self.wrap_as_activity(activity_name=activity_name, func=task)

        # # For partials, we pass the partial's arguments
        # args = task.args if isinstance(task, functools.partial) else ()
        try:
            result = await workflow.execute_activity(
                activity_name,
                args=kwargs.get("args", ()),
                task_queue=self.config.task_queue,
                schedule_to_close_timeout=schedule_to_close,
                retry_policy=retry_policy,
                **kwargs,
            )
            return result
        except Exception as e:
            # Properly propagate activity errors
            if isinstance(e, exceptions.ActivityError):
                raise e.cause if e.cause else e
            raise

    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        # Must be called from within a workflow
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            return await asyncio.gather(
                *(self._execute_task(task, **kwargs) for task in tasks),
                return_exceptions=True,
            )

    async def execute_streaming(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute_streaming must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            # Create futures for all tasks
            futures = [self._execute_task(task, **kwargs) for task in tasks]
            pending = set(futures)

            while pending:
                done, pending = await workflow.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        result = await future
                        yield result
                    except Exception as e:
                        yield e

    async def ensure_client(self):
        """Ensure we have a connected Temporal client."""
        if self.client is None:
            self.client = await TemporalClient.connect(
                target_host=self.config.host,
                namespace=self.config.namespace,
                api_key=self.config.api_key,
            )

        return self.client

    async def start_worker(self):
        """
        Start a worker in this process, auto-registering all tasks
        from the global registry. Also picks up any classes decorated
        with @workflow_defn as recognized workflows.
        """
        await self.ensure_client()

        if self._worker is None:
            # We'll collect the activities from the global registry
            # and optionally wrap them with `activity.defn` if we want
            # (Not strictly required if your code calls `execute_activity("name")` by name)
            activity_registry = self.context.task_registry
            activities = []
            for name in activity_registry.list_activities():
                activities.append(activity_registry.get_activity(name))

            # Now we attempt to discover any classes that are recognized as workflows
            # But in this simple example, we rely on the user specifying them or
            # we might do a dynamic scan.
            # For demonstration, we'll just assume the user is only using
            # the workflow classes they decorate with `@workflow_defn`.
            # We'll rely on them passing the classes or scanning your code.

            self._worker = Worker(
                client=self.client,
                task_queue=self.config.task_queue,
                activities=activities,
                workflows=[],  # We'll auto-load by Python scanning or let the user specify
            )
            print(
                f"Starting Temporal Worker on task queue '{self.config.task_queue}' with {len(activities)} activities."
            )

        await self._worker.run()
