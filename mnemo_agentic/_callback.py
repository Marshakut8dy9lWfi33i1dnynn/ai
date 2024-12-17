"""
Module to handle agent callbacks
"""

import inspect
from typing import Any, Dict, Optional, List, Callable

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from .types import AgentStatusType


class AgentCallbackHandler(BaseCallbackHandler):
    """
    Callback handler to track agent status

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of agent progress.

    Args:
        fn: callable function agent will call back to report on agent progress
    """

    def __init__(self, fn: Optional[Callable] = None) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.fn = fn

    # Existing synchronous methods
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if self.fn is not None and payload is not None:
            if inspect.iscoroutinefunction(self.fn):
                raise ValueError("Synchronous callback handler cannot use async callback function")
            # Handle events as before
            self._handle_event(event_type, payload)
        return event_id

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of an event

        Args:
            event_type: the type of event
            payload: the event payload
            event_id: the event ID
            kwargs: additional keyword arguments

        Returns:
            None
        """
        if self.fn is not None and payload is not None:
            if inspect.iscoroutinefunction(self.fn):
                raise ValueError("Synchronous callback handler cannot use async callback function")
            # Handle events as before
            self._handle_event(event_type, payload)

    # New asynchronous methods
    async def aon_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Handle the start of an event

        Args:
            event_type: the type of event
            payload: the event payload
            event_id: the event ID
            parent_id: the parent event ID
            kwargs: additional keyword arguments

        Returns:
            event_id: the event ID
        """
        if self.fn is not None and payload is not None:
            await self._ahandle_event(event_type, payload)
        return event_id

    async def aon_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of an event (async)
        """
        if self.fn is not None and payload is not None:
            await self._ahandle_event(event_type, payload)

    # Helper methods for handling events
    def _handle_event(self, event_type: CBEventType, payload: Dict[str, Any]) -> None:
        if event_type == CBEventType.LLM:
            self._handle_llm(payload)
        elif event_type == CBEventType.FUNCTION_CALL:
            self._handle_function_call(payload)
        elif event_type == CBEventType.AGENT_STEP:
            self._handle_agent_step(payload)
        elif event_type == CBEventType.EXCEPTION:
            print(f"Exception: {payload.get(EventPayload.EXCEPTION)}")
        else:
            print(f"Unknown event type: {event_type}, payload={payload}")

    async def _ahandle_event(self, event_type: CBEventType, payload: Dict[str, Any]) -> None:
        if event_type == CBEventType.LLM:
            await self._ahandle_llm(payload)
        elif event_type == CBEventType.FUNCTION_CALL:
            await self._ahandle_function_call(payload)
        elif event_type == CBEventType.AGENT_STEP:
            await self._ahandle_agent_step(payload)
        elif event_type == CBEventType.EXCEPTION:
            print(f"Exception: {payload.get(EventPayload.EXCEPTION)}")
        else:
            print(f"Unknown event type: {event_type}, payload={payload}")

    # Synchronous handlers
    def _handle_llm(self, payload: dict) -> None:
        if EventPayload.MESSAGES in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if response and response not in ["None", "assistant: None"]:
                self.fn(AgentStatusType.AGENT_UPDATE, response)
        else:
            print(f"No messages or prompt found in payload {payload}")

    def _handle_function_call(self, payload: dict) -> None:
        if EventPayload.FUNCTION_CALL in payload:
            fcall = str(payload.get(EventPayload.FUNCTION_CALL))
            tool = payload.get(EventPayload.TOOL)
            if tool:
                tool_name = tool.name
                self.fn(
                    AgentStatusType.TOOL_CALL,
                    f"Executing '{tool_name}' with arguments: {fcall}",
                )
        elif EventPayload.FUNCTION_OUTPUT in payload:
            response = str(payload.get(EventPayload.FUNCTION_OUTPUT))
            self.fn(AgentStatusType.TOOL_OUTPUT, response)
        else:
            print(f"No function call or output found in payload {payload}")

    def _handle_agent_step(self, payload: dict) -> None:
        if EventPayload.MESSAGES in payload:
            msg = str(payload.get(EventPayload.MESSAGES))
            self.fn(AgentStatusType.AGENT_STEP, msg)
        elif EventPayload.RESPONSE in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            self.fn(AgentStatusType.AGENT_STEP, response)
        else:
            print(f"No messages or prompt found in payload {payload}")

    # Asynchronous handlers
    async def _ahandle_llm(self, payload: dict) -> None:
        if EventPayload.MESSAGES in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if response and response not in ["None", "assistant: None"]:
                if inspect.iscoroutinefunction(self.fn):
                    await self.fn(AgentStatusType.AGENT_UPDATE, response)
                else:
                    self.fn(AgentStatusType.AGENT_UPDATE, response)
        else:
            print(f"No messages or prompt found in payload {payload}")

    async def _ahandle_function_call(self, payload: dict) -> None:
        if EventPayload.FUNCTION_CALL in payload:
            fcall = str(payload.get(EventPayload.FUNCTION_CALL))
            tool = payload.get(EventPayload.TOOL)
            if tool:
                tool_name = tool.name
                if inspect.iscoroutinefunction(self.fn):
                    await self.fn(
                        AgentStatusType.TOOL_CALL,
                        f"Executing '{tool_name}' with arguments: {fcall}",
                    )
                else:
                    self.fn(
                        AgentStatusType.TOOL_CALL,
                        f"Executing '{tool_name}' with arguments: {fcall}",
                    )
        elif EventPayload.FUNCTION_OUTPUT in payload:
            response = str(payload.get(EventPayload.FUNCTION_OUTPUT))
            if inspect.iscoroutinefunction(self.fn):
                await self.fn(AgentStatusType.TOOL_OUTPUT, response)
            else:
                self.fn(AgentStatusType.TOOL_OUTPUT, response)
        else:
            print(f"No function call or output found in payload {payload}")

    async def _ahandle_agent_step(self, payload: dict) -> None:
        if EventPayload.MESSAGES in payload:
            msg = str(payload.get(EventPayload.MESSAGES))
            if inspect.iscoroutinefunction(self.fn):
                await self.fn(AgentStatusType.AGENT_STEP, msg)
            else:
                self.fn(AgentStatusType.AGENT_STEP, msg)
        elif EventPayload.RESPONSE in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if inspect.iscoroutinefunction(self.fn):
                await self.fn(AgentStatusType.AGENT_STEP, response)
            else:
                self.fn(AgentStatusType.AGENT_STEP, response)
        else:
            print(f"No messages or prompt found in payload {payload}")
