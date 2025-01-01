"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

from datetime import timedelta
from typing import Any, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.shared.session import (
    ReceiveResultT,
    ReceiveNotificationT,
    RequestId,
    SendNotificationT,
    SendRequestT,
    SendResultT,
)

from mcp.shared.context import RequestContext

from mcp.client.session import (
    ListRootsFnT,
    LoggingFnT,
    MessageHandlerFnT,
    SamplingFnT,
)

from mcp.types import (
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    Implementation,
    JSONRPCMessage,
    ServerRequest,
    TextContent,
    ListRootsResult,
    Root,
)

from mnemo_agent.config import MCPServerSettings
from mnemo_agent.context_dependent import ContextDependent
from mnemo_agent.logging.logger import get_logger

logger = get_logger(__name__)


class MCPAgentClientSession(ClientSession, ContextDependent):
    """
    MCP Agent framework acts as a client to the servers providing tools/resources/prompts for the agent workloads.
    This is a simple client session for those server connections, and supports
        - handling sampling requests
        - notifications
        - MCP root configuration

    Developers can extend this class to add more custom functionality as needed
    """

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[JSONRPCMessage | Exception],
        write_stream: MemoryObjectSendStream[JSONRPCMessage],
        read_timeout_seconds: timedelta | None = None,
        sampling_callback: SamplingFnT | None = None,
        list_roots_callback: ListRootsFnT | None = None,
        logging_callback: LoggingFnT | None = None,
        message_handler: MessageHandlerFnT | None = None,
        client_info: Implementation | None = None,
    ):
        if sampling_callback is None:
            sampling_callback = self._handle_sampling_callback
        if list_roots_callback is None:
            list_roots_callback = self._handle_list_roots_callback

        super().__init__(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=read_timeout_seconds,
            sampling_callback=sampling_callback,
            list_roots_callback=list_roots_callback,
            logging_callback=logging_callback,
            message_handler=message_handler,
            client_info=client_info,
        )
        self.server_config: Optional[MCPServerSettings] = None

    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
        request_read_timeout_seconds: timedelta | None = None,
    ) -> ReceiveResultT:
        logger.debug("send_request: request=", data=request.model_dump())
        try:
            result = await super().send_request(
                request, result_type, request_read_timeout_seconds
            )
            logger.debug("send_request: response=", data=result.model_dump())
            return result
        except Exception as e:
            logger.error(f"send_request failed: {e}")
            raise

    async def send_notification(self, notification: SendNotificationT) -> None:
        logger.debug("send_notification:", data=notification.model_dump())
        try:
            return await super().send_notification(notification)
        except Exception as e:
            logger.error("send_notification failed", data=e)
            raise

    async def _send_response(
        self, request_id: RequestId, response: SendResultT | ErrorData
    ) -> None:
        logger.debug(
            f"send_response: request_id={request_id}, response=",
            data=response.model_dump(),
        )
        return await super()._send_response(request_id, response)

    async def _received_notification(self, notification: ReceiveNotificationT) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.info(
            "_received_notification: notification=",
            data=notification.model_dump(),
        )
        return await super()._received_notification(notification)

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Sends a progress notification for a request that is currently being
        processed.
        """
        logger.debug(
            "send_progress_notification: progress_token={progress_token}, progress={progress}, total={total}"
        )
        return await super().send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )

    async def _handle_sampling_callback(
        self,
        context: RequestContext["ClientSession", Any],
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        logger.info("Handling sampling request: %s", params)
        config = self.context.config
        server_session = self.context.upstream_session
        if server_session is None:
            # TODO: saqadri - consider whether we should be handling the sampling request here as a client
            logger.warning(
                "Error: No upstream client available for sampling requests. Request:",
                data=params,
            )
            try:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic(api_key=config.anthropic.api_key)

                response = await client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=params.maxTokens,
                    messages=[
                        {
                            "role": m.role,
                            "content": m.content.text
                            if hasattr(m.content, "text")
                            else m.content.data,
                        }
                        for m in params.messages
                    ],
                    system=getattr(params, "systemPrompt", None),
                    temperature=getattr(params, "temperature", 0.7),
                    stop_sequences=getattr(params, "stopSequences", None),
                )

                return CreateMessageResult(
                    model="claude-3-sonnet-20240229",
                    role="assistant",
                    content=TextContent(type="text", text=response.content[0].text),
                )
            except Exception as e:
                logger.error(f"Error handling sampling request: {e}")
                return ErrorData(code=-32603, message=str(e))
        else:
            try:
                # If a server_session is available, we'll pass-through the sampling request to the upstream client
                result = await server_session.send_request(
                    request=ServerRequest(
                        CreateMessageRequest(
                            method="sampling/createMessage", params=params
                        )
                    ),
                    result_type=CreateMessageResult,
                )

                # Pass the result from the upstream client back to the server. We just act as a pass-through client here.
                return result
            except Exception as e:
                return ErrorData(code=-32603, message=str(e))

    async def _handle_list_roots_callback(
        self,
        context: RequestContext["ClientSession", Any],
    ) -> ListRootsResult | ErrorData:
        # Handle list_roots request by returning configured roots
        if hasattr(self, "server_config") and self.server_config.roots:
            roots = [
                Root(
                    uri=root.server_uri_alias or root.uri,
                    name=root.name,
                )
                for root in self.server_config.roots
            ]

            return ListRootsResult(roots=roots)
        else:
            return ListRootsResult(roots=[])
