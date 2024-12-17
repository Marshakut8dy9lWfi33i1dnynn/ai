"""
This module contains functions to start the agent behind an API endpoint.
"""
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from .agent import Agent
from .agent_config import AgentConfig

api_key_header = APIKeyHeader(name="X-API-Key")

class ChatRequest(BaseModel):
    """
    A request model for the chat endpoint.
    """
    message: str


def create_app(agent: Agent, config: AgentConfig) -> FastAPI:
    """
    Create a FastAPI application with a chat endpoint.
    """
    app = FastAPI()
    logger = logging.getLogger("uvicorn.error")
    logging.basicConfig(level=logging.INFO)
    endpoint_api_key = config.endpoint_api_key

    @app.get("/chat", summary="Chat with the agent")
    async def chat(message: str, api_key: str = Depends(api_key_header)):
        logger.info(f"Received message: {message}")
        if api_key != endpoint_api_key:
            logger.warning("Unauthorized access attempt")
            raise HTTPException(status_code=403, detail="Unauthorized")

        if not message:
            logger.error("No message provided in the request")
            raise HTTPException(status_code=400, detail="No message provided")

        try:
            response = agent.chat(message)
            logger.info(f"Generated response: {response}")
            return {"response": response}
        except Exception as e:
            logger.error(f"Error during agent processing: {e}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return app


def start_app(agent: Agent, host='0.0.0.0', port=8000):
    """
    Start the FastAPI server.

    Args:
        host (str, optional): The host address for the API. Defaults to '127.0.0.1'.
        port (int, optional): The port for the API. Defaults to 8000.
    """
    app = create_app(agent, config=AgentConfig())
    uvicorn.run(app, host=host, port=port)
