"""
This module contains the tools catalog for the MNEMO Agentic.
"""
from typing import List
from functools import lru_cache
from datetime import date
import requests

from pydantic import Field

from .types import LLMRole
from .utils import get_llm

req_session = requests.Session()

get_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

def get_current_date() -> str:
    """
    Returns: the current date.
    """
    return date.today().strftime("%A, %B %d, %Y")

#
# Standard Tools
#
@lru_cache(maxsize=None)
def summarize_text(
    text: str = Field(description="the original text."),
    expertise: str = Field(
        description="the expertise to apply to the summarization.",
    ),
) -> str:
    """
    This is a helper tool.
    Use this tool to summarize text using a given expertise
    with no more than summary_max_length characters.

    Args:
        text (str): The original text.
        expertise (str): The expertise to apply to the summarization.

    Returns:
        str: The summarized text.
    """
    if not isinstance(expertise, str):
        return "Please provide a valid string for expertise."
    if not isinstance(text, str):
        return "Please provide a valid string for text."
    expertise = "general" if len(expertise) < 3 else expertise.lower()
    prompt = f"As an expert in {expertise}, summarize the provided text"
    prompt += " into a concise summary."
    prompt += f"\noriginal text: {text}\nsummary:"
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


@lru_cache(maxsize=None)
def rephrase_text(
    text: str = Field(description="the original text."),
    instructions: str = Field(description="the specific instructions for how to rephrase the text."),
) -> str:
    """
    This is a helper tool.
    Use this tool to rephrase the text according to the provided instructions.
    For example, instructions could be "as a 5 year old would say it."

    Args:
        text (str): The original text.
        instructions (str): The specific instructions for how to rephrase the text.

    Returns:
        str: The rephrased text.
    """
    prompt = f"""
    Rephrase the provided text according to the following instructions: {instructions}.
    If the input is Markdown, keep the output in Markdown as well.
    original text: {text}
    rephrased text:
    """
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


@lru_cache(maxsize=None)
def critique_text(
    text: str = Field(description="the original text."),
    role: str = Field(default=None, description="the role of the person providing critique."),
    point_of_view: str = Field(default=None, description="the point of view with which to provide critique."),
) -> str:
    """
    This is a helper tool.
    Critique the text from the specified point of view.

    Args:
        text (str): The original text.
        role (str): The role of the person providing critique.
        point_of_view (str): The point of view with which to provide critique.

    Returns:
        str: The critique of the text.
    """
    if role:
        prompt = f"As a {role}, critique the provided text from the point of view of {point_of_view}."
    else:
        prompt = f"Critique the provided text from the point of view of {point_of_view}."
    prompt += "Structure the critique as bullet points.\n"
    prompt += f"Original text: {text}\nCritique:"
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


#
# Guardrails tool: returns list of topics to avoid
#
def get_bad_topics() -> List[str]:
    """
    Get the list of topics to avoid in the response.
    """
    return [
        "politics",
        "religion",
        "violence",
        "hate speech",
        "adult content",
        "illegal activities",
    ]
