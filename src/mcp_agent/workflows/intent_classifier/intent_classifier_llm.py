from typing import List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel

from mnemo_agent.workflows.llm.augmented_llm import AugmentedLLM
from mnemo_agent.workflows.intent_classifier.intent_classifier_base import (
    Intent,
    IntentClassifier,
    IntentClassificationResult,
)

if TYPE_CHECKING:
    from mnemo_agent.context import Context

DEFAULT_INTENT_CLASSIFICATION_INSTRUCTION = """
You are a precise intent classifier that analyzes user requests to determine their intended action or purpose.
Below are the available intents with their descriptions and examples:

{context}

Your task is to analyze the following request and determine the most likely intent(s). Consider:
- How well the request matches the intent descriptions and examples
- Any specific entities or parameters that should be extracted
- The confidence level in the classification

Request: {request}

Respond in JSON format:
{{
    "classifications": [
        {{
            "intent": <intent name>,
            "confidence": <float between 0 and 1>,
            "extracted_entities": {{
                "entity_name": "entity_value"
            }},
            "reasoning": <brief explanation>
        }}
    ]
}}

Return up to {top_k} most likely intents. Only include intents with reasonable confidence (>0.5).
If no intents match well, return an empty list.
"""


class LLMIntentClassificationResult(IntentClassificationResult):
    """The result of intent classification using an LLM."""

    confidence: Literal["low", "medium", "high"]
    """Confidence level of the classification"""

    reasoning: str | None = None
    """Optional explanation of why this intent was chosen"""


class StructuredIntentResponse(BaseModel):
    """The complete structured response from the LLM"""

    classifications: List[LLMIntentClassificationResult]


class LLMIntentClassifier(IntentClassifier):
    """
    An intent classifier that uses an LLM to determine the user's intent.
    Particularly useful when you need:
    - Flexible understanding of natural language
    - Detailed reasoning about classifications
    - Entity extraction alongside classification
    """

    def __init__(
        self,
        llm: AugmentedLLM,
        intents: List[Intent],
        classification_instruction: str | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        super().__init__(intents=intents, context=context, **kwargs)
        self.llm = llm
        self.classification_instruction = classification_instruction

    @classmethod
    async def create(
        cls,
        llm: AugmentedLLM,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ) -> "LLMIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            llm=llm,
            intents=intents,
            classification_instruction=classification_instruction,
        )
        await instance.initialize()
        return instance

    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[LLMIntentClassificationResult]:
        if not self.initialized:
            self.initialize()

        classification_instruction = (
            self.classification_instruction or DEFAULT_INTENT_CLASSIFICATION_INSTRUCTION
        )

        # Generate the context with intent descriptions and examples
        context = self._generate_context()

        # Format the prompt with all the necessary information
        prompt = classification_instruction.format(
            context=context, request=request, top_k=top_k
        )

        # Get classification from LLM
        response = await self.llm.generate_structured(
            message=prompt, response_model=StructuredIntentResponse
        )

        if not response or not response.classifications:
            return []

        results = []
        for classification in response.classifications:
            intent = self.intents.get(classification.intent)
            if not intent:
                # Skip invalid categories
                # TODO: saqadri - log or raise an error
                continue

            results.append(classification)

        return results[:top_k]

    def _generate_context(self) -> str:
        """Generate a formatted context string describing all intents"""
        context_parts = []

        for idx, intent in enumerate(self.intents.values(), 1):
            description = (
                f"{idx}. Intent: {intent.name}\nDescription: {intent.description}"
            )

            if intent.examples:
                examples = "\n".join(f"- {example}" for example in intent.examples)
                description += f"\nExamples:\n{examples}"

            if intent.metadata:
                metadata = "\n".join(
                    f"- {key}: {value}" for key, value in intent.metadata.items()
                )
                description += f"\nAdditional Information:\n{metadata}"

            context_parts.append(description)

        return "\n\n".join(context_parts)
