import asyncio
from rich import print

from mnemo_agent.app import MCPApp
from mnemo_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mnemo_agent.workflows.intent_classifier.intent_classifier_llm_openai import (
    OpenAILLMIntentClassifier,
)
from mnemo_agent.workflows.intent_classifier.intent_classifier_embedding_openai import (
    OpenAIEmbeddingIntentClassifier,
)

app = MCPApp(name="intent_classifier")


async def example_usage():
    async with app.run() as intent_app:
        logger = intent_app.logger
        context = intent_app.context
        logger.info("Current config:", data=context.config.model_dump())

        embedding_intent_classifier = OpenAIEmbeddingIntentClassifier(
            intents=[
                Intent(
                    name="greeting",
                    description="A friendly greeting",
                    examples=["Hello", "Hi there", "Good morning"],
                ),
                Intent(
                    name="farewell",
                    description="A friendly farewell",
                    examples=["Goodbye", "See you later", "Take care"],
                ),
            ],
            context=context,
        )

        results = await embedding_intent_classifier.classify(
            request="Hello, how are you?",
            top_k=1,
        )

        logger.info("LLM-based Intent classification results:", data=results)

        llm_intent_classifier = OpenAILLMIntentClassifier(
            intents=[
                Intent(
                    name="greeting",
                    description="A friendly greeting",
                    examples=["Hello", "Hi there", "Good morning"],
                ),
                Intent(
                    name="farewell",
                    description="A friendly farewell",
                    examples=["Goodbye", "See you later", "Take care"],
                ),
            ],
            context=context,
        )

        results = await llm_intent_classifier.classify(
            request="Hello, how are you?",
            top_k=1,
        )

        logger.info("LLM-based Intent classification results:", data=results)


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
