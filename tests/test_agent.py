import unittest
from datetime import date

from mnemo_agentic.agent import _get_prompt, Agent, AgentType, FunctionTool
from mnemo_agentic.agent_config import AgentConfig
from mnemo_agentic.types import ModelProvider, ObserverType

class TestAgentPackage(unittest.TestCase):
    def test_get_prompt(self):
        prompt_template = "{chat_topic} on {today} with {custom_instructions}"
        topic = "Programming"
        custom_instructions = "Always do as your mother tells you!"
        expected_output = (
            "Programming on "
            + date.today().strftime("%A, %B %d, %Y")
            + " with Always do as your mother tells you!"
        )
        self.assertEqual(
            _get_prompt(prompt_template, topic, custom_instructions), expected_output
        )

    def test_agent_init(self):
        def mult(x, y):
            return x * y

        tools = [
            FunctionTool.from_defaults(
                fn=mult, name="mult", description="Multiplication functions"
            )
        ]
        topic = "AI"
        custom_instructions = "Always do as your mother tells you!"
        agent = Agent(tools, topic, custom_instructions)
        self.assertEqual(agent.agent_type, AgentType.OPENAI)
        self.assertEqual(agent.tools, tools)
        self.assertEqual(agent._topic, topic)
        self.assertEqual(agent._custom_instructions, custom_instructions)

        # To run this test, you must have OPENAI_API_KEY in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).replace("$", "\\$"),
            "50",
        )

    def test_agent_config(self):
        def mult(x, y):
            return x * y

        tools = [
            FunctionTool.from_defaults(
                fn=mult, name="mult", description="Multiplication functions"
            )
        ]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        config = AgentConfig(
            agent_type=AgentType.REACT,
            main_llm_provider=ModelProvider.ANTHROPIC,
            main_llm_model_name="claude-3-5-sonnet-20241022",
            tool_llm_provider=ModelProvider.TOGETHER,
            tool_llm_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            observer=ObserverType.ARIZE_PHOENIX
        )

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=config
        )
        self.assertEqual(agent.tools, tools)
        self.assertEqual(agent._topic, topic)
        self.assertEqual(agent._custom_instructions, instructions)
        self.assertEqual(agent.agent_type, AgentType.REACT)
        self.assertEqual(agent.agent_config.observer, ObserverType.ARIZE_PHOENIX)
        self.assertEqual(agent.agent_config.main_llm_provider, ModelProvider.ANTHROPIC)
        self.assertEqual(agent.agent_config.tool_llm_provider, ModelProvider.TOGETHER)

        # To run this test, you must have OPENAI_API_KEY in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).replace("$", "\\$"),
            "50",
        )

    def test_from_corpus(self):
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            mnemo_customer_id="4584783",
            mnemo_corpus_id="4",
            mnemo_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
        )

        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent._topic, "question answering")

    def test_serialization(self):
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            mnemo_customer_id="4584783",
            mnemo_corpus_id="4",
            mnemo_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
        )

        agent_reloaded = agent.loads(agent.dumps())
        self.assertIsInstance(agent_reloaded, Agent)
        self.assertEqual(agent, agent_reloaded)


if __name__ == "__main__":
    unittest.main()
