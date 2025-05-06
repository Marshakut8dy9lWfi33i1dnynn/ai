# Mnemo

<p align="center">
  <img src="assets/Mnemo_Logo.png" alt="Mnemo Logo" width="300" />
</p>

<p align="center">
  <em>Composable AI Agents & Realtime Data Interfaces Powered by Model Context Protocol</em>
</p>

<p align="center">
<img src="https://img.shields.io/pypi/l/mnemo" />
</p>

---

## Overview

**Mnemo** is a modular agent framework built on top of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), designed to orchestrate Retrieval-Augmented Generation (RAG) pipelines and intelligent agent workflows using real-time, pluggable data services.

Mnemo integrates two emerging standards:

1. **Model Context Protocol (MCP)**: Enables real-time, protocol-based interaction with external tools, data streams, and services via MCP servers.
2. **Composable Agent Architecture**: Inspired by effective production patterns, Mnemo allows developers to build, chain, and orchestrate modular agents across tasks and domains.

### Why Mnemo?

Mnemo is purpose-built to:

* 🔌 **Plug into any MCP-compliant data or tool service**
* 🔍 **Enable real-time RAG pipelines with multi-modal inputs**
* 🧠 **Build chainable, domain-specific agents with memory, logic and persistence**
* 🧩 **Expose agents as MCP clients or servers, enabling two-way integration**

Whether you're building autonomous workflows, human-in-the-loop systems, or live decision agents powered by streaming on-chain or enterprise data—Mnemo provides the infrastructure layer to deploy them quickly.

---

## Features

* ⚙️ **MCP-Oriented Design**: Fully compatible with MCP server/client pattern; enables hot-swappable data interfaces and execution environments.
* 📚 **RAG-Native Agent Workflows**: First-class support for Retrieval-Augmented Generation with vector store and unstructured data integration.
* 🤖 **Composable Agent Engine**: Build modular agents that orchestrate, call tools, persist memory, and coordinate via workflows.
* 🪝 **Real-Time Tool Calls**: Automatically fetch, retrieve, and operate on data exposed by any MCP-compliant service (e.g., filesystem, fetch, email, SQL, vector DBs).
* 🧪 **Multi-Agent Orchestration**: Supports cooperative task planning, evaluation agents, and Swarm-style distributed processing.

---

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environments:

```bash
uv add "mnemo"
```

Or simply use pip:

```bash
pip install mnemo
```

---

## Quickstart

Clone the repo and run a basic demo agent:

```bash
cd examples/basic/mnemo_demo_agent
cp mnemo.secrets.yaml.example mnemo.secrets.yaml  # Add your API keys
uv run main.py
```

### Example: File and Web Agent

```python
from mnemo.app import MnemoApp
from mnemo.agents.agent import Agent
from mnemo.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MnemoApp(name="web_reader_agent")

async def run():
    async with app.run() as session:
        reader = Agent(
            name="finder",
            instruction="""
            You can read files and browse web links. Return requested information on demand.
            """,
            server_names=["filesystem", "fetch"],
        )

        async with reader:
            tools = await reader.list_tools()
            llm = await reader.attach_llm(OpenAIAugmentedLLM)

            output = await llm.generate_str("Read me the first 10 lines of README.md")
            print("README preview:", output)

            result = await llm.generate_str("Summarize this article: https://www.anthropic.com/research/building-effective-agents")
            print("Summary:", result)
```

---

## Applications

### ✅ RAG-Enhanced Q\&A

Integrate with vector DBs (e.g. Qdrant, Weaviate) to retrieve relevant text passages and enable context-rich answering.

### 🧾 Enterprise Memory Agents

Deploy agents with long-term memory over internal knowledge, business logic, or customer records.

### 📡 On-Chain Analytics Agents

Stream blockchain data via MCP-compatible servers and perform structured analysis or alerts.

### 🛠️ Custom Toolchains

Create domain-specific agents that orchestrate tasks using external APIs or plugins via the MCP layer.

### 🧠 Multimodal Reasoning

Extend beyond text: support for image embeddings, structured documents, web interfaces, and speech-ready agents.

---

## Roadmap

* ✅ Multi-agent Swarm workflows (inspired by OpenAI's Swarm)
* ✅ Long-running workflow orchestration with pause/resume
* ⏳ Persistent agent memory & streaming input support
* 🧠 LLM model switch support (Claude, GPT-4o, etc.)
* 🧩 More MCP server connectors: calendar, cloud docs, database, sensors

---

## Credits

Built with ❤️ on top of MCP and inspired by Anthropic’s vision for composable, intelligent agents.
