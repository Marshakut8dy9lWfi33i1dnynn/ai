# 
<p align="center">
<img  src="assets/Mnemo_Logo.png" alt="Mnemo Logo" width="30" height="30" style="vertical-align: middle;"> Mnemo AI



<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

## ✨ Overview

Mnemo AI is the next-gen framework powering intelligent agent development through Retrieval-Augmented Generation.

<p align="center">
   <img align="center" src="assets/Mnemo_AI_Diagram.png" alt="Mnemo AI Diagram">
</p>

###  Features

- Enables easy creation of custom AI assistants and agents.
- Create a Mnemo RAG tool or search tool with a single line of code.
- Supports `ReAct`, `OpenAIAgent`, `LATS` and `LLMCompiler` agent types.
- Includes pre-built tools for various domains (e.g., finance, legal).
- Integrates with various LLM inference services like OpenAI, DeepSeek, Anthropic, Gemini, GROQ, Together.AI, Cohere, Bedrock and Fireworks
- Built-in support for observability with Arize Phoenix




## 🚀 Quick Start

### 1. Initialize the Mnemo tool factory

```python
import os
from mnemo_agentic.tools import MnemoToolFactory

vec_factory = MnemoToolFactory(
    mnemo_api_key=os.environ['MNEMO_API_KEY'],
    mnemo_customer_id=os.environ['MNEMO_CUSTOMER_ID'],
    mnemo_corpus_id=os.environ['MNEMO_CORPUS_ID']
)
```

### 2. Create a Mnemo RAG Tool

A RAG tool calls the full Mnemo RAG pipeline to provide summarized responses to queries grounded in data.

```python
from pydantic import BaseModel, Field

years = list(range(2024, 2025))
tickers = {
    "TRUMP": "OFFICIAL TRUMP",
    "VINE": "Vine Coin",
    "PENGU": "Pudgy Penguins",
    "GOAT": "Goatseus Maximus",
}

class QueryMemecoinReportsArgs(BaseModel):
    query: str = Field(..., description="The user query.")
    year: int | str = Field(..., description=f"The year this query relates to. An integer between {min(years)} and {max(years)} or a string specifying a condition on the year (example: '>2020').")
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")

query_memecoin_reports_tool = vec_factory.create_rag_tool(
    tool_name="query_memecoin_reports",
    tool_description="Query memecoin reports for a memecoin and date",
    tool_args_schema=QueryMemecoinReportsArgs,
    lambda_val=0.005,
    summary_num_results=7, 
    # Additional arguments
)
```


### 3. Create other tools (optional)

In addition to RAG tools, you can generate a lot of other types of tools the agent can use. These could be mathematical tools, tools 
that call other APIs to get more information, or any other type of tool.


### 4. Create your agent

```python
from mnemo_agentic import Agent

agent = Agent(
    tools=[query_memecoin_reports_tool],
    topic="10-K memecoin reports",
    custom_instructions="""
        - You are a helpful memecoin assistant in conversation with a user. Use your memecoin expertise when crafting a query to the tool, to ensure you get the most accurate information.
        - You can answer questions, provide insights, or summarize any information from memecoin reports.
        - A user may refer to a memecoin's ticker instead of its full name - consider those the same when a user is asking about a memecoin.
        - When calculating a memecoin metric, make sure you have all the information from tools to complete the calculation.
        - In many cases you may need to query tools on each sub-metric separately before computing the final metric.
        - Report memecoin data in a consistent manner. For example if you report values in Solana, always report values in Solana.
    """
)
```



### 5. Run your agent

```python
res = agent.chat("How much did the top traders make on $GOAT?")
print(res.response)
```

Note that:
1. `mnemo-agentic` also supports `achat()` and two streaming variants `stream_chat()` and `astream_chat()`.
2. The response types from `chat()` and `achat()` are of type `AgentResponse`. If you just need the actual string
   response it's available as the `response` variable, or just use `str()`. For advanced use-cases you can look 
   at other `AgentResponse` variables [such as `sources`](https://github.com/run-llama/llama_index/blob/659f9faaafbecebb6e6c65f42143c0bf19274a37/llama-index-core/llama_index/core/chat_engine/types.py#L53).

## 🧰 Mnemo tools

`mnemo-agentic` provides two helper functions to connect with Mnemo RAG
* `create_rag_tool()` to create an agent tool that connects with a Mnemo corpus for querying. 
* `create_search_tool()` to create a tool to search a Mnemo corpus and return a list of matching documents.

See the documentation for the full list of arguments for `create_rag_tool()` and `create_search_tool()`, 
to understand how to configure Mnemo query performed by those tools.

### Creating a Mnemo RAG tool

A Mnemo RAG tool is often the main workhorse for any Agentic RAG application, and enables the agent to query 
one or more Mnemo RAG corpora. 

The tool generated always includes the `query` argument, followed by 1 or more optional arguments used for 
metadata filtering, defined by `tool_args_schema`.

For example, in the quickstart example the schema is:

```
class QueryMemecoinReportsArgs(BaseModel):
    query: str = Field(..., description="The user query.")
    year: int | str = Field(..., description=f"The year this query relates to. An integer between {min(years)} and {max(years)} or a string specifying a condition on the year (example: '>2020').")
    ticker: str = Field(..., description=f"The token ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")
```

The `query` is required and is always the query string.
The other arguments are optional and will be interpreted as Mnemo metadata filters.

For example, in the example above, the agent may call the `query_memecoin_reports_tool` tool with 
query='how much did the top traders make?', year=2024 and ticker='GOAT'. Subsequently the RAG tool will issue
a Mnemo RAG query with the same query, but with metadata filtering (doc.year=2024 and doc.ticker='GOAT').

There are also additional cool features supported here:
* An argument can be a condition, for example year='>2024' translates to the correct metadata 
  filtering condition doc.year>2024
* if `fixed_filter` is defined in the RAG tool, it provides a constant metadata filtering that is always applied.
  For example, if fixed_filter=`doc.filing_type='10K'` then a query with query='what is the market cap', year=2024
  and ticker='GOAT' would translate into query='what is the market cap' with metadata filtering condition of
  "doc.year=2024 AND doc.ticker='GOAT' and doc.filing_type='10K'"

Note that `tool_args_type` is an optional dictionary that indicates the level at which metadata filtering
is applied for each argument (`doc` or `part`)

### Creating a Mnemo search tool

The Mnemo search tool allows the agent to list documents that match a query.
This can be helpful to the agent to answer queries like "how many documents discuss the iPhone?" or other
similar queries that require a response in terms of a list of matching documents.

## 🛠️ Agent Tools at a Glance

`mnemo-agentic` provides a few tools out of the box:
1. **Standard tools**: 
- `summarize_text`: a tool to summarize a long text into a shorter summary (uses LLM)
- `rephrase_text`: a tool to rephrase a given text, given a set of rephrase instructions (uses LLM)

2. **Memecoin tools**: based on tools from Dexscreener:
- tools to understand the memecoins of a pump.fun: `market_cap`, `volume`, `holder_distribution`
- `token_news`: provides news about a token
- `token_analyst_recommendations`: provides token analyst recommendations for a memecoin.

3. **Database tools**: providing tools to inspect and query a database
- `list_tables`: list all tables in the database
- `describe_tables`: describe the schema of tables in the database
- `load_data`: returns data based on a SQL query
- `load_sample_data`: returns the first 25 rows of a table
- `load_unique_values`: returns the top unique values for a given column

In addition, we include various other tools from LlamaIndex ToolSpecs:
* Tavily search and EXA.AI
* arxiv
* neo4j & Kuzu for Graph DB integration
* Google tools (including gmail, calendar, and search)
* Slack

Note that some of these tools may require API keys as environment variables

You can create your own tool directly from a Python function using the `create_tool()` method of the `ToolsFactory` class:

```python
def mult_func(x, y):
    return x * y

mult_tool = ToolsFactory().create_tool(mult_func)
```

## 🛠️ Configuration

The main way to control the behavior of `mnemo-agentic` is by passing an `AgentConfig` object to your `Agent` when creating it.
This object will include the following items:
- `MNEMO_AGENTIC_AGENT_TYPE`: valid values are `REACT`, `LLMCOMPILER`, `LATS` or `OPENAI` (default: `OPENAI`)
- `MNEMO_AGENTIC_MAIN_LLM_PROVIDER`: valid values are `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, `COHERE`, `BEDROCK`, `GEMINI` or `FIREWORKS` (default: `OPENAI`)
- `MNEMO_AGENTIC_MAIN_MODEL_NAME`: agent model name (default depends on provider)
- `MNEMO_AGENTIC_TOOL_LLM_PROVIDER`: tool LLM provider (default: `OPENAI`)
- `MNEMO_AGENTIC_TOOL_MODEL_NAME`: tool model name (default depends on provider)
- `MNEMO_AGENTIC_OBSERVER_TYPE`: valid values are `ARIZE_PHOENIX` or `NONE` (default: `NONE`)
- `MNEMO_AGENTIC_API_KEY`: a secret key if using the API endpoint option (defaults to `dev-api-key`)

If any of these are not provided, `AgentConfig` first tries to read the values from the OS environment.

When creating a `MnemoToolFactory`, you can pass in a `mnemo_api_key`, `mnemo_customer_id`, and `mnemo_corpus_id` to the factory. If not passed in, it will be taken from the environment variables (`MNEMO_API_KEY`, `MNEMO_CUSTOMER_ID` and `MNEMO_CORPUS_ID`). Note that `MNEMO_CORPUS_ID` can be a single ID or a comma-separated list of IDs (if you want to query multiple corpora).

## ℹ️ Additional Information

### About Custom Instructions for your Agent

The custom instructions you provide to the agent guide its behavior.
Here are some guidelines when creating your instructions:
- Write precise and clear instructions, without overcomplicating.
- Consider edge cases and unusual or atypical scenarios.
- Be cautious to not over-specify behavior based on your primary use-case, as it may limit the agent's ability to behave properly in others.

###  Diagnostics

The `Agent` class defines a few helpful methods to help you understand the internals of your application. 
* The `report()` method prints out the agent object's type, the tools, and the LLMs used for the main agent and tool calling.
* The `token_counts()` method tells you how many tokens you have used in the current session for both the main agent and tool calling LLMs. This can be helpful if you want to track spend by token.

###  Serialization

The `Agent` class supports serialization. Use the `dumps()` to serialize and `loads()` to read back from a serialized stream.


## 🌐 API Endpoint

`mnemo-agentic` can be easily hosted locally or on a remote machine behind an API endpoint, by following theses steps:

### Step 1: Setup your API key
Ensure that you have your API key set up as an environment variable:

```
export MNEMO_AGENTIC_API_KEY=<YOUR-ENDPOINT-API-KEY>
```

if you don't specify an Endpoint API key it uses the default "dev-api-key".

### Step 2: Start the API Server
Initialize the agent and start the FastAPI server by following this example:


```
from mnemo_agentic.agent import Agent
from mnemo_agentic.agent_endpoint import start_app
agent = Agent(...)            # Initialize your agent with appropriate parameters
start_app(agent)
```

You can customize the host and port by passing them as arguments to `start_app()`:
* Default: host="0.0.0.0" and port=8000.
For example:
```
start_app(agent, host="0.0.0.0", port=8000)
```

### Step 3: Access the API Endpoint
Once the server is running, you can interact with it using curl or any HTTP client. For example:

```
curl -G "http://<remote-server-ip>:8000/chat" \
--data-urlencode "message=What is Mnemo?" \
-H "X-API-Key: <YOUR-ENDPOINT-API-KEY>"
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guide](https://github.com/mnemo-agentic/mnemo/blob/main/CONTRIBUTING.md) for more information.

## 📝 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/mnemo-agentic/mnemo/blob/master/LICENSE) file for details.

## 📞 Contact

- Twitter: [@mnemo_ai](https://twitter.com/mnemo_ai
)
- GitHub: [MnemoAi](https://github.com/MnemoAI)