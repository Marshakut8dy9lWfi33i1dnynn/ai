from mnemo_agentic.agent import Agent
from mnemo_agentic.agent_endpoint import start_app

from dotenv import load_dotenv
load_dotenv(override=True)

customer_id = '1366999410'
corpus_id = '1'
api_key = 'zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA'

assistant = Agent.from_corpus(
    tool_name = 'query_mnemo_website',
    mnemo_customer_id = customer_id,
    mnemo_corpus_id = corpus_id,
    mnemo_api_key = api_key,
    data_description = 'Data from mnemo.com website',
    assistant_specialty = 'mnemo'
)

start_app(assistant, host="0.0.0.0", port=8000)
