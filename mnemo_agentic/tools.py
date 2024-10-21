"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re
import importlib
import os

from typing import Callable, List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable
from llama_index.indices.managed.mnemo import MNEMOIndex
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools.types import ToolMetadata, ToolOutput


from .types import ToolType
from .tools_catalog import summarize_text, rephrase_text, critique_text, get_bad_topics
from .db_tools import DBLoadSampleData, DBLoadUniqueValues, DBLoadData
from .utils import is_float

LI_packages = {
    "yahoo_finance": ToolType.QUERY,
    "arxiv": ToolType.QUERY,
    "tavily_research": ToolType.QUERY,
    "exa": ToolType.QUERY,
    "neo4j": ToolType.QUERY,
    "kuzu": ToolType.QUERY,
    "database": ToolType.QUERY,
    "google": {
        "GmailToolSpec": {
            "load_data": ToolType.QUERY,
            "search_messages": ToolType.QUERY,
            "create_draft": ToolType.ACTION,
            "update_draft": ToolType.ACTION,
            "get_draft": ToolType.QUERY,
            "send_draft": ToolType.ACTION,
        },
        "GoogleCalendarToolSpec": {
            "load_data": ToolType.QUERY,
            "create_event": ToolType.ACTION,
            "get_date": ToolType.QUERY,
        },
        "GoogleSearchToolSpec": {"google_search": ToolType.QUERY},
    },
    "slack": {
        "SlackToolSpec": {
            "load_data": ToolType.QUERY,
            "send_message": ToolType.ACTION,
            "fetch_channel": ToolType.QUERY,
        }
    }
}

class MNEMOToolMetadata(ToolMetadata):
    """
    A subclass of ToolMetadata adding the tool_type attribute.
    """
    tool_type: ToolType

    def __init__(self, tool_type: ToolType, **kwargs):
        super().__init__(**kwargs)
        self.tool_type = tool_type

    def __repr__(self) -> str:
        """
        Returns a string representation of the MNEMOToolMetadata object, including the tool_type attribute.
        """
        base_repr = super().__repr__()
        return f"{base_repr}, tool_type={self.tool_type}"

class MNEMOTool(FunctionTool):
    """
    A subclass of FunctionTool adding the tool_type attribute.
    """

    def __init__(
        self,
        tool_type: ToolType,
        metadata: ToolMetadata,
        fn: Optional[Callable[..., Any]] = None,
        async_fn: Optional[AsyncCallable] = None,
    ) -> None:
        metadata_dict = metadata.dict() if hasattr(metadata, 'dict') else metadata.__dict__
        vm = MNEMOToolMetadata(tool_type=tool_type, **metadata_dict)
        super().__init__(fn, vm, async_fn)

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        tool_type: ToolType = ToolType.QUERY,
    ) -> "MNEMOTool":
        tool = FunctionTool.from_defaults(fn, name, description, return_direct, fn_schema, async_fn, tool_metadata)
        mnemo_tool = cls(tool_type=tool_type, fn=tool.fn, metadata=tool.metadata, async_fn=tool.async_fn)
        return mnemo_tool

    def __eq__(self, other):
        if self.metadata.tool_type != other.metadata.tool_type:
            return False

        # Check if fn_schema is an instance of a BaseModel or a class itself (metaclass)
        self_schema_dict = self.metadata.fn_schema.model_fields
        other_schema_dict = other.metadata.fn_schema.model_fields
        is_equal = True
        for key in self_schema_dict.keys():
            if key not in other_schema_dict:
                is_equal = False
                break
            if (
                self_schema_dict[key].annotation != other_schema_dict[key].annotation
                or self_schema_dict[key].description != other_schema_dict[key].description
                or self_schema_dict[key].is_required() != other_schema_dict[key].is_required()
            ):
                is_equal = False
                break
        return is_equal

def _build_filter_string(kwargs: Dict[str, Any], tool_args_type: Dict[str, str], fixed_filter: str) -> str:
    """
    Build filter string for MNEMO from kwargs
    """
    filter_parts = []
    comparison_operators = [">=", "<=", "!=", ">", "<", "="]
    numeric_only_ops = {">", "<", ">=", "<="}

    for key, value in kwargs.items():
        if value is None or value == "":
            continue

        # Determine the prefix for the key. Valid values are "doc" or "part"
        # default to 'doc' if not specified
        prefix = tool_args_type.get(key, "doc")

        if prefix not in ["doc", "part"]:
            raise ValueError(
                f'Unrecognized prefix {prefix}. Please make sure to use either "doc" or "part" for the prefix.'
            )

        if value is PydanticUndefined:
            raise ValueError(
                f"Value of argument {key} is undefined, and this is invalid. "
                "Please form proper arguments and try again."
            )

        # value of the argument
        val_str = str(value).strip()

        # Special handling for range operator
        if val_str.startswith(("[", "(")) and val_str.endswith(("]", ")")):
            # Extract the boundary types
            start_inclusive = val_str.startswith("[")
            end_inclusive = val_str.endswith("]")

            # Remove the boundaries and strip whitespace
            val_str = val_str[1:-1].strip()

            if "," in val_str:
                val_str = val_str.split(",")
                if len(val_str) != 2:
                    raise ValueError(
                        f"Range operator requires two values for {key}: {value}"
                    )

                # Validate both bounds as numeric or empty (for unbounded ranges)
                start_val, end_val = val_str[0].strip(), val_str[1].strip()
                if start_val and not (start_val.isdigit() or is_float(start_val)):
                    raise ValueError(
                        f"Range operator requires numeric operands for {key}: {value}"
                    )
                if end_val and not (end_val.isdigit() or is_float(end_val)):
                    raise ValueError(
                        f"Range operator requires numeric operands for {key}: {value}"
                    )

                # Build the SQL condition
                range_conditions = []
                if start_val:
                    operator = ">=" if start_inclusive else ">"
                    range_conditions.append(f"{prefix}.{key} {operator} {start_val}")
                if end_val:
                    operator = "<=" if end_inclusive else "<"
                    range_conditions.append(f"{prefix}.{key} {operator} {end_val}")

                # Join the range conditions with AND
                filter_parts.append('( ' + " AND ".join(range_conditions) + ' )')
                continue

            raise ValueError(
                f"Range operator requires two values for {key}: {value}"
            )

        # Check if value contains a known comparison operator at the start
        matched_operator = None
        for op in comparison_operators:
            if val_str.startswith(op):
                matched_operator = op
                break

        # Break down operator from value
        # e.g. val_str = ">2022" --> operator = ">", rhs = "2022"
        if matched_operator:
            rhs = val_str[len(matched_operator):].strip()

            if matched_operator in numeric_only_ops:
                # Must be numeric
                if not (rhs.isdigit() or is_float(rhs)):
                    raise ValueError(
                        f"Operator {matched_operator} requires a numeric operand for {key}: {val_str}"
                    )
                filter_parts.append(f"{prefix}.{key}{matched_operator}{rhs}")
            else:
                # = and != operators can be numeric or string
                if rhs.isdigit() or is_float(rhs):
                    filter_parts.append(f"{prefix}.{key}{matched_operator}{rhs}")
                elif rhs.lower() in ["true", "false"]:
                    filter_parts.append(f"{prefix}.{key}{matched_operator}{rhs.lower()}")
                else:
                    # For string operands, wrap them in quotes
                    filter_parts.append(f"{prefix}.{key}{matched_operator}'{rhs}'")
        else:
            if val_str.isdigit() or is_float(val_str):
                filter_parts.append(f"{prefix}.{key}={val_str}")
            elif val_str.lower() in ["true", "false"]:
                # This is to handle boolean values.
                # This is not complete solution - the best solution would be to test if the field is boolean
                # That can be done after we move to APIv2
                filter_parts.append(f"{prefix}.{key}={val_str.lower()}")
            else:
                filter_parts.append(f"{prefix}.{key}='{val_str}'")

    filter_str = " AND ".join(filter_parts)
    return f"({fixed_filter}) AND ({filter_str})" if fixed_filter else filter_str

class MNEMOToolFactory:
    """
    A factory class for creating MNEMO RAG tools.
    """

    def __init__(
        self,
        mnemo_customer_id: str = str(os.environ.get("VECTARA_CUSTOMER_ID", "")),
        mnemo_corpus_id: str = str(os.environ.get("VECTARA_CORPUS_ID", "")),
        mnemo_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
    ) -> None:
        """
        Initialize the MNEMOToolFactory
        Args:
            mnemo_customer_id (str): The MNEMO customer ID.
            mnemo_corpus_id (str): The MNEMO corpus ID (or comma separated list of IDs).
            mnemo_api_key (str): The MNEMO API key.
        """
        self.mnemo_customer_id = mnemo_customer_id
        self.mnemo_corpus_id = mnemo_corpus_id
        self.mnemo_api_key = mnemo_api_key
        self.num_corpora = len(mnemo_corpus_id.split(","))

    def create_search_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_args_schema: type[BaseModel],
        tool_args_type: Dict[str, str] = {},
        fixed_filter: str = "",
        lambda_val: float = 0.005,
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        verbose: bool = False,
    ) -> MNEMOTool:
        """
        Creates a MNEMO search/retrieval tool

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel): The schema for the tool arguments.
            tool_args_type (Dict[str, str], optional): The type of each argument (doc or part).
            fixed_filter (str, optional): A fixed MNEMO filter condition to apply to all queries.
            lambda_val (float, optional): Lambda value for the MNEMO query.
            reranker (str, optional): The reranker mode.
            rerank_k (int, optional): Number of top-k documents for reranking.
            mmr_diversity_bias (float, optional): MMR diversity bias.
            udf_expression (str, optional): the user defined expression for reranking results.
            rerank_chain (List[Dict], optional): A list of rerankers to be applied sequentially.
                Each dictionary should specify the "type" of reranker (mmr, slingshot, udf)
                and any other parameters (e.g. "limit" or "cutoff" for any type,
                "diversity_bias" for mmr, and "user_function" for udf).
                If using slingshot/multilingual_reranker_v1, it must be first in the list.
            verbose (bool, optional): Whether to print verbose output.

        Returns:
            MNEMOTool: A MNEMOTool object.
        """

        mnemo = MNEMOIndex(
            mnemo_api_key=self.mnemo_api_key,
            mnemo_customer_id=self.mnemo_customer_id,
            mnemo_corpus_id=self.mnemo_corpus_id,
            x_source_str="mnemo-agentic",
        )

        # Dynamically generate the search function
        def search_function(*args, **kwargs) -> ToolOutput:
            """
            Dynamically generated function for semantic search MNEMO.
            """
            # Convert args to kwargs using the function signature
            sig = inspect.signature(search_function)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments

            query = kwargs.pop("query")
            top_k = kwargs.pop("top_k", 10)
            try:
                filter_string = _build_filter_string(kwargs, tool_args_type, fixed_filter)
            except ValueError as e:
                return ToolOutput(
                    tool_name=search_function.__name__,
                    content=str(e),
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": str(e)},
                )

            mnemo_retriever = mnemo.as_retriever(
                summary_enabled=False,
                similarity_top_k=top_k,
                reranker=reranker,
                rerank_k=rerank_k if rerank_k * self.num_corpora <= 100 else int(100 / self.num_corpora),
                mmr_diversity_bias=mmr_diversity_bias,
                udf_expression=udf_expression,
                rerank_chain=rerank_chain,
                lambda_val=lambda_val,
                filter=filter_string,
                x_source_str="mnemo-agentic",
                verbose=verbose,
            )
            response = mnemo_retriever.retrieve(query)

            if len(response) == 0:
                msg = "MNEMO Tool failed to retreive any results for the query."
                return ToolOutput(
                    tool_name=search_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )
            tool_output = "Matching documents:\n"
            unique_ids = set()
            for doc in response:
                if doc.id_ in unique_ids:
                    continue
                unique_ids.add(doc.id_)
                tool_output += f"document '{doc.id_}' metadata: {doc.metadata}\n"
            out = ToolOutput(
                tool_name=search_function.__name__,
                content=tool_output,
                raw_input={"args": args, "kwargs": kwargs},
                raw_output=response,
            )
            return out

        fields = tool_args_schema.model_fields
        params = [
            inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=field_info.default,
                annotation=field_info,
            )
            for field_name, field_info in fields.items()
        ]

        # Create a new signature using the extracted parameters
        sig = inspect.Signature(params)
        search_function.__signature__ = sig
        search_function.__annotations__["return"] = dict[str, Any]
        search_function.__name__ = "_" + re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

        # Create the tool function signature string
        fields = []
        for name, field in tool_args_schema.__fields__.items():
            annotation = field.annotation
            type_name = annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)
            fields.append(f"{name}: {type_name}")
        args_str = ", ".join(fields)
        function_str = f"{tool_name}({args_str}) -> str"

        # Create the tool
        tool = MNEMOTool.from_defaults(
            fn=search_function,
            name=tool_name,
            description=function_str + ". " + tool_description,
            fn_schema=tool_args_schema,
            tool_type=ToolType.QUERY,
        )
        return tool

    def create_rag_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_args_schema: type[BaseModel],
        tool_args_type: Dict[str, str] = {},
        fixed_filter: str = "",
        mnemo_summarizer: str = "mnemo-summary-ext-24-05-sml",
        mnemo_prompt_text: str = None,
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        lambda_val: float = 0.005,
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        include_citations: bool = True,
        fcs_threshold: float = 0.0,
        verbose: bool = False,
    ) -> MNEMOTool:
        """
        Creates a RAG (Retrieve and Generate) tool.

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel): The schema for the tool arguments.
            tool_args_type (Dict[str, str], optional): The type of each argument (doc or part).
            fixed_filter (str, optional): A fixed MNEMO filter condition to apply to all queries.
            mnemo_summarizer (str, optional): The MNEMO summarizer to use.
            mnemo_prompt_text (str, optional): The prompt text for the MNEMO summarizer.
            summary_num_results (int, optional): The number of summary results.
            summary_response_lang (str, optional): The response language for the summary.
            n_sentences_before (int, optional): Number of sentences before the summary.
            n_sentences_after (int, optional): Number of sentences after the summary.
            lambda_val (float, optional): Lambda value for the MNEMO query.
            reranker (str, optional): The reranker mode.
            rerank_k (int, optional): Number of top-k documents for reranking.
            mmr_diversity_bias (float, optional): MMR diversity bias.
            udf_expression (str, optional): the user defined expression for reranking results.
            rerank_chain (List[Dict], optional): A list of rerankers to be applied sequentially.
                Each dictionary should specify the "type" of reranker (mmr, slingshot, udf)
                and any other parameters (e.g. "limit" or "cutoff" for any type,
                "diversity_bias" for mmr, and "user_function" for udf).
                If using slingshot/multilingual_reranker_v1, it must be first in the list.
            include_citations (bool, optional): Whether to include citations in the response.
                If True, uses markdown mnemo citations that requires the MNEMO scale plan.
            fcs_threshold (float, optional): a threshold for factual consistency.
                If set above 0, the tool notifies the calling agent that it "cannot respond" if FCS is too low.
            verbose (bool, optional): Whether to print verbose output.

        Returns:
            MNEMOTool: A MNEMOTool object.
        """

        mnemo = MNEMOIndex(
            mnemo_api_key=self.mnemo_api_key,
            mnemo_customer_id=self.mnemo_customer_id,
            mnemo_corpus_id=self.mnemo_corpus_id,
            x_source_str="mnemo-agentic",
        )

        # Dynamically generate the RAG function
        def rag_function(*args, **kwargs) -> ToolOutput:
            """
            Dynamically generated function for RAG query with MNEMO.
            """
            # Convert args to kwargs using the function signature
            sig = inspect.signature(rag_function)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments

            query = kwargs.pop("query")
            try:
                filter_string = _build_filter_string(kwargs, tool_args_type, fixed_filter)
            except ValueError as e:
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=str(e),
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": str(e)},
                )

            mnemo_query_engine = mnemo.as_query_engine(
                summary_enabled=True,
                summary_num_results=summary_num_results,
                summary_response_lang=summary_response_lang,
                summary_prompt_name=mnemo_summarizer,
                prompt_text=mnemo_prompt_text,
                reranker=reranker,
                rerank_k=rerank_k if rerank_k * self.num_corpora <= 100 else int(100 / self.num_corpora),
                mmr_diversity_bias=mmr_diversity_bias,
                udf_expression=udf_expression,
                rerank_chain=rerank_chain,
                n_sentence_before=n_sentences_before,
                n_sentence_after=n_sentences_after,
                lambda_val=lambda_val,
                filter=filter_string,
                citations_style="MARKDOWN" if include_citations else None,
                citations_url_pattern="{doc.url}" if include_citations else None,
                x_source_str="mnemo-agentic",
                verbose=verbose,
            )
            response = mnemo_query_engine.query(query)

            if len(response.source_nodes) == 0:
                msg = "Tool failed to generate a response since no matches were found."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )
            if str(response) == "None":
                msg = "Tool failed to generate a response."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )

            # Extract citation metadata
            pattern = r"\[(\d+)\]"
            matches = re.findall(pattern, response.response)
            citation_numbers = sorted(set(int(match) for match in matches))
            citation_metadata = ""
            keys_to_ignore = ["lang", "offset", "len"]
            for citation_number in citation_numbers:
                metadata = response.source_nodes[citation_number - 1].metadata
                citation_metadata += (
                    f"[{citation_number}]: "
                    + "; ".join(
                        [
                            f"{k}='{v}'"
                            for k, v in metadata.items()
                            if k not in keys_to_ignore
                        ]
                    )
                    + ".\n"
                )
            fcs = response.metadata["fcs"] if "fcs" in response.metadata else 0.0
            if fcs < fcs_threshold:
                msg = f"Could not answer the query due to suspected hallucination (fcs={fcs})."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )
            res = {
                "response": response.response,
                "references_metadata": citation_metadata,
            }
            if len(citation_metadata) > 0:
                tool_output = f"""
                    Response: '''{res['response']}'''
                    References:
                    {res['references_metadata']}
                """
            else:
                tool_output = f"Response: '''{res['response']}'''"
            out = ToolOutput(
                tool_name=rag_function.__name__,
                content=tool_output,
                raw_input={"args": args, "kwargs": kwargs},
                raw_output=res,
            )
            return out

        fields = tool_args_schema.model_fields
        params = [
            inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=field_info.default,
                annotation=field_info,
            )
            for field_name, field_info in fields.items()
        ]

        # Create a new signature using the extracted parameters
        sig = inspect.Signature(params)
        rag_function.__signature__ = sig
        rag_function.__annotations__["return"] = dict[str, Any]
        rag_function.__name__ = "_" + re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

        # Create the tool function signature string
        fields = []
        for name, field in tool_args_schema.__fields__.items():
            annotation = field.annotation
            type_name = annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)
            fields.append(f"{name}: {type_name}")
        args_str = ", ".join(fields)
        function_str = f"{tool_name}({args_str}) -> str"

        # Create the tool
        tool = MNEMOTool.from_defaults(
            fn=rag_function,
            name=tool_name,
            description=function_str + ". " + tool_description,
            fn_schema=tool_args_schema,
            tool_type=ToolType.QUERY,
        )
        return tool

class ToolsFactory:
    """
    A factory class for creating agent tools.
    """

    def create_tool(self, function: Callable, tool_type: ToolType = ToolType.QUERY) -> MNEMOTool:
        """
        Create a tool from a function.

        Args:
            function (Callable): a function to convert into a tool.
            tool_type (ToolType): the type of tool.

        Returns:
            MNEMOTool: A MNEMOTool object.
        """
        return MNEMOTool.from_defaults(tool_type=tool_type, fn=function)

    def get_llama_index_tools(
        self,
        tool_package_name: str,
        tool_spec_name: str,
        tool_name_prefix: str = "",
        **kwargs: dict,
    ) -> List[MNEMOTool]:
        """
        Get a tool from the llama_index hub.

        Args:
            tool_package_name (str): The name of the tool package.
            tool_spec_name (str): The name of the tool spec.
            tool_name_prefix (str, optional): The prefix to add to the tool names (added to every tool in the spec).
            kwargs (dict): The keyword arguments to pass to the tool constructor (see Hub for tool specific details).

        Returns:
            List[MNEMOTool]: A list of MNEMOTool objects.
        """
        # Dynamically install and import the module
        if tool_package_name not in LI_packages:
            raise ValueError(f"Tool package {tool_package_name} from LlamaIndex not supported by MNEMO-agentic.")

        module_name = f"llama_index.tools.{tool_package_name}"
        module = importlib.import_module(module_name)

        # Get the tool spec class or function from the module
        tool_spec = getattr(module, tool_spec_name)

        func_type = LI_packages[tool_package_name]
        tools = tool_spec(**kwargs).to_tool_list()
        vtools = []
        for tool in tools:
            if len(tool_name_prefix) > 0:
                tool.metadata.name = tool_name_prefix + "_" + tool.metadata.name
            if isinstance(func_type, dict):
                if tool_spec_name not in func_type.keys():
                    raise ValueError(f"Tool spec {tool_spec_name} not found in package {tool_package_name}.")
                tool_type = func_type[tool_spec_name]
            else:
                tool_type = func_type
            vtool = MNEMOTool(tool_type=tool_type, fn=tool.fn, metadata=tool.metadata, async_fn=tool.async_fn)
            vtools.append(vtool)
        return vtools

    def standard_tools(self) -> List[FunctionTool]:
        """
        Create a list of standard tools.
        """
        return [self.create_tool(tool) for tool in [summarize_text, rephrase_text]]

    def guardrail_tools(self) -> List[FunctionTool]:
        """
        Create a list of guardrail tools to avoid controversial topics.
        """
        return [self.create_tool(get_bad_topics)]

    def financial_tools(self):
        """
        Create a list of financial tools.
        """
        return self.get_llama_index_tools(tool_package_name="yahoo_finance", tool_spec_name="YahooFinanceToolSpec")

    def legal_tools(self) -> List[FunctionTool]:
        """
        Create a list of legal tools.
        """

        def summarize_legal_text(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Use this tool to summarize legal text with no more than summary_max_length characters.
            """
            return summarize_text(text, expertise="law")

        def critique_as_judge(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Critique the legal document.
            """
            return critique_text(
                text,
                role="judge",
                point_of_view="""
                an experienced judge evaluating a legal document to provide areas of concern
                or that may require further legal scrutiny or legal argument.
                """,
            )

        return [self.create_tool(tool) for tool in [summarize_legal_text, critique_as_judge]]

    def database_tools(
        self,
        tool_name_prefix: str = "",
        content_description: Optional[str] = None,
        sql_database: Optional[SQLDatabase] = None,
        scheme: Optional[str] = None,
        host: str = "localhost",
        port: str = "5432",
        user: str = "postgres",
        password: str = "Password",
        dbname: str = "postgres",
        max_rows: int = 500,
    ) -> List[MNEMOTool]:
        """
        Returns a list of database tools.

        Args:

            tool_name_prefix (str, optional): The prefix to add to the tool names. Defaults to "".
            content_description (str, optional): The content description for the database. Defaults to None.
            sql_database (SQLDatabase, optional): The SQLDatabase object. Defaults to None.
            scheme (str, optional): The database scheme. Defaults to None.
            host (str, optional): The database host. Defaults to "localhost".
            port (str, optional): The database port. Defaults to "5432".
            user (str, optional): The database user. Defaults to "postgres".
            password (str, optional): The database password. Defaults to "Password".
            dbname (str, optional): The database name. Defaults to "postgres".
               You must specify either the sql_database object or the scheme, host, port, user, password, and dbname.
            max_rows (int, optional): if specified, instructs the load_data tool to never return more than max_rows
               rows. Defaults to 500.

        Returns:
            List[MNEMOTool]: A list of MNEMOTool objects.
        """
        if sql_database:
            tools = self.get_llama_index_tools(
                tool_package_name="database",
                tool_spec_name="DatabaseToolSpec",
                tool_name_prefix=tool_name_prefix,
                sql_database=sql_database,
            )
        else:
            if scheme in ["postgresql", "mysql", "sqlite", "mssql", "oracle"]:
                tools = self.get_llama_index_tools(
                    tool_package_name="database",
                    tool_spec_name="DatabaseToolSpec",
                    tool_name_prefix=tool_name_prefix,
                    scheme=scheme,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                )
            else:
                raise ValueError(
                    "Please provide a SqlDatabase option or a valid DB scheme type "
                    " (postgresql, mysql, sqlite, mssql, oracle)."
                )

        # Update tools with description
        for tool in tools:
            if content_description:
                tool.metadata.description = (
                    tool.metadata.description + f"The database tables include data about {content_description}."
                )

        # Add two new tools: load_sample_data and load_unique_values
        load_data_tool_index = next(i for i, t in enumerate(tools) if t.metadata.name.endswith("load_data"))
        load_data_fn_original = tools[load_data_tool_index].fn

        load_data_fn = DBLoadData(load_data_fn_original, max_rows=max_rows)
        load_data_fn.__name__ = f"{tool_name_prefix}_load_data"
        load_data_tool = self.create_tool(load_data_fn, ToolType.QUERY)

        sample_data_fn = DBLoadSampleData(load_data_fn_original)
        sample_data_fn.__name__ = f"{tool_name_prefix}_load_sample_data"
        sample_data_tool = self.create_tool(sample_data_fn, ToolType.QUERY)

        load_unique_values_fn = DBLoadUniqueValues(load_data_fn_original)
        load_unique_values_fn.__name__ = f"{tool_name_prefix}_load_unique_values"
        load_unique_values_tool = self.create_tool(load_unique_values_fn, ToolType.QUERY)

        tools[load_data_tool_index] = load_data_tool
        tools.extend([sample_data_tool, load_unique_values_tool])
        return tools
