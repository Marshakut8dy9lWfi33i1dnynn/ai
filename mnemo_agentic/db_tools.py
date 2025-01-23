"""
This module contains the code to extend and improve DatabaseToolSpec
Specifically adding load_sample_data and load_unique_values methods, as well as
making sure the load_data method returns a list of text values from the database, not Document[] objects.
"""
from abc import ABC
from typing import Callable, Any

#
# Additional database tool
#
class DBTool(ABC):
    """
    A base class for mnemo-agentic database tools extensions
    """
    def __init__(self, load_data_fn: Callable, max_rows: int = 500):
        self.load_data_fn = load_data_fn
        self.max_rows = max_rows

class DBLoadData(DBTool):
    """
    A tool to Run SQL query on the database and return the result.
    """
    def __call__(self, query: str) -> Any:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): an SQL query to filter tables and rows.

        Returns:
            List[text]: a list of text values from the database.
        """
        count_query = f"SELECT COUNT(*) FROM ({query})"
        try:
            count_rows = self.load_data_fn(count_query)
        except Exception as e:
            return [f"Error ({str(e)}) occurred while counting number of rows"]
        num_rows = int(count_rows[0].text)
        if num_rows > self.max_rows:
            return [
                f"The query is expected to return more than {self.max_rows} rows. "
                "Please refine your query to make it return less rows."
            ]
        try:
            res = self.load_data_fn(query)
        except Exception as e:
            return [f"Error ({str(e)}) occurred while executing the query {query}"]
        return [d.text for d in res]

class DBLoadSampleData(DBTool):
    """
    A tool to load a sample of data from the specified database table.

    This tool fetches the first num_rows (default 25) rows from the given table
    using a provided database query function.
    """
    def __call__(self, table_name: str, num_rows: int = 25) -> Any:
        """
        Fetches the first num_rows rows from the specified database table.

        Args:
            table_name (str): The name of the database table.

        Returns:
            Any: The result of the database query.
        """
        try:
            res = self.load_data_fn(f"SELECT * FROM {table_name} LIMIT {num_rows}")
        except Exception as e:
            return [f"Error ({str(e)}) occurred while loading sample data for table {table_name}"]
        return res

class DBLoadUniqueValues(DBTool):
    """
    A tool to list all unique values for each column in a set of columns of a database table.
    """
    def __call__(self, table_name: str, columns: list[str], num_vals: int = 200) -> dict:
        """
        Fetches the first num_vals unique values from the specified columns of the database table.

        Args:
            table_name (str): The name of the database table.
            columns (list[str]): The list of columns to fetch unique values from.
            num_vals (int): The number of unique values to fetch for each column. Default is 200.

        Returns:
            dict: A dictionary containing the unique values for each column.
        """
        res = {}
        try:
            for column in columns:
                unique_vals = self.load_data_fn(f'SELECT DISTINCT "{column}" FROM {table_name} LIMIT {num_vals}')
                res[column] = [d.text for d in unique_vals]
        except Exception as e:
            return {f"Error ({str(e)}) occurred while loading unique values for table {table_name}"}
        return res
