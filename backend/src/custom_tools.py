from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

def get_db_tools(db: SQLDatabase):
    """
    Returns a list of custom tools bound to the specific database instance.
    """

    @tool
    def sql_db_query_distinct_values(table_name: str, column_name: str) -> str:
        """
        Use this tool to get the top 10 distinct values for a specific column in a table.
        Useful for understanding categorical data or exact string spellings (e.g. 'Status' vs 'status').
        """
        try:
            return db.run(f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT 10")
        except Exception as e:
            return f"Error: {e}"

    @tool
    def sql_db_sample_rows(table_name: str) -> str:
        """
        Use this tool to get 3 sample rows from a table.
        Useful for understanding date formats, name formats, and data context.
        """
        try:
            return db.run(f"SELECT * FROM {table_name} LIMIT 3")
        except Exception as e:
            return f"Error: {e}"
    
    @tool
    def sql_db_find_table_by_column_name(keyword: str) -> str:
        """
        Search for tables that contain a specific column name keyword.
        Useful when you know the concept (e.g. 'label', 'score', 'risk') but not the table.
        """
        query = f"""
        SELECT TABLE_NAME, COLUMN_NAME 
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND COLUMN_NAME LIKE '%{keyword}%';
        """
        return db.run(query)

    return [sql_db_query_distinct_values, sql_db_sample_rows, sql_db_find_table_by_column_name]

