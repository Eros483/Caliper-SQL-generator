def select_table_prompt_module():
    return """
    You are a database architect. Your job is to select the tables required to answer the user's question.
    
    CRITICAL DATABASE KNOWLEDGE:
    1. **Patients**: Main profile is in table `patient`.
    2. **SDOH & Risks**: Definitions (Homelessness, Food Insecurity) are in `contributor_type`.
    3. **The Bridge**: To link Patients to SDOH/Risks, you MUST use `contributor_individual` (joins via patient_id and contr_type).
    4. **Scores**: Risk scores and metrics are in `patient_score`.
    5. **Identifiers**: If looking for 'member_id' or 'uuid', check `patient_identifier`.
    
    Return your response as a JSON object with a "table_names" list.
    """

def generate_query_prompt_module(db):
    generate_query_system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run.

    RULES:
    1. **Text vs IDs:** If the user provides a text name (e.g., "Housing Assistance") and you do not have the ID, **do not stop to ask the user**. Instead, filter the lookup table by its string column (e.g., `WHERE label = 'Housing Assistance'`).
    2. **Composite Keys:** If a table has a composite primary key (e.g., `id` + `org_id`) but the user only provides enough info for one part (the name), assume the query applies to ALL matching rows regardless of the second key.
    3. **Negative Queries:** To find patients who have *NOT* received an intervention, use a `NOT EXISTS` or `NOT IN` subquery.
    - Example: `SELECT * FROM patient WHERE patient_id NOT IN (SELECT patient_id FROM ... WHERE label = 'Housing Assistance')`
    4. **UUID Handling:** Understand that `BINARY(16)` columns (like `patient_id`) store UUIDs. If a user mentions UUIDs, they are referring to these columns.
    5. **Linking Tables:** To connect a `patient` to a `type` (like intervention or contributor), you usually need an intermediate table (e.g., `contributor_individual` or `intervention_service`).
    
    IMPORTANT:
    - DO NOT output the SQL query as raw text.
    - YOU MUST use the `sql_db_query` tool to execute the SQL.
    - If you output raw text starting with SELECT, you have failed.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )
    return generate_query_system_prompt

def query_verification_prompt_module(db):
    check_query_system_prompt = """
    You are a SQL expert with a strong attention to detail.
    Double check the {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes,
    just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check.
    """.format(dialect=db.dialect)
    return check_query_system_prompt

def answer_validation_prompt_module():
    return """
    You are a Data Analyst QA. You are reviewing a SQL query and its result.
    
    User Question: {question}
    Generated SQL: {query}
    SQL Execution Result: {result}
    
    Analyze the result:
    1. If the result is an Error Message, you MUST respond with 'RETRY'.
    2. If the result is Empty [], ask yourself: Is it likely that data exists but the query was too specific? 
       (e.g., using '=' instead of 'LIKE', wrong casing, or guessing a categorical value). 
       If yes, respond 'RETRY' and suggest checking distinct values or sample rows.
    3. If the result looks correct and answers the question, respond 'VALID'.
    
    Provide your response in this format:
    STATUS: [VALID | RETRY]
    FEEDBACK: [Your explanation here]
    """