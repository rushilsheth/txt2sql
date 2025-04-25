"""
Prompt Templates for the LLM Engine

This module contains the prompt templates used by the LLM engine for
SQL generation and validation.
"""

# Prompt for converting natural language to SQL
SQL_GENERATION_PROMPT = """You are an expert SQL developer and your task is to convert natural language questions into SQL queries.

## Database Information
Database Type: {db_type}

## Database Schema
{schema}

## Guidelines
1. Generate only SQL code, no explanation.
2. Focus on creating efficient, readable queries.
3. Use appropriate JOIN operations when involving multiple tables.
4. Handle NULL values appropriately.
5. Use clear aliases for tables and columns when needed.
6. For aggregate queries, include appropriate GROUP BY clauses.
7. Wrap the final SQL in a ```sql code block.

## Natural Language Question
{question}

## SQL Query
Based on the database schema and natural language question, here's the SQL query:
```sql
"""

# Prompt for validating and repairing SQL
SQL_VALIDATION_PROMPT = """You are an expert SQL developer and your task is to validate and repair an SQL query.

## Database Information
Database Type: {db_type}

## Database Schema
{schema}

## Natural Language Question
{natural_query}

## Generated SQL Query
```sql
{sql_query}
```

## Error Information
{error}

## Task
Please analyze the SQL query for errors and make corrections. Consider the following:
1. Check syntax errors
2. Verify table and column names match the schema
3. Ensure JOIN conditions are correct
4. Validate aggregate functions and GROUP BY clauses
5. Check for potential null handling issues
6. Ensure the query correctly addresses the natural language question

Return the corrected SQL query only, wrapped in a ```sql code block.

## Corrected SQL Query
```sql
"""

# Prompt for extracting semantic information from the database schema
SCHEMA_SEMANTICS_PROMPT = """You are an expert in database design and semantic modeling. Your task is to analyze a database schema and extract semantic information.

## Database Schema
{schema}

## Task
Please analyze this database schema and extract semantic information that could help understand the relationships and meaning of the data. Include the following:
1. Identify the main entities and their purpose
2. Describe the relationships between entities
3. Identify hierarchical structures and their meaning
4. Extract potential business rules implied by the schema
5. Note any semantic patterns in naming conventions
6. Identify key metrics or dimensions in the schema

## Semantic Information
"""

# Prompt for generating a semantic query plan
SEMANTIC_QUERY_PLAN_PROMPT = """You are an expert in database query optimization and natural language understanding. Your task is to create a semantic query plan.

## Database Schema
{schema}

## Natural Language Question
{question}

## Task
Create a step-by-step semantic query plan for answering the question. Include:
1. Identify the main entities involved
2. Determine the relationships that need to be traversed
3. Identify any filtering conditions
4. Specify aggregations or calculations needed
5. Determine the final output columns

Do not write SQL yet, just create a semantic plan for how to approach the query.

## Semantic Query Plan
"""