"""
LLM-based Agent Implementations

This module provides implementations of the agent interfaces using LLMs.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from text_to_sql.agent.types import (
    Agent, AgentContext, AgentRole,
    QueryUnderstandingAgent, SchemaAnalysisAgent, 
    SQLGenerationAgent, QueryValidationAgent, 
    ResultExplanationAgent, VisualizationAgent,
    CoordinatorAgent
)
from text_to_sql.db.base import DatabaseManager
from text_to_sql.db.schema import SchemaAnalyzer
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.llm.prompts import (
    SQL_GENERATION_PROMPT, SQL_VALIDATION_PROMPT, 
    SCHEMA_SEMANTICS_PROMPT, SEMANTIC_QUERY_PLAN_PROMPT
)

logger = logging.getLogger(__name__)


class LLMQueryUnderstandingAgent(QueryUnderstandingAgent):
    """
    LLM-based implementation of the QueryUnderstandingAgent.
    
    Uses an LLM to extract intent and entities from natural language queries.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        name: str = "LLMQueryUnderstanding", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM query understanding agent.
        
        Args:
            llm_engine: LLM engine instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, f"Analyzing user query: '{context.user_query}'")
        
        # Extract intent
        start_time = time.time()
        intent = self.extract_intent(context.user_query)
        context.query_intent = intent
        context.execution_time["intent_extraction"] = time.time() - start_time
        
        self.log_reasoning(context, f"Extracted intent: {intent}")
        
        # Extract entities
        start_time = time.time()
        entities = self.extract_entities(context.user_query)
        context.query_entities = entities
        context.execution_time["entity_extraction"] = time.time() - start_time
        
        self.log_reasoning(context, f"Extracted entities: {', '.join(entities)}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def extract_intent(self, query: str) -> str:
        """
        Extract the primary intent from a query using an LLM.
        
        Args:
            query: User's natural language query
            
        Returns:
            Primary intent
        """
        prompt = f"""You are an expert in natural language understanding.
Extract the primary intent from this database query:

Query: "{query}"

Possible intents:
- SELECT: Basic retrieval of data
- AGGREGATE: Computing aggregates like count, sum, average, etc.
- FILTER: Retrieving data with specific conditions
- JOIN: Combining data from multiple tables
- SORT: Ordering data
- GROUP: Grouping and summarizing data
- COMPLEX: Multiple operations combined

Intent:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract the intent from the response
        intent = response.strip().split("\n")[0]
        
        # Remove any extraneous text
        for prefix in ["Intent:", "The intent is:", "Primary intent:"]:
            if intent.startswith(prefix):
                intent = intent[len(prefix):].strip()
        
        return intent
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from a query using an LLM.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of entity names
        """
        prompt = f"""You are an expert in natural language understanding.
Extract the database entities (tables, columns, values) from this query:

Query: "{query}"

List each entity as a simple name, with no explanation. Use the format:
ENTITY_TYPE: entity_name

Where ENTITY_TYPE is one of: TABLE, COLUMN, VALUE

Entities:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract entities from the response
        entities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Parse the entity type and name
            if ":" in line:
                entity_type, entity_name = line.split(":", 1)
                entity_name = entity_name.strip()
                
                # Add to the list
                if entity_name:
                    entities.append(entity_name)
        
        return entities


class LLMSchemaAnalysisAgent(SchemaAnalysisAgent):
    """
    LLM-based implementation of the SchemaAnalysisAgent.
    
    Uses an LLM to identify relevant tables and columns for a query.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        db_manager: DatabaseManager,
        name: str = "LLMSchemaAnalysis", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM schema analysis agent.
        
        Args:
            llm_engine: LLM engine instance
            db_manager: Database manager instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
        self.db_manager = db_manager
        
        # Schema analyzer
        self.schema_analyzer = SchemaAnalyzer(db_manager.get_schema())
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Analyzing database schema for relevant parts")
        
        # Find relevant tables
        start_time = time.time()
        relevant_tables = self.identify_relevant_tables(context)
        context.relevant_tables = relevant_tables
        context.execution_time["table_identification"] = time.time() - start_time
        
        self.log_reasoning(context, f"Identified relevant tables: {', '.join(relevant_tables)}")
        
        # Find relevant columns
        start_time = time.time()
        relevant_columns = self.identify_relevant_columns(context)
        context.relevant_columns = relevant_columns
        context.execution_time["column_identification"] = time.time() - start_time
        
        # Log the identified columns
        column_log = []
        for table, columns in relevant_columns.items():
            column_log.append(f"{table}: {', '.join(columns)}")
        
        self.log_reasoning(context, f"Identified relevant columns: {'; '.join(column_log)}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def identify_relevant_tables(self, context: AgentContext) -> List[str]:
        """
        Identify the tables relevant to a query.
        
        Args:
            context: Agent context
            
        Returns:
            List of relevant table names
        """
        # Get schema
        schema = self.db_manager.get_schema()
        
        # Format the schema for the prompt
        schema_str = self.llm_engine._format_schema_for_prompt(schema)
        
        # Create the prompt
        prompt = f"""You are an expert database analyst.
Identify the tables that are relevant to this query:

Query: "{context.user_query}"

Database Schema:
{schema_str}

List only the table names (schema.table format) that are directly relevant to answering this query.
Do not include tables that are not needed to answer the query.

Relevant Tables:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract table names from the response
        tables = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Remove any list markers
            if line.startswith("- "):
                line = line[2:]
            if line.startswith("* "):
                line = line[2:]
            if line[0].isdigit() and line[1:].startswith(". "):
                line = line[line.find(". ") + 2:]
            
            # Add to the list
            if line and "." in line:  # Ensuring it's a schema.table format
                tables.append(line)
        
        return tables
    
    def identify_relevant_columns(self, context: AgentContext) -> Dict[str, List[str]]:
        """
        Identify the columns relevant to a query.
        
        Args:
            context: Agent context
            
        Returns:
            Dictionary mapping table names to lists of column names
        """
        # Get schema
        schema = self.db_manager.get_schema()
        
        # Init result
        relevant_columns = {}
        
        # Process each relevant table
        for table_name in context.relevant_tables:
            # Parse schema and table name
            if "." in table_name:
                schema_name, table = table_name.split(".", 1)
            else:
                schema_name = "public"
                table = table_name
            
            # Check if table exists
            if schema_name not in schema or table not in schema[schema_name]:
                continue
            
            # Get table columns
            table_columns = list(schema[schema_name][table]["columns"].keys())
            
            # Create a prompt to identify relevant columns
            prompt = f"""You are an expert database analyst.
Identify the columns from table {table_name} that are relevant to this query:

Query: "{context.user_query}"

Available columns in {table_name}:
{', '.join(table_columns)}

List only the column names that are directly relevant to answering this query.
Do not include columns that are not needed to answer the query.

Relevant Columns:"""
            
            response = self.llm_engine._call_llm(prompt)
            
            # Extract column names from the response
            columns = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # Remove any list markers
                if line.startswith("- "):
                    line = line[2:]
                if line.startswith("* "):
                    line = line[2:]
                if line[0].isdigit() and line[1:].startswith(". "):
                    line = line[line.find(". ") + 2:]
                
                # Add to the list
                if line in table_columns:
                    columns.append(line)
            
            # Add to the result
            if columns:
                relevant_columns[table_name] = columns
        
        return relevant_columns


class LLMSQLGenerationAgent(SQLGenerationAgent):
    """
    LLM-based implementation of the SQLGenerationAgent.
    
    Uses an LLM to generate SQL queries based on natural language and schema information.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        db_manager: DatabaseManager,
        name: str = "LLMSQLGeneration", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM SQL generation agent.
        
        Args:
            llm_engine: LLM engine instance
            db_manager: Database manager instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
        self.db_manager = db_manager
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Generating SQL query from natural language")
        
        # Generate SQL
        start_time = time.time()
        sql_query, sql_params = self.generate_sql(context)
        context.sql_query = sql_query
        context.sql_params = sql_params
        context.execution_time["sql_generation"] = time.time() - start_time
        
        self.log_reasoning(context, f"Generated SQL query: {sql_query}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def generate_sql(self, context: AgentContext) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an SQL query from the context.
        
        Args:
            context: Agent context
            
        Returns:
            Tuple containing:
            - Generated SQL query
            - Parameters for the query
        """
        # Get schema for relevant tables
        schema = self.db_manager.get_schema()
        filtered_schema = {}
        
        for table_name in context.relevant_tables:
            # Parse schema and table name
            if "." in table_name:
                schema_name, table = table_name.split(".", 1)
            else:
                schema_name = "public"
                table = table_name
            
            # Check if table exists
            if schema_name not in schema or table not in schema[schema_name]:
                continue
            
            # Initialize schema if needed
            if schema_name not in filtered_schema:
                filtered_schema[schema_name] = {}
            
            # Add table to filtered schema
            filtered_schema[schema_name][table] = schema[schema_name][table]
        
        # Format the schema for the prompt
        schema_str = self.llm_engine._format_schema_for_prompt(filtered_schema)
        
        # Create additional context from the relevant columns
        columns_context = ""
        if context.relevant_columns:
            columns_list = []
            for table, columns in context.relevant_columns.items():
                columns_list.append(f"{table}: {', '.join(columns)}")
            
            columns_context = "Relevant columns:\n" + "\n".join(columns_list)
        
        # Create the prompt
        prompt = f"""You are an expert SQL developer.
Generate an SQL query for this request using the provided database schema.

Query: "{context.user_query}"

Intent: {context.query_intent}

Database Type: {self.db_manager.get_database_type()}

Database Schema:
{schema_str}

{columns_context}

Guidelines:
1. Only use tables and columns that are in the provided schema.
2. Use appropriate JOIN operations when involving multiple tables.
3. Handle NULL values appropriately.
4. Use clear aliases for tables and columns when needed.
5. For aggregate queries, include appropriate GROUP BY clauses.
6. Focus on creating an efficient, readable query.
7. Wrap the final SQL in a ```sql code block.

SQL Query:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract the SQL query from the response
        sql_query = self.llm_engine._extract_sql_from_response(response)
        
        # For now, we don't support parameterized queries
        sql_params = {}
        
        return sql_query, sql_params


class LLMQueryValidationAgent(QueryValidationAgent):
    """
    LLM-based implementation of the QueryValidationAgent.
    
    Uses an LLM to validate and fix SQL queries.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        db_manager: DatabaseManager,
        name: str = "LLMQueryValidation", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM query validation agent.
        
        Args:
            llm_engine: LLM engine instance
            db_manager: Database manager instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
        self.db_manager = db_manager
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Validating generated SQL query")
        
        # Validate query
        start_time = time.time()
        is_valid, error = self.validate_query(context)
        context.execution_time["query_validation"] = time.time() - start_time
        
        if is_valid:
            self.log_reasoning(context, "SQL query is valid")
        else:
            self.log_reasoning(context, f"SQL query is invalid: {error}")
            
            # Fix the query
            start_time = time.time()
            fixed_query = self.fix_query(context)
            context.sql_query = fixed_query
            context.execution_time["query_fixing"] = time.time() - start_time
            
            self.log_reasoning(context, f"Fixed SQL query: {fixed_query}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def validate_query(self, context: AgentContext) -> Tuple[bool, Optional[str]]:
        """
        Validate an SQL query.
        
        Args:
            context: Agent context
            
        Returns:
            Tuple containing:
            - Boolean indicating if the query is valid
            - Error message if invalid, None otherwise
        """
        return self.db_manager.validate_query(context.sql_query)
    
    def fix_query(self, context: AgentContext) -> str:
        """
        Fix an invalid SQL query.
        
        Args:
            context: Agent context
            
        Returns:
            Fixed SQL query
        """
        # Get schema for relevant tables
        schema = self.db_manager.get_schema()
        filtered_schema = {}
        
        for table_name in context.relevant_tables:
            # Parse schema and table name
            if "." in table_name:
                schema_name, table = table_name.split(".", 1)
            else:
                schema_name = "public"
                table = table_name
            
            # Check if table exists
            if schema_name not in schema or table not in schema[schema_name]:
                continue
            
            # Initialize schema if needed
            if schema_name not in filtered_schema:
                filtered_schema[schema_name] = {}
            
            # Add table to filtered schema
            filtered_schema[schema_name][table] = schema[schema_name][table]
        
        # Format the schema for the prompt
        schema_str = self.llm_engine._format_schema_for_prompt(filtered_schema)
        
        # Create the prompt
        prompt = f"""You are an expert SQL developer.
Fix this invalid SQL query to make it work correctly.

Original Query: "{context.user_query}"

Generated SQL:
```sql
{context.sql_query}
```

Error: {context.result_error}

Database Type: {self.db_manager.get_database_type()}

Database Schema:
{schema_str}

Guidelines:
1. Only use tables and columns that are in the provided schema.
2. Use appropriate JOIN operations when involving multiple tables.
3. Handle NULL values appropriately.
4. Use clear aliases for tables and columns when needed.
5. For aggregate queries, include appropriate GROUP BY clauses.
6. Focus on creating an efficient, readable query.
7. Wrap the final SQL in a ```sql code block.

Fixed SQL Query:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract the SQL query from the response
        fixed_query = self.llm_engine._extract_sql_from_response(response)
        
        return fixed_query


class LLMResultExplanationAgent(ResultExplanationAgent):
    """
    LLM-based implementation of the ResultExplanationAgent.
    
    Uses an LLM to explain query results in natural language.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        name: str = "LLMResultExplanation", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM result explanation agent.
        
        Args:
            llm_engine: LLM engine instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Explaining query results")
        
        # Check if there are results to explain
        if not context.query_results:
            explanation = "No results were returned by the query."
            self.add_explanation(context, "results", explanation)
            self.log_reasoning(context, explanation)
        else:
            # Generate explanation
            start_time = time.time()
            explanation = self.explain_results(context)
            context.execution_time["result_explanation"] = time.time() - start_time
            
            self.add_explanation(context, "results", explanation)
            self.log_reasoning(context, f"Explanation generated: {explanation}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def explain_results(self, context: AgentContext) -> str:
        """
        Generate a natural language explanation of the query results.
        
        Args:
            context: Agent context
            
        Returns:
            Natural language explanation
        """
        # Prepare the results for the prompt
        max_results = 10  # Limit to avoid token limits
        result_str = ""
        
        if len(context.query_results) <= max_results:
            # Format all results
            result_str = "\n".join([str(row) for row in context.query_results])
        else:
            # Format a subset of results
            first_results = "\n".join([str(row) for row in context.query_results[:5]])
            last_results = "\n".join([str(row) for row in context.query_results[-5:]])
            result_str = f"{first_results}\n...\n[{len(context.query_results) - 10} more rows]\n...\n{last_results}"
        
        # Create the prompt
        prompt = f"""You are an expert data analyst.
Explain these query results in a clear, concise way:

Original Query: "{context.user_query}"

SQL Query:
```sql
{context.sql_query}
```

Query Results:
{result_str}

Explain what these results mean in relation to the original query.
Be specific about key findings, patterns, or insights.
Keep the explanation concise (3-5 sentences).

Explanation:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Clean up the response
        explanation = response.strip()
        
        # Remove any prefixes
        for prefix in ["Explanation:", "The results show:", "Based on the results:"]:
            if explanation.startswith(prefix):
                explanation = explanation[len(prefix):].strip()
        
        return explanation


class LLMVisualizationAgent(VisualizationAgent):
    """
    LLM-based implementation of the VisualizationAgent.
    
    Uses an LLM to suggest appropriate visualizations for query results.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        name: str = "LLMVisualization", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM visualization agent.
        
        Args:
            llm_engine: LLM engine instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Suggesting visualization for query results")
        
        # Check if there are results to visualize
        if not context.query_results:
            self.log_reasoning(context, "No results to visualize")
        else:
            # Suggest visualization
            start_time = time.time()
            viz_config = self.suggest_visualization(context)
            context.execution_time["visualization_suggestion"] = time.time() - start_time
            
            # Add to context
            context.metadata["visualization"] = viz_config
            
            self.log_reasoning(context, f"Suggested visualization type: {viz_config.get('type', 'unknown')}")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def suggest_visualization(self, context: AgentContext) -> Dict[str, Any]:
        """
        Suggest an appropriate visualization for the query results.
        
        Args:
            context: Agent context
            
        Returns:
            Dictionary with visualization configuration
        """
        # Prepare the results for the prompt
        max_results = 10  # Limit to avoid token limits
        result_str = ""
        
        if len(context.query_results) <= max_results:
            # Format all results
            result_str = "\n".join([str(row) for row in context.query_results])
        else:
            # Format a subset of results
            first_results = "\n".join([str(row) for row in context.query_results[:5]])
            last_results = "\n".join([str(row) for row in context.query_results[-5:]])
            result_str = f"{first_results}\n...\n[{len(context.query_results) - 10} more rows]\n...\n{last_results}"
        
        # Extract column names from the first result
        if context.query_results:
            columns = list(context.query_results[0].keys())
            columns_str = ", ".join(columns)
        else:
            columns = []
            columns_str = "No columns available"
        
        # Create the prompt
        prompt = f"""You are an expert data visualization specialist.
Suggest the most appropriate visualization for these query results:

Original Query: "{context.user_query}"

SQL Query:
```sql
{context.sql_query}
```

Available Columns: {columns_str}

Query Results:
{result_str}

Available Visualization Types:
- Table: Simple tabular display
- Bar Chart: Compare values across categories
- Line Chart: Show trends over time or sequence
- Scatter Plot: Show relationship between two variables
- Heatmap: Show patterns in a matrix of values

For the chosen visualization, specify:
1. Visualization type
2. X-axis column (if applicable)
3. Y-axis column (if applicable) 
4. Color column (if applicable)
5. Aggregation function (if applicable): Count, Sum, Average, Min, Max

Format your response as a JSON object with these keys:
{{
  "type": "...",
  "x_column": "...",
  "y_column": "...",
  "color_column": "...",
  "aggregation": "..."
}}

Only include key-value pairs that are applicable to the chosen visualization type.

JSON:"""
        
        response = self.llm_engine._call_llm(prompt)
        
        # Extract the JSON from the response
        import json
        
        # Find JSON in the response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            
            try:
                # Parse the JSON
                viz_config = json.loads(json_str)
                
                # Validate the config
                valid_types = ["Table", "Bar Chart", "Line Chart", "Scatter Plot", "Heatmap"]
                valid_aggs = ["None", "Count", "Sum", "Average", "Min", "Max"]
                
                # Ensure type is valid
                if "type" not in viz_config or viz_config["type"] not in valid_types:
                    viz_config["type"] = "Table"
                
                # Ensure columns exist
                for key in ["x_column", "y_column", "color_column"]:
                    if key in viz_config and viz_config[key] not in columns and viz_config[key] != "None":
                        viz_config[key] = columns[0] if columns else None
                
                # Ensure aggregation is valid
                if "aggregation" in viz_config and viz_config["aggregation"] not in valid_aggs:
                    viz_config["aggregation"] = "None"
                
                return viz_config
                
            except json.JSONDecodeError:
                # Fallback to table
                return {"type": "Table"}
        
        # Fallback to table
        return {"type": "Table"}


class SimpleCoordinatorAgent(CoordinatorAgent):
    """
    Simple implementation of the CoordinatorAgent.
    
    Defines a fixed pipeline of agents to process queries.
    """
    
    def __init__(
        self, 
        name: str = "SimpleCoordinator", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the simple coordinator agent.
        
        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Define the pipeline order
        self.pipeline = [
            "LLMQueryUnderstanding",
            "LLMSchemaAnalysis",
            "LLMSQLGeneration",
            "LLMQueryValidation",
            "LLMResultExplanation",
            "LLMVisualization"
        ]
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context by coordinating other agents.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Starting query processing pipeline")
        
        # Reset the context for a new query
        context.query_intent = ""
        context.query_entities = []
        context.sql_query = ""
        context.sql_params = {}
        context.query_results = []
        context.result_error = None
        context.relevant_tables = []
        context.relevant_columns = {}
        context.reasoning_steps = []
        context.explanations = {}
        context.current_agent = None
        context.agent_history = []
        context.iterations = 0
        context.confidence = 0.0
        context.execution_time = {}
        context.metadata = {}
        
        # Process the query through the pipeline
        start_time = time.time()
        
        while context.iterations < context.max_iterations:
            # Decide which agent to invoke next
            next_agent = self.decide_next_agent(context)
            
            if next_agent is None:
                self.log_reasoning(context, "Pipeline completed")
                break
            
            if next_agent not in self.agents:
                self.log_reasoning(context, f"Agent {next_agent} not found, skipping")
                continue
            
            # Invoke the agent
            agent = self.agents[next_agent]
            self.log_reasoning(context, f"Invoking agent: {agent.name}")
            
            # Process with the agent
            agent_start_time = time.time()
            context = agent.process(context)
            agent_time = time.time() - agent_start_time
            
            self.log_reasoning(context, f"Agent {agent.name} completed in {agent_time:.2f} seconds")
            
            # Increment iteration counter
            context.iterations += 1
        
        # Record total processing time
        context.execution_time["total"] = time.time() - start_time
        
        self.log_reasoning(context, f"Pipeline finished after {context.iterations} iterations in {context.execution_time['total']:.2f} seconds")
        
        # Update current agent
        context.current_agent = self.name
        context.agent_history.append(self.name)
        
        return context
    
    def decide_next_agent(self, context: AgentContext) -> Optional[str]:
        """
        Decide which agent to invoke next.
        
        Args:
            context: Agent context
            
        Returns:
            Name of the next agent to invoke, or None if done
        """
        # If no agents have been invoked yet, start at the beginning
        if not context.agent_history:
            return self.pipeline[0]
        
        # Get the last agent that was invoked
        last_agent = context.agent_history[-1]
        
        # Find the next agent in the pipeline
        try:
            last_index = self.pipeline.index(last_agent)
            next_index = last_index + 1
            
            if next_index < len(self.pipeline):
                return self.pipeline[next_index]
            else:
                return None  # End of pipeline
                
        except ValueError:
            # Last agent not in pipeline, start from beginning
            return self.pipeline[0]