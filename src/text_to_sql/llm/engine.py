"""
LLM Engine Module

This module provides the core functionality for converting natural language queries to SQL
using large language models (LLMs).
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import openai

from text_to_sql.db.base import DatabaseManager
from text_to_sql.llm.prompts import SQL_GENERATION_PROMPT, SQL_VALIDATION_PROMPT

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Engine for converting natural language to SQL using LLMs.
    
    This class handles the interaction with language models to generate
    SQL queries from natural language questions.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_tokens: int = 1024,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the LLM engine.
        
        Args:
            model: The LLM model to use
            api_key: API key for the LLM provider
            temperature: Temperature parameter for the LLM
            timeout: Timeout for LLM API calls in seconds
            max_tokens: Maximum tokens in the LLM response
            db_manager: Database manager instance
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.db_manager = db_manager
        
        try:
            self.client = openai.OpenAI()
        except openai.OpenAIError as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    def set_db_manager(self, db_manager: DatabaseManager):
        """
        Set the database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    def generate_sql(self, query: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate an SQL query from a natural language question.
        
        Args:
            query: Natural language question
            
        Returns:
            Tuple containing:
            - Generated SQL query
            - Confidence score (0-1)
            - Additional metadata about the generation
        """
        if not self.db_manager:
            raise ValueError("Database manager not set. Please set it before generating SQL.")
        
        # Get database schema
        schema = self.db_manager.get_schema()
        
        # Format the schema for the prompt
        schema_str = self._format_schema_for_prompt(schema)
        
        # Create the prompt
        prompt = SQL_GENERATION_PROMPT.format(
            question=query,
            schema=schema_str,
            db_type=self.db_manager.get_database_type(),
        )
        
        # Call the LLM
        start_time = time.time()
        response = self.call_llm(prompt, json_response=True)
        generation_time = time.time() - start_time
        
        # Extract the SQL query from the response
        sql_query = self._extract_sql_from_response(response)
        
        # Validate the query
        is_valid, error = self.db_manager.validate_query(sql_query)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            is_valid=is_valid,
            error=error,
            response=response
        )
        
        # Prepare metadata
        metadata = {
            "generation_time": generation_time,
            "model": self.model,
            "is_valid": is_valid,
            "error": error,
            "raw_response": response
        }
        
        return sql_query, confidence, metadata
    
    def validate_and_repair_sql(self, sql_query: str, natural_query: str, error: Optional[str] = None) -> Tuple[str, bool]:
        """
        Validate and potentially repair an SQL query using the LLM.
        
        Args:
            sql_query: The SQL query to validate
            natural_query: The original natural language query
            error: Optional error message from previous validation
            
        Returns:
            Tuple containing:
            - Potentially repaired SQL query
            - Boolean indicating if the query was repaired
        """
        if not self.db_manager:
            raise ValueError("Database manager not set. Please set it before validating SQL.")
        
        # Get database schema
        schema = self.db_manager.get_schema()
        
        # Format the schema for the prompt
        schema_str = self._format_schema_for_prompt(schema)
        
        # Create the prompt
        prompt = SQL_VALIDATION_PROMPT.format(
            natural_query=natural_query,
            sql_query=sql_query,
            schema=schema_str,
            db_type=self.db_manager.get_database_type(),
            error=error or "No specific error message provided."
        )
        
        # Call the LLM
        response = self.call_llm(prompt, json_response=True)
        
        # Extract the repaired SQL query from the response
        repaired_sql = self._extract_sql_from_response(response)
        
        # Check if the query was actually repaired
        was_repaired = repaired_sql != sql_query
        
        return repaired_sql, was_repaired
    
    def _fallback_extract_sql_from_response(self, response: str) -> str:
        """
        Fallback method to extract SQL using original logic.
        """
        # Try to extract SQL from markdown code blocks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try to extract from generic code blocks
        code_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no code blocks, look for lines that might be SQL
        lines = response.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if (line.upper().startswith('SELECT') or
                line.upper().startswith('WITH') or
                line.upper().startswith('(SELECT')):
                sql_lines.append(line)
                continue
            # Add subsequent lines that might be part of the SQL query
            if sql_lines and (
                line.upper().startswith('FROM') or
                line.upper().startswith('WHERE') or
                line.upper().startswith('GROUP BY') or
                line.upper().startswith('ORDER BY') or
                line.upper().startswith('HAVING') or
                line.upper().startswith('LIMIT') or
                line.upper().startswith('OFFSET') or
                line.upper().startswith('UNION') or
                line.upper().startswith('JOIN') or
                line.upper().startswith('AND') or
                line.upper().startswith('OR') or
                line.upper().startswith('IN') or
                line.upper().startswith('ON') or
                ';' in line
            ):
                sql_lines.append(line)
        
        if sql_lines:
            return " ".join(sql_lines)
        
        logger.warning("Could not extract SQL from response, returning entire response")
        return response

    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL query from the LLM response.
        Attempts JSON parsing first before falling back to original logic.
        """
        try:
            parsed = json.loads(response)
            if 'sql' in parsed:
                return parsed['sql'].strip()
        except json.JSONDecodeError:
            logger.debug("Response is not valid JSON, using fallback extraction.")
        
        return self._fallback_extract_sql_from_response(response)

    def call_llm(self, prompt: str, json_response: bool = False) -> str:
        """
        Call the language model with a prompt, requesting a JSON response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response text
        """
        try:
            response_format = "json_object" if json_response else "text"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                response_format={"type": response_format}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise RuntimeError(f"Failed to generate SQL: {e}")
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """
        Format the database schema for inclusion in the prompt.
        
        Args:
            schema: The database schema
            
        Returns:
            Formatted schema string
        """
        formatted_schemas = []
        
        for schema_name, tables in schema.items():
            formatted_tables = []
            
            for table_name, table_info in tables.items():
                columns_info = []
                
                for column_name, column_info in table_info['columns'].items():
                    # Format column information
                    col_desc = [
                        f"{column_name} {column_info['data_type']}"
                    ]
                    
                    # Add constraints
                    if column_info['is_primary_key']:
                        col_desc.append("PRIMARY KEY")
                    
                    if column_info['not_null']:
                        col_desc.append("NOT NULL")
                    
                    if column_info['is_foreign_key'] and column_info['constraint']:
                        col_desc.append(f"FOREIGN KEY {column_info['constraint']}")
                    
                    # Add description if available
                    if column_info['description']:
                        col_desc.append(f"-- {column_info['description']}")
                    
                    columns_info.append(" ".join(col_desc))
                
                # Format table
                table_str = f"Table: {schema_name}.{table_name}\n"
                
                if table_info['description']:
                    table_str += f"Description: {table_info['description']}\n"
                
                table_str += "Columns:\n  " + "\n  ".join(columns_info)
                formatted_tables.append(table_str)
            
            formatted_schemas.append("\n\n".join(formatted_tables))
        
        return "\n\n".join(formatted_schemas)
    
    def _calculate_confidence(self, is_valid: bool, error: Optional[str], response: str) -> float:
        """
        Calculate a confidence score for the generated SQL.
        
        Args:
            is_valid: Whether the query is valid
            error: Error message if the query is invalid
            response: Raw LLM response
            
        Returns:
            Confidence score (0-1)
        """
        # Start with base confidence
        confidence = 0.7
        
        # Lower confidence if the query is invalid
        if not is_valid:
            confidence -= 0.4
            
            # Further lower based on error severity
            if error and ('syntax error' in error.lower() or 'parse error' in error.lower()):
                confidence -= 0.1
        
        # Check for indicators of uncertainty in the response
        uncertainty_phrases = [
            "i'm not sure",
            "i am not sure",
            "not certain",
            "might be",
            "could be",
            "not confident",
            "uncertain",
            "approximation"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase in response.lower():
                confidence -= 0.05
                # Don't let confidence go below 0.1
                if confidence < 0.1:
                    confidence = 0.1
                break
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        return confidence