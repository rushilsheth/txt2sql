"""
Semantic Module for LLM Engine

This module extends the LLM engine with semantic understanding capabilities
for database schemas and natural language queries.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from text_to_sql.db.base import DatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.llm.prompts import SCHEMA_SEMANTICS_PROMPT, SEMANTIC_QUERY_PLAN_PROMPT

logger = logging.getLogger(__name__)

class SemanticEngine:
    """
    Engine for semantic understanding of database schemas and queries.
    
    This class extends the LLM engine with capabilities for understanding
    the semantic meaning of database schemas and natural language queries.
    """
    
    def __init__(
        self,
        llm_engine: Optional[LLMEngine] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the semantic engine.
        
        Args:
            llm_engine: LLM engine instance
            db_manager: Database manager instance
        """
        self.llm_engine = llm_engine
        self.db_manager = db_manager
        self.schema_semantics = None
    
    def set_llm_engine(self, llm_engine: LLMEngine):
        """
        Set the LLM engine.
        
        Args:
            llm_engine: LLM engine instance
        """
        self.llm_engine = llm_engine
    
    def set_db_manager(self, db_manager: DatabaseManager):
        """
        Set the database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        
        # Reset schema semantics when database changes
        self.schema_semantics = None
    
    def analyze_schema_semantics(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze the semantic meaning of the database schema.
        
        Args:
            refresh: Whether to refresh the analysis
            
        Returns:
            Dictionary containing semantic information about the schema
        """
        if self.schema_semantics is not None and not refresh:
            return self.schema_semantics
        
        if not self.db_manager:
            raise ValueError("Database manager not set")
        
        if not self.llm_engine:
            raise ValueError("LLM engine not set")
        
        # Get the database schema
        schema = self.db_manager.get_schema()
        
        # Format the schema for the prompt
        schema_str = self.llm_engine._format_schema_for_prompt(schema)
        
        # Create the prompt
        prompt = SCHEMA_SEMANTICS_PROMPT.format(schema=schema_str)
        
        # Call the LLM
        response = self.llm_engine._call_llm(prompt)
        
        # Parse the semantic information
        semantics = self._parse_semantic_information(response)
        
        # Cache the semantics
        self.schema_semantics = semantics
        
        return semantics
    
    def generate_semantic_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Generate a semantic query plan for a natural language question.
        
        Args:
            query: Natural language question
            
        Returns:
            Dictionary containing the semantic query plan
        """
        if not self.db_manager:
            raise ValueError("Database manager not set")
        
        if not self.llm_engine:
            raise ValueError("LLM engine not set")
        
        # Get the database schema
        schema = self.db_manager.get_schema()
        
        # Format the schema for the prompt
        schema_str = self.llm_engine._format_schema_for_prompt(schema)
        
        # Create the prompt
        prompt = SEMANTIC_QUERY_PLAN_PROMPT.format(
            schema=schema_str,
            question=query
        )
        
        # Call the LLM
        response = self.llm_engine._call_llm(prompt)
        
        # Parse the query plan
        query_plan = self._parse_query_plan(response)
        
        return query_plan
    
    def _parse_semantic_information(self, response: str) -> Dict[str, Any]:
        """
        Parse semantic information from the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary containing structured semantic information
        """
        # Initialize the semantic information
        semantics = {
            "entities": [],
            "relationships": [],
            "hierarchies": [],
            "business_rules": [],
            "naming_patterns": [],
            "metrics_dimensions": [],
            "raw_response": response
        }
        
        # Extract entities
        entities_section = self._extract_section(response, "entities", ["relationships", "hierarchies", "business rules", "naming patterns", "metrics"])
        if entities_section:
            semantics["entities"] = self._extract_list_items(entities_section)
        
        # Extract relationships
        relationships_section = self._extract_section(response, "relationships", ["hierarchies", "business rules", "naming patterns", "metrics"])
        if relationships_section:
            semantics["relationships"] = self._extract_list_items(relationships_section)
        
        # Extract hierarchies
        hierarchies_section = self._extract_section(response, "hierarchies", ["business rules", "naming patterns", "metrics"])
        if hierarchies_section:
            semantics["hierarchies"] = self._extract_list_items(hierarchies_section)
        
        # Extract business rules
        business_rules_section = self._extract_section(response, "business rules", ["naming patterns", "metrics"])
        if business_rules_section:
            semantics["business_rules"] = self._extract_list_items(business_rules_section)
        
        # Extract naming patterns
        naming_patterns_section = self._extract_section(response, "naming patterns", ["metrics"])
        if naming_patterns_section:
            semantics["naming_patterns"] = self._extract_list_items(naming_patterns_section)
        
        # Extract metrics/dimensions
        metrics_section = self._extract_section(response, "metrics", [])
        if metrics_section:
            semantics["metrics_dimensions"] = self._extract_list_items(metrics_section)
        
        return semantics
    
    def _parse_query_plan(self, response: str) -> Dict[str, Any]:
        """
        Parse query plan from the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary containing the structured query plan
        """
        # Initialize the query plan
        query_plan = {
            "entities": [],
            "relationships": [],
            "filters": [],
            "aggregations": [],
            "output_columns": [],
            "raw_response": response
        }
        
        # Extract entities
        entities_section = self._extract_section(response, "entities", ["relationships", "filtering", "aggregations", "output"])
        if entities_section:
            query_plan["entities"] = self._extract_list_items(entities_section)
        
        # Extract relationships
        relationships_section = self._extract_section(response, "relationships", ["filtering", "aggregations", "output"])
        if relationships_section:
            query_plan["relationships"] = self._extract_list_items(relationships_section)
        
        # Extract filters
        filters_section = self._extract_section(response, "filtering", ["aggregations", "output"])
        if filters_section:
            query_plan["filters"] = self._extract_list_items(filters_section)
        
        # Extract aggregations
        aggregations_section = self._extract_section(response, "aggregations", ["output"])
        if aggregations_section:
            query_plan["aggregations"] = self._extract_list_items(aggregations_section)
        
        # Extract output columns
        output_section = self._extract_section(response, "output", [])
        if output_section:
            query_plan["output_columns"] = self._extract_list_items(output_section)
        
        return query_plan
    
    def _extract_section(self, text: str, section_name: str, next_sections: List[str]) -> str:
        """
        Extract a section from text based on keywords.
        
        Args:
            text: The text to extract from
            section_name: The name of the section to extract
            next_sections: Names of sections that might follow
            
        Returns:
            The extracted section text
        """
        # Create pattern for the current section
        pattern = f"(?i){section_name}"
        
        # Find the start of the section
        start_match = None
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in [section_name.lower(), f"{section_name.lower()}:"]):
                start_match = line
                break
        
        if not start_match:
            return ""
        
        # Get the start index
        start_idx = text.find(start_match)
        if start_idx == -1:
            return ""
        
        # Find the end of the section (start of the next section)
        end_idx = len(text)
        for next_section in next_sections:
            for line in text[start_idx:].split('\n'):
                if any(keyword in line.lower() for keyword in [next_section.lower(), f"{next_section.lower()}:"]):
                    section_idx = text[start_idx:].find(line)
                    if section_idx != -1:
                        section_idx += start_idx
                        if section_idx < end_idx:
                            end_idx = section_idx
                    break
        
        # Extract the section
        section = text[start_idx:end_idx].strip()
        
        # Remove the section title
        lines = section.split('\n')
        if lines:
            section = '\n'.join(lines[1:]).strip()
        
        return section
    
    def _extract_list_items(self, text: str) -> List[str]:
        """
        Extract list items from text.
        
        Args:
            text: The text to extract from
            
        Returns:
            List of extracted items
        """
        items = []
        
        # Check for numbered list (1. Item)
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for numbered list (1. Item)
            if line[0].isdigit() and line[1:].startswith('. '):
                items.append(line[line.find('. ') + 2:].strip())
                continue
                
            # Check for bullet points
            if line.startswith('- ') or line.startswith('* '):
                items.append(line[2:].strip())
                continue
                
            # Check for Markdown style (. Item)
            if line.startswith('. '):
                items.append(line[2:].strip())
                continue
                
            # If no list marker but previous line had one, treat as continuation
            if items and not line.startswith(('- ', '* ')) and not (line[0].isdigit() and line[1:].startswith('. ')):
                items[-1] += ' ' + line
                continue
                
            # Otherwise, add as a new item if it's not part of a list
            if not items:
                items.append(line)
        
        return items