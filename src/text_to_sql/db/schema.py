"""
Schema Module

This module provides utilities for analyzing and working with database schemas.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class SchemaAnalyzer:
    """
    Utility class for analyzing database schemas.
    
    This class provides methods for analyzing database schemas,
    detecting relationships, and identifying key entities.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the schema analyzer.
        
        Args:
            schema: Database schema from DatabaseManager.get_schema()
        """
        self.schema = schema
        self.relationships = []
        self.primary_entities = []
        self.reference_tables = []
        
        # Perform analysis
        self._analyze()
    
    def _analyze(self):
        """Analyze the schema to extract relationships and entities."""
        self._extract_relationships()
        self._identify_primary_entities()
        self._identify_reference_tables()
    
    def _extract_relationships(self):
        """Extract relationships between tables based on foreign keys."""
        relationships = []
        
        for schema_name, tables in self.schema.items():
            for table_name, table_info in tables.items():
                for column_name, column_info in table_info['columns'].items():
                    if column_info['is_foreign_key'] and column_info['constraint']:
                        # Extract referenced table from constraint
                        referenced_table = self._extract_referenced_table(column_info['constraint'])
                        
                        if referenced_table:
                            relationship = {
                                'source_schema': schema_name,
                                'source_table': table_name,
                                'source_column': column_name,
                                'target_table': referenced_table,
                                'relation_type': 'many-to-one'
                            }
                            relationships.append(relationship)
        
        self.relationships = relationships
    
    def _extract_referenced_table(self, constraint: str) -> Optional[str]:
        """
        Extract the referenced table from a foreign key constraint.
        
        Args:
            constraint: Foreign key constraint definition
            
        Returns:
            Referenced table name or None if not found
        """
        # Pattern to match the referenced table in a constraint
        pattern = r"REFERENCES\s+([^\s\(]+)"
        match = re.search(pattern, constraint, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return None
    
    def _identify_primary_entities(self):
        """Identify primary entities in the schema."""
        # Count incoming relationships for each table
        incoming_relationships = {}
        
        for rel in self.relationships:
            target = rel['target_table']
            if target not in incoming_relationships:
                incoming_relationships[target] = 0
            
            incoming_relationships[target] += 1
        
        # Tables with many incoming relationships are likely to be primary entities
        primary_entities = []
        
        for table, count in incoming_relationships.items():
            if count >= 2:  # At least 2 other tables reference this one
                primary_entities.append({
                    'table': table,
                    'reference_count': count
                })
        
        # Sort by reference count (descending)
        primary_entities.sort(key=lambda x: x['reference_count'], reverse=True)
        
        self.primary_entities = primary_entities
    
    def _identify_reference_tables(self):
        """Identify reference tables (lookup tables) in the schema."""
        reference_tables = []
        
        for schema_name, tables in self.schema.items():
            for table_name, table_info in tables.items():
                # Check if this might be a reference table
                column_count = len(table_info['columns'])
                
                # Reference tables typically have few columns (2-4)
                if 2 <= column_count <= 4:
                    # Check if it has a primary key and a descriptive column
                    has_primary_key = False
                    has_name_column = False
                    
                    for column_name, column_info in table_info['columns'].items():
                        if column_info['is_primary_key']:
                            has_primary_key = True
                        
                        # Check if column name suggests it's a descriptive field
                        if any(name in column_name.lower() for name in ['name', 'description', 'title', 'label']):
                            has_name_column = True
                    
                    if has_primary_key and has_name_column:
                        reference_tables.append({
                            'schema': schema_name,
                            'table': table_name
                        })
        
        self.reference_tables = reference_tables
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific table.
        
        Args:
            table_name: Table name to get relationships for
            
        Returns:
            List of relationships where the table is the source or target
        """
        relationships = []
        
        for rel in self.relationships:
            if rel['source_table'] == table_name or rel['target_table'] == table_name:
                relationships.append(rel)
        
        return relationships
    
    def get_join_path(self, source_table: str, target_table: str) -> List[Dict[str, Any]]:
        """
        Get a join path between two tables.
        
        Args:
            source_table: Source table name
            target_table: Target table name
            
        Returns:
            List of relationships representing the join path
        """
        # Simple BFS to find the shortest path
        visited = set()
        queue = [(source_table, [])]
        
        while queue:
            current_table, path = queue.pop(0)
            
            if current_table == target_table:
                return path
            
            if current_table in visited:
                continue
            
            visited.add(current_table)
            
            # Add all neighboring tables
            for rel in self.relationships:
                if rel['source_table'] == current_table and rel['target_table'] not in visited:
                    new_path = path + [rel]
                    queue.append((rel['target_table'], new_path))
                elif rel['target_table'] == current_table and rel['source_table'] not in visited:
                    # Reverse the relationship
                    reversed_rel = {
                        'source_schema': rel.get('source_schema'),
                        'source_table': rel['target_table'],
                        'source_column': self._find_pk_for_table(rel['target_table']),
                        'target_table': rel['source_table'],
                        'target_column': rel['source_column'],
                        'relation_type': 'one-to-many'  # Reversed from many-to-one
                    }
                    new_path = path + [reversed_rel]
                    queue.append((rel['source_table'], new_path))
        
        # No path found
        return []
    
    def _find_pk_for_table(self, table_name: str) -> Optional[str]:
        """
        Find the primary key column for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Primary key column name or None if not found
        """
        for schema_name, tables in self.schema.items():
            if table_name in tables:
                for column_name, column_info in tables[table_name]['columns'].items():
                    if column_info['is_primary_key']:
                        return column_name
        
        return None
    
    def get_column_samples(self, table_name: str, db_manager) -> Dict[str, List[Any]]:
        """
        Get sample values for string columns in a table.
        
        Args:
            table_name: Table name
            db_manager: Database manager instance
            
        Returns:
            Dictionary mapping column names to sample values
        """
        samples = {}
        
        # Find the schema and table
        schema_name = None
        table_info = None
        
        for schema, tables in self.schema.items():
            if table_name in tables:
                schema_name = schema
                table_info = tables[table_name]
                break
        
        if not table_info:
            return samples
        
        # Get string columns
        string_columns = []
        
        for column_name, column_info in table_info['columns'].items():
            if 'char' in column_info['data_type'].lower() or 'text' in column_info['data_type'].lower():
                string_columns.append(column_name)
        
        # Get sample values for each column
        for column in string_columns:
            query = f"SELECT DISTINCT {column} FROM {schema_name}.{table_name} WHERE {column} IS NOT NULL LIMIT 10"
            results, error = db_manager.execute_query(query)
            
            if not error and results:
                samples[column] = [row[column] for row in results]
        
        return samples
    
    def suggest_query_tables(self, query_terms: List[str]) -> List[str]:
        """
        Suggest tables relevant to a natural language query.
        
        Args:
            query_terms: List of terms from the natural language query
            
        Returns:
            List of suggested table names
        """
        scores = {}
        
        # Initialize scores for all tables
        for schema_name, tables in self.schema.items():
            for table_name in tables:
                scores[f"{schema_name}.{table_name}"] = 0
        
        # Score tables based on term matches
        for term in query_terms:
            term = term.lower()
            
            for schema_name, tables in self.schema.items():
                for table_name, table_info in tables.items():
                    table_key = f"{schema_name}.{table_name}"
                    
                    # Match table name
                    if term in table_name.lower():
                        scores[table_key] += 2
                    
                    # Match table description
                    if table_info['description'] and term in table_info['description'].lower():
                        scores[table_key] += 1
                    
                    # Match column names and descriptions
                    for column_name, column_info in table_info['columns'].items():
                        if term in column_name.lower():
                            scores[table_key] += 1
                        
                        if column_info['description'] and term in column_info['description'].lower():
                            scores[table_key] += 0.5
        
        # Sort tables by score (descending)
        sorted_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-scoring tables (score > 0)
        return [table for table, score in sorted_tables if score > 0]