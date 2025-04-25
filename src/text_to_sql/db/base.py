"""
Base Database Manager Module

This module defines the abstract base class for database managers.
All database-specific implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class DatabaseManager(ABC):
    """
    Abstract base class for database connections and operations.
    
    This class defines the interface that all database implementations
    must follow. It provides methods for connecting to a database,
    executing queries, and introspecting the schema.
    """
    
    def __init__(self, connection_params: Dict[str, Any] = None):
        """
        Initialize the database manager with optional connection parameters.
        
        Args:
            connection_params: A dictionary of connection parameters
        """
        self.connection = None
        self.connection_params = connection_params or {}
        self.schema_cache = None
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish a connection to the database.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close the database connection.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Execute an SQL query and return the results.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            Tuple containing:
            - List of dictionaries, each representing a row of results
            - Error message if an error occurred, None otherwise
        """
        pass
    
    @abstractmethod
    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve the database schema.
        
        Args:
            refresh: Whether to refresh the schema cache
            
        Returns:
            Dictionary representing the database schema
        """
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: The name of the table
            
        Returns:
            Dictionary containing table information
        """
        pass
    
    @abstractmethod
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of data from a table.
        
        Args:
            table_name: The name of the table
            limit: Maximum number of rows to return
            
        Returns:
            List of dictionaries, each representing a row from the table
        """
        pass
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an SQL query without executing it.
        
        Args:
            query: The SQL query to validate
            
        Returns:
            Tuple containing:
            - Boolean indicating if the query is valid
            - Error message if invalid, None otherwise
        """
        # Default implementation - subclasses can override with 
        # database-specific validation
        try:
            # Simple validation - check if it's a SELECT query
            query = query.strip().lower()
            if not query.startswith('select'):
                return False, "Only SELECT queries are allowed"
            return True, None
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return False, str(e)
    
    def get_database_type(self) -> str:
        """
        Get the type of database being managed.
        
        Returns:
            String identifying the database type
        """
        return self.__class__.__name__.replace('DatabaseManager', '')
    
    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()