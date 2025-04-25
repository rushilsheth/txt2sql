"""
PostgreSQL Database Manager Module

This module implements the DatabaseManager interface for PostgreSQL databases.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from text_to_sql.db.base import DatabaseManager

logger = logging.getLogger(__name__)

class PostgresDatabaseManager(DatabaseManager):
    """
    PostgreSQL implementation of the DatabaseManager.
    
    This class provides methods for connecting to a PostgreSQL database,
    executing queries, and introspecting the schema.
    """
    
    def __init__(self, connection_params: Dict[str, Any] = None):
        """
        Initialize the PostgreSQL database manager.
        
        Args:
            connection_params: A dictionary of connection parameters for PostgreSQL
                - host: Database server host
                - port: Database server port
                - dbname: Database name
                - user: Username
                - password: Password
                - sslmode: SSL mode
                - min_connections: Minimum number of connections in the pool
                - max_connections: Maximum number of connections in the pool
        """
        super().__init__(connection_params)
        self.pool = None
    
    def connect(self) -> bool:
        """
        Establish a connection pool to the PostgreSQL database.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Extract connection parameters
            min_conn = self.connection_params.get('min_connections', 1)
            max_conn = self.connection_params.get('max_connections', 5)
            
            # Create connection dictionary for psycopg2
            db_params = {
                'host': self.connection_params.get('host', 'localhost'),
                'port': self.connection_params.get('port', 5432),
                'dbname': self.connection_params.get('dbname', 'postgres'),
                'user': self.connection_params.get('user', 'postgres'),
                'password': self.connection_params.get('password', ''),
                'sslmode': self.connection_params.get('sslmode', 'prefer')
            }
            
            # Create a connection pool
            self.pool = pool.ThreadedConnectionPool(
                min_conn, max_conn,
                **db_params
            )
            
            self.connected = True
            logger.info(f"Connected to PostgreSQL database: {db_params.get('dbname')}@{db_params.get('host')}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Close the PostgreSQL connection pool.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        try:
            if self.pool:
                self.pool.closeall()
                self.pool = None
            
            self.connected = False
            logger.info("Disconnected from PostgreSQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL database: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Execute an SQL query on the PostgreSQL database and return the results.
        
        Args:
            query: The SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            Tuple containing:
            - List of dictionaries, each representing a row of results
            - Error message if an error occurred, None otherwise
        """
        conn = None
        cursor = None
        results = []
        error = None
        
        try:
            # Get a connection from the pool
            conn = self.pool.getconn()
            
            # Create a cursor that returns dictionaries
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Execute the query
            start_time = time.time()
            cursor.execute(query, params or {})
            execution_time = time.time() - start_time
            
            # Fetch all results
            results = cursor.fetchall()
            logger.info(f"Query executed in {execution_time:.3f} seconds, returned {len(results)} rows")
            
            # Convert from RealDictRow to regular dictionaries
            results = [dict(row) for row in results]
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error executing query: {e}")
            
        finally:
            # Close cursor
            if cursor:
                cursor.close()
            
            # Return connection to the pool
            if conn:
                self.pool.putconn(conn)
        
        return results, error
    
    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve the PostgreSQL database schema.
        
        Args:
            refresh: Whether to refresh the schema cache
            
        Returns:
            Dictionary representing the database schema
        """
        if self.schema_cache is not None and not refresh:
            return self.schema_cache
        
        schema_query = """
        SELECT 
            n.nspname AS schema_name,
            c.relname AS table_name,
            a.attname AS column_name,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
            CASE 
                WHEN a.attnotnull THEN true 
                ELSE false 
            END AS not_null,
            CASE 
                WHEN co.contype = 'p' THEN true 
                ELSE false 
            END AS is_primary_key,
            CASE 
                WHEN co.contype = 'f' THEN true 
                ELSE false 
            END AS is_foreign_key,
            obj_description(c.oid) AS table_description,
            col_description(c.oid, a.attnum) AS column_description,
            pg_catalog.pg_get_constraintdef(co.oid, true) AS constraint_def
        FROM 
            pg_catalog.pg_attribute a
        JOIN 
            pg_catalog.pg_class c ON a.attrelid = c.oid
        JOIN 
            pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        LEFT JOIN 
            pg_catalog.pg_constraint co ON (
                co.conrelid = c.oid 
                AND a.attnum = ANY(co.conkey) 
                AND (co.contype = 'p' OR co.contype = 'f')
            )
        WHERE 
            c.relkind = 'r' 
            AND a.attnum > 0 
            AND NOT a.attisdropped
            AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 
            n.nspname, c.relname, a.attnum;
        """
        
        results, error = self.execute_query(schema_query)
        
        if error:
            logger.error(f"Error retrieving schema: {error}")
            return {}
        
        # Process the results into a structured schema
        schema = {}
        
        for row in results:
            schema_name = row['schema_name']
            table_name = row['table_name']
            column_name = row['column_name']
            
            # Initialize schema if needed
            if schema_name not in schema:
                schema[schema_name] = {}
            
            # Initialize table if needed
            if table_name not in schema[schema_name]:
                schema[schema_name][table_name] = {
                    'description': row['table_description'] or '',
                    'columns': {}
                }
            
            # Add column information
            schema[schema_name][table_name]['columns'][column_name] = {
                'data_type': row['data_type'],
                'not_null': row['not_null'],
                'is_primary_key': row['is_primary_key'],
                'is_foreign_key': row['is_foreign_key'],
                'description': row['column_description'] or '',
                'constraint': row['constraint_def'] if row['is_primary_key'] or row['is_foreign_key'] else None
            }
        
        # Cache the schema
        self.schema_cache = schema
        
        return schema
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific PostgreSQL table.
        
        Args:
            table_name: The name of the table (can include schema as schema.table)
            
        Returns:
            Dictionary containing table information
        """
        # Parse schema and table name
        parts = table_name.split('.')
        if len(parts) == 1:
            schema_name = 'public'
            table_name = parts[0]
        else:
            schema_name = parts[0]
            table_name = parts[1]
        
        # Ensure schema is loaded
        schema = self.get_schema()
        
        # Return table info if found
        if schema_name in schema and table_name in schema[schema_name]:
            return schema[schema_name][table_name]
        
        logger.warning(f"Table {schema_name}.{table_name} not found in schema")
        return {}
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of data from a PostgreSQL table.
        
        Args:
            table_name: The name of the table (can include schema as schema.table)
            limit: Maximum number of rows to return
            
        Returns:
            List of dictionaries, each representing a row from the table
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        results, error = self.execute_query(query)
        
        if error:
            logger.error(f"Error retrieving sample from {table_name}: {error}")
            return []
        
        return results
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a PostgreSQL query without executing it.
        
        Args:
            query: The SQL query to validate
            
        Returns:
            Tuple containing:
            - Boolean indicating if the query is valid
            - Error message if invalid, None otherwise
        """
        conn = None
        cursor = None
        
        try:
            # Get a connection from the pool
            conn = self.pool.getconn()
            
            # Create a cursor
            cursor = conn.cursor()
            
            # Start a transaction
            conn.autocommit = False
            
            # Parse the query without executing it
            cursor.execute(f"EXPLAIN {query}")
            
            # If we get here, the query is valid
            valid = True
            error = None
            
            # Rollback the transaction
            conn.rollback()
            
        except Exception as e:
            valid = False
            error = str(e)
            
            # Rollback the transaction
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            
        finally:
            # Close cursor
            if cursor:
                cursor.close()
            
            # Return connection to the pool
            if conn:
                self.pool.putconn(conn)
        
        return valid, error