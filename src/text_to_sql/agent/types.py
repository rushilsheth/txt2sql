"""
Agent Types Module

This module defines the types and interfaces for agents in the text-to-SQL system.
These agents are responsible for different aspects of the query processing pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Role of an agent in the text-to-SQL system."""
    
    # Core roles
    QUERY_UNDERSTANDING = "query_understanding"  # Understands the user's intent
    SCHEMA_ANALYSIS = "schema_analysis"          # Analyzes database schema
    SQL_GENERATION = "sql_generation"            # Generates SQL queries
    QUERY_VALIDATION = "query_validation"        # Validates and fixes queries
    RESULT_EXPLANATION = "result_explanation"    # Explains query results
    
    # Meta roles
    COORDINATOR = "coordinator"                  # Coordinates other agents
    DEBUG = "debug"                              # Provides debugging information
    
    # Optional roles
    VISUALIZATION = "visualization"              # Suggests visualizations
    OPTIMIZATION = "optimization"                # Optimizes queries


@dataclass
class AgentContext:
    """Context shared between agents."""
    
    # Original user input
    user_query: str = ""
    
    # Extracted intent and entities
    query_intent: str = ""
    query_entities: List[str] = field(default_factory=list)
    
    # Generated SQL
    sql_query: str = ""
    sql_params: Dict[str, Any] = field(default_factory=dict)
    
    # Query execution results
    query_results: List[Dict[str, Any]] = field(default_factory=list)
    result_error: Optional[str] = None
    
    # Schema information
    relevant_tables: List[str] = field(default_factory=list)
    relevant_columns: Dict[str, List[str]] = field(default_factory=dict)
    
    # Agent traces and explanations
    reasoning_steps: List[str] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    
    # State tracking
    current_agent: Optional[str] = None
    agent_history: List[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 5
    
    # Metadata
    confidence: float = 0.0
    execution_time: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all agents in the text-to-SQL system."""
    
    def __init__(self, name: str, role: AgentRole, config: Dict[str, Any] = None):
        """
        Initialize the agent.
        
        Args:
            name: Agent name
            role: Agent role
            config: Configuration dictionary
        """
        self.name = name
        self.role = role
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{name.lower()}")
    
    @abstractmethod
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and update it.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        pass
    
    def log_reasoning(self, context: AgentContext, reasoning: str):
        """
        Log a reasoning step to the context.
        
        Args:
            context: Agent context
            reasoning: Reasoning step
        """
        step = f"[{self.name}] {reasoning}"
        context.reasoning_steps.append(step)
        
        # Also log to the logger
        self.logger.info(reasoning)
    
    def add_explanation(self, context: AgentContext, key: str, explanation: str):
        """
        Add an explanation to the context.
        
        Args:
            context: Agent context
            key: Explanation key
            explanation: Explanation text
        """
        context.explanations[key] = explanation


class QueryUnderstandingAgent(Agent):
    """
    Agent responsible for understanding the user's natural language query.
    
    This agent extracts the intent, entities, and other semantic information
    from the user's query to guide the SQL generation process.
    """
    
    def __init__(self, name: str = "QueryUnderstanding", config: Dict[str, Any] = None):
        """Initialize the query understanding agent."""
        super().__init__(name, AgentRole.QUERY_UNDERSTANDING, config)
    
    @abstractmethod
    def extract_intent(self, query: str) -> str:
        """
        Extract the primary intent from a query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Primary intent (e.g., "select", "aggregate", "filter")
        """
        pass
    
    @abstractmethod
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from a query.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of entity names mentioned in the query
        """
        pass


class SchemaAnalysisAgent(Agent):
    """
    Agent responsible for analyzing the database schema.
    
    This agent identifies the relevant tables and columns for a query,
    and provides information about their relationships.
    """
    
    def __init__(self, name: str = "SchemaAnalysis", config: Dict[str, Any] = None):
        """Initialize the schema analysis agent."""
        super().__init__(name, AgentRole.SCHEMA_ANALYSIS, config)
    
    @abstractmethod
    def identify_relevant_tables(self, context: AgentContext) -> List[str]:
        """
        Identify the tables relevant to a query.
        
        Args:
            context: Agent context
            
        Returns:
            List of relevant table names
        """
        pass
    
    @abstractmethod
    def identify_relevant_columns(self, context: AgentContext) -> Dict[str, List[str]]:
        """
        Identify the columns relevant to a query.
        
        Args:
            context: Agent context
            
        Returns:
            Dictionary mapping table names to lists of column names
        """
        pass


class SQLGenerationAgent(Agent):
    """
    Agent responsible for generating SQL queries.
    
    This agent takes the understood intent and schema information
    and generates an SQL query to fulfill the user's request.
    """
    
    def __init__(self, name: str = "SQLGeneration", config: Dict[str, Any] = None):
        """Initialize the SQL generation agent."""
        super().__init__(name, AgentRole.SQL_GENERATION, config)
    
    @abstractmethod
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
        pass


class QueryValidationAgent(Agent):
    """
    Agent responsible for validating and fixing SQL queries.
    
    This agent checks if the generated SQL is valid and fixes
    any issues before execution.
    """
    
    def __init__(self, name: str = "QueryValidation", config: Dict[str, Any] = None):
        """Initialize the query validation agent."""
        super().__init__(name, AgentRole.QUERY_VALIDATION, config)
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def fix_query(self, context: AgentContext) -> str:
        """
        Fix an invalid SQL query.
        
        Args:
            context: Agent context
            
        Returns:
            Fixed SQL query
        """
        pass


class ResultExplanationAgent(Agent):
    """
    Agent responsible for explaining query results.
    
    This agent provides natural language explanations of the
    query results to help the user understand them.
    """
    
    def __init__(self, name: str = "ResultExplanation", config: Dict[str, Any] = None):
        """Initialize the result explanation agent."""
        super().__init__(name, AgentRole.RESULT_EXPLANATION, config)
    
    @abstractmethod
    def explain_results(self, context: AgentContext) -> str:
        """
        Generate a natural language explanation of the query results.
        
        Args:
            context: Agent context
            
        Returns:
            Natural language explanation
        """
        pass


class VisualizationAgent(Agent):
    """
    Agent responsible for suggesting visualizations.
    
    This agent suggests appropriate visualizations for the
    query results based on their structure and content.
    """
    
    def __init__(self, name: str = "Visualization", config: Dict[str, Any] = None):
        """Initialize the visualization agent."""
        super().__init__(name, AgentRole.VISUALIZATION, config)
    
    @abstractmethod
    def suggest_visualization(self, context: AgentContext) -> Dict[str, Any]:
        """
        Suggest an appropriate visualization for the query results.
        
        Args:
            context: Agent context
            
        Returns:
            Dictionary with visualization configuration
        """
        pass


class CoordinatorAgent(Agent):
    """
    Agent responsible for coordinating other agents.
    
    This agent orchestrates the overall query processing pipeline
    by deciding which agents to invoke and when.
    """
    
    def __init__(self, name: str = "Coordinator", config: Dict[str, Any] = None):
        """Initialize the coordinator agent."""
        super().__init__(name, AgentRole.COORDINATOR, config)
        
        # Agents to coordinate
        self.agents: Dict[str, Agent] = {}
    
    def add_agent(self, agent: Agent):
        """
        Add an agent to be coordinated.
        
        Args:
            agent: Agent to add
        """
        self.agents[agent.name] = agent
    
    @abstractmethod
    def decide_next_agent(self, context: AgentContext) -> Optional[str]:
        """
        Decide which agent to invoke next.
        
        Args:
            context: Agent context
            
        Returns:
            Name of the next agent to invoke, or None if done
        """
        pass