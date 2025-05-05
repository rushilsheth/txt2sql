"""
Agent System Main Module

This module provides the entry point for the agent-based text-to-SQL system.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from text_to_sql.agent.llm_agents import (
    LLMQueryUnderstandingAgent,
    LLMSchemaAnalysisAgent,
    LLMSQLGenerationAgent,
    LLMQueryValidationAgent,
    LLMResultExplanationAgent,
    LLMVisualizationAgent,
    SimpleCoordinatorAgent
)
from text_to_sql.agent.dynamic_coordinator import DynamicCoordinatorAgent
from text_to_sql.agent.types import AgentContext
from text_to_sql.db.base import DatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.utils.config_types import AgentConfig

logger = logging.getLogger(__name__)


class TextToSQLAgent:
    """
    Main entry point for the agent-based text-to-SQL system.
    
    This class creates and coordinates all the agents needed to process
    natural language queries and convert them to SQL.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        llm_engine: LLMEngine,
        config: AgentConfig
    ):
        """
        Initialize the text-to-SQL agent system.
        
        Args:
            db_manager: Database manager instance
            llm_engine: LLM engine instance
            config: Configuration dictionary
        """
        self.db_manager = db_manager
        self.llm_engine = llm_engine
        self.config = config or AgentConfig(**config)
        
        # Create agents
        self.query_understanding_agent = LLMQueryUnderstandingAgent(
            llm_engine=llm_engine,
            config=self.config.query_understanding
        )
        logger.info("Created LLMQueryUnderstandingAgent")
        
        self.schema_analysis_agent = LLMSchemaAnalysisAgent(
            llm_engine=llm_engine,
            db_manager=db_manager,
            config=self.config.schema_analysis
        )
        logger.info("Created LLMSchemaAnalysisAgent")
        
        self.sql_generation_agent = LLMSQLGenerationAgent(
            llm_engine=llm_engine,
            db_manager=db_manager,
            config=self.config.sql_generation
        )
        logger.info("Created LLMSQLGenerationAgent")
        
        self.query_validation_agent = LLMQueryValidationAgent(
            llm_engine=llm_engine,
            db_manager=db_manager,
            config=self.config.query_validation
        )
        logger.info("Created LLMQueryValidationAgent")
        
        self.result_explanation_agent = LLMResultExplanationAgent(
            llm_engine=llm_engine,
            config=self.config.result_explanation
        )
        logger.info("Created LLMResultExplanationAgent")
        
        self.visualization_agent = LLMVisualizationAgent(
            llm_engine=llm_engine,
            config=self.config.visualization
        )
        logger.info("Created LLMVisualizationAgent")
        
        # Determine which coordinator to use
        use_dynamic_coordinator = self.config.use_dynamic_coordinator
        
        if use_dynamic_coordinator:
            # Create dynamic coordinator
            self.coordinator = DynamicCoordinatorAgent(
                llm_engine=llm_engine,
                config=self.configcoordinator
            )
        else:
            # Create simple coordinator
            self.coordinator = SimpleCoordinatorAgent(
                config=self.config.coordinator
            )
        
        # Add agents to coordinator
        self.coordinator.add_agent(self.query_understanding_agent)
        self.coordinator.add_agent(self.schema_analysis_agent)
        self.coordinator.add_agent(self.sql_generation_agent)
        self.coordinator.add_agent(self.query_validation_agent)
        self.coordinator.add_agent(self.result_explanation_agent)
        self.coordinator.add_agent(self.visualization_agent)
    
    def process_query(self, query: str) -> Tuple[AgentContext, Dict[str, Any]]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple containing:
            - Agent context with processing results
            - Dictionary with additional info (e.g., timing)
        """
        
        # Create context
        context = AgentContext(
            user_query=query,
            max_iterations=self.config.max_iterations
        )
        
        # Process through coordinator
        start_time = time.time()
        context = self.coordinator.process(context)
        processing_time = time.time() - start_time
        
        # If we have a valid SQL query, execute it
        if context.sql_query and not context.result_error:
            # Execute the query
            start_time = time.time()
            results, error = self.db_manager.execute_query(context.sql_query, context.sql_params)
            execution_time = time.time() - start_time
            
            # Update context
            context.query_results = results or []
            context.result_error = error
            context.execution_time["query_execution"] = execution_time
            
            if error:
                logger.error(f"Error executing query: {error}")
                
                # Try to fix the query if there's an error
                if self.config.auto_fix_errors:
                    logger.info("Attempting to fix the query")
                    
                    # Update context with the error
                    context.result_error = error
                    
                    # Process with validation agent
                    context = self.query_validation_agent.process(context)
                    
                    # Try executing the fixed query
                    if context.sql_query:
                        logger.info(f"Executing fixed SQL query: {context.sql_query}")
                        
                        # Execute the query
                        start_time = time.time()
                        results, error = self.db_manager.execute_query(context.sql_query, context.sql_params)
                        execution_time = time.time() - start_time
                        
                        # Update context
                        context.query_results = results or []
                        context.result_error = error
                        context.execution_time["fixed_query_execution"] = execution_time
                        
                        if error:
                            logger.error(f"Error executing fixed query: {error}")
            else:
                logger.info(f"Query executed successfully, returned {len(results)} rows")
        
        # Generate explanation if we have results
        if context.query_results and not context.result_error:
            # Process with explanation agent
            context = self.result_explanation_agent.process(context)
            
            # Process with visualization agent
            context = self.visualization_agent.process(context)
        
        # Prepare additional info
        info = {
            "processing_time": sum(context.execution_time.values()),
            "execution_times": context.execution_time,
            "iterations": context.iterations,
            "agents_invoked": context.agent_history,
            "coordinator_type": self.coordinator.__class__.__name__
        }
        
        return context, info


def create_text_to_sql_agent(
    db_manager: DatabaseManager,
    llm_engine: LLMEngine,
    config: AgentConfig
) -> TextToSQLAgent:
    """
    Create a text-to-SQL agent.
    
    Args:
        db_manager: Database manager instance
        llm_engine: LLM engine instance
        config: Configuration object
        
    Returns:
        TextToSQLAgent instance
    """
    return TextToSQLAgent(
        db_manager=db_manager,
        llm_engine=llm_engine,
        config=config
    )