"""
Dynamic Coordinator Agent Module

This module provides an enhanced coordinator agent with greater autonomy
and decision-making capabilities, using OpenAI function calling for structured
agent interactions.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import openai
from openai.types.chat import ChatCompletion

from text_to_sql.agent.types import Agent, AgentContext, AgentRole, CoordinatorAgent
from text_to_sql.db.base import DatabaseManager
from text_to_sql.llm.engine import LLMEngine

logger = logging.getLogger(__name__)

# Define the function schemas for OpenAI function calling
AGENT_FUNCTIONS = [
    {
        "name": "invoke_query_understanding",
        "description": "Analyze the natural language query to extract intent and entities",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "invoke_schema_analysis",
        "description": "Analyze the database schema to identify relevant tables and columns",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "invoke_sql_generation",
        "description": "Generate SQL from the natural language query and schema information",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "invoke_query_validation",
        "description": "Validate and potentially fix the generated SQL query",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "invoke_result_explanation",
        "description": "Explain the query results in natural language",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "invoke_visualization",
        "description": "Suggest appropriate visualizations for the query results",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why this agent should be invoked"
                }
            },
            "required": ["reasoning"]
        }
    },
    {
        "name": "finish_processing",
        "description": "Complete the processing pipeline when no more agents need to be invoked",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for why processing should be completed"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished in the processing pipeline"
                }
            },
            "required": ["reasoning", "summary"]
        }
    }
]

# Define the system prompt for the dynamic coordinator
COORDINATOR_SYSTEM_PROMPT = """You are an expert coordinator agent for a text-to-SQL system. 
Your job is to decide which specialized agent to invoke next based on the current state of processing.

Available agents:
1. Query Understanding - Extracts intent and entities from natural language queries
2. Schema Analysis - Identifies relevant tables and columns for a query
3. SQL Generation - Converts natural language to SQL with schema awareness
4. Query Validation - Validates and repairs SQL before execution
5. Result Explanation - Provides natural language explanations of results
6. Visualization - Suggests appropriate visualizations for query results

You will be given the current context of the processing pipeline, including:
- The original user query
- Progress made so far
- Results from previously invoked agents
- Any errors or issues encountered

Based on this context, you must decide which agent should be invoked next, or if processing is complete.
Always provide clear reasoning for your decision.

Important guidelines:
- You should follow a logical sequence that makes sense for the query
- For new queries, typically start with Query Understanding
- Only invoke Result Explanation and Visualization after a successful query execution
- If there are errors in the SQL, invoke Query Validation
- If Schema Analysis hasn't been done before SQL Generation, invoke Schema Analysis first
- When all necessary agents have been invoked and the query has been executed successfully, finish the processing
"""


class DynamicCoordinatorAgent(CoordinatorAgent):
    """
    Enhanced implementation of the CoordinatorAgent with dynamic decision-making.
    
    Uses OpenAI function calling to make decisions about which agent to invoke next
    based on the current state of the context.
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        name: str = "DynamicCoordinator", 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the dynamic coordinator agent.
        
        Args:
            llm_engine: LLM engine instance
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.llm_engine = llm_engine
        
        # LLM model to use for function calling
        self.model = config.get("model", "gpt-4o")
        
        # Function name to agent name mapping
        self.function_to_agent = {
            "invoke_query_understanding": "LLMQueryUnderstanding",
            "invoke_schema_analysis": "LLMSchemaAnalysis",
            "invoke_sql_generation": "LLMSQLGeneration",
            "invoke_query_validation": "LLMQueryValidation",
            "invoke_result_explanation": "LLMResultExplanation",
            "invoke_visualization": "LLMVisualization"
        }
        
        # Add reflection capabilities if enabled
        self.use_reflection = config.get("use_reflection", True)
        self.reflection_frequency = config.get("reflection_frequency", 2)
    
    def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context by dynamically coordinating other agents.
        
        Args:
            context: The current agent context
            
        Returns:
            Updated agent context
        """
        self.log_reasoning(context, "Starting dynamic query processing pipeline")
        
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
        
        # Process the query through a dynamic pipeline
        start_time = time.time()
        
        while context.iterations < context.max_iterations:
            # Reflect on progress if enabled
            if self.use_reflection and context.iterations > 0 and context.iterations % self.reflection_frequency == 0:
                self.reflect_on_progress(context)
            
            # Decide which agent to invoke next
            next_agent, reasoning = self.decide_next_agent(context)
            
            if next_agent is None:
                self.log_reasoning(context, f"Pipeline completed: {reasoning}")
                break
            
            if next_agent not in self.agents:
                self.log_reasoning(context, f"Agent {next_agent} not found, skipping")
                continue
            
            # Invoke the agent
            agent = self.agents[next_agent]
            self.log_reasoning(context, f"Invoking agent: {agent.name} - Reason: {reasoning}")
            
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
    
    def decide_next_agent(self, context: AgentContext) -> Tuple[Optional[str], str]:
        """
        Decide which agent to invoke next using OpenAI function calling.
        
        Args:
            context: Agent context
            
        Returns:
            Tuple containing:
            - Name of the next agent to invoke, or None if done
            - Reasoning for the decision
        """
        # Prepare the context for the prompt
        context_summary = self._format_context_for_prompt(context)
        
        # Create the prompt
        user_prompt = f"""Current Context:
{context_summary}

Based on this context, decide which agent should be invoked next, or if processing is complete.
"""
        
        try:
            # Call OpenAI with function calling
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": func} for func in AGENT_FUNCTIONS],
                tool_choice="auto"
            )
            
            # Extract the function call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Extract reasoning
                reasoning = function_args.get("reasoning", "No reasoning provided")
                
                # If finishing, return None
                if function_name == "finish_processing":
                    summary = function_args.get("summary", "")
                    context.metadata["processing_summary"] = summary
                    return None, reasoning
                
                # Map function name to agent name
                agent_name = self.function_to_agent.get(function_name)
                
                if agent_name:
                    return agent_name, reasoning
                else:
                    return None, f"Unknown function: {function_name}"
            else:
                # No function call was made
                return None, "No agent selection was made"
                
        except Exception as e:
            logger.error(f"Error deciding next agent: {e}")
            
            # Fallback to a simple decision
            if not context.query_intent:
                return "LLMQueryUnderstanding", "Fallback: Need to understand query first"
            elif not context.relevant_tables:
                return "LLMSchemaAnalysis", "Fallback: Need schema analysis"
            elif not context.sql_query:
                return "LLMSQLGeneration", "Fallback: Need to generate SQL"
            elif context.result_error:
                return "LLMQueryValidation", "Fallback: Need to fix query errors"
            elif context.query_results and "results" not in context.explanations:
                return "LLMResultExplanation", "Fallback: Need to explain results"
            elif context.query_results and "visualization" not in context.metadata:
                return "LLMVisualization", "Fallback: Need visualization suggestions"
            else:
                return None, "Fallback: Pipeline complete"
    
    def reflect_on_progress(self, context: AgentContext) -> None:
        """
        Reflect on the progress made so far and adjust the strategy if needed.
        
        Args:
            context: Agent context
        """
        # Prepare the context for the prompt
        context_summary = self._format_context_for_prompt(context)
        
        # Create the prompt
        reflection_prompt = f"""Current Progress:
{context_summary}

Reflect on the progress made so far:
1. What has been accomplished?
2. Are there any issues or bottlenecks?
3. Should the strategy be adjusted?
4. What are the priorities for the next steps?

Provide a brief reflection on the current state of processing.
"""
        
        try:
            # Call the LLM
            response = self.llm_engine._call_llm(reflection_prompt)
            
            # Log the reflection
            self.log_reasoning(context, f"Reflection: {response.strip()}")
            
            # Add to context metadata
            if "reflections" not in context.metadata:
                context.metadata["reflections"] = []
            
            context.metadata["reflections"].append({
                "iteration": context.iterations,
                "reflection": response.strip()
            })
            
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
    
    def _format_context_for_prompt(self, context: AgentContext) -> str:
        """
        Format the agent context for inclusion in prompts.
        
        Args:
            context: Agent context
            
        Returns:
            Formatted context string
        """
        # User query
        formatted = f"USER QUERY: {context.user_query}\n\n"
        
        # Current state
        formatted += "CURRENT STATE:\n"
        formatted += f"- Iterations completed: {context.iterations}\n"
        
        # Query understanding
        if context.query_intent:
            formatted += f"- Extracted intent: {context.query_intent}\n"
        if context.query_entities:
            formatted += f"- Extracted entities: {', '.join(context.query_entities)}\n"
        
        # Schema analysis
        if context.relevant_tables:
            formatted += f"- Relevant tables: {', '.join(context.relevant_tables)}\n"
        
        # SQL generation
        if context.sql_query:
            formatted += f"- Generated SQL: {context.sql_query}\n"
        
        # Query execution
        if context.result_error:
            formatted += f"- Execution error: {context.result_error}\n"
        elif context.query_results:
            formatted += f"- Query executed successfully with {len(context.query_results)} results\n"
        
        # Post-processing
        if "results" in context.explanations:
            formatted += "- Results have been explained\n"
        if "visualization" in context.metadata:
            formatted += "- Visualization has been suggested\n"
        
        # Agent history
        formatted += "\nAGENT HISTORY:\n"
        for agent in context.agent_history:
            formatted += f"- {agent}\n"
        
        return formatted