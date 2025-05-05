import os
import json
import logging
from typing import Literal

import gradio as gr
from openai import OpenAI
from text_to_sql import run_app

logger = logging.getLogger(__name__)

RouterMode = Literal["agent", "standard", "dynamic"]

def route_query(query: str) -> RouterMode:
    """
    Uses an LLM to determine which mode to use based on the query.
    Returns one of: 'agent', 'standard', or 'dynamic'.
    """
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found. Defaulting to agent mode.")
        return "agent"
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Define the function for the LLM to call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "select_query_mode",
                    "description": "Select the most appropriate mode to handle a SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["agent", "standard", "dynamic"],
                                "description": "The mode to use for processing the query"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Explanation for why this mode was selected"
                            }
                        },
                        "required": ["mode", "explanation"]
                    }
                }
            }
        ]
        
        # Create a system message that explains the different modes
        system_message = """
        You are a query router for a text-to-SQL system. Analyze the user's query and select the most appropriate processing mode:

        1. "agent" mode: Uses a simple coordinator with multiple agents for handling multi-step queries that require reasoning.
        
        2. "standard" mode: Direct text-to-SQL transformation without agent approach. Best for straightforward, single SQL queries.
        
        3. "dynamic" mode: Uses an advanced dynamic coordinator for very complex queries requiring advanced planning and multiple specialized agents.
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Or another appropriate model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Query: {query}"}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "select_query_mode"}}
        )
        
        # Extract the function call
        tool_call = response.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        
        # Log the mode selection and explanation
        mode = function_args.get("mode")
        explanation = function_args.get("explanation")
        logger.info(f"LLM router selected '{mode}' mode: {explanation}")
        
        return mode
    
    except Exception as e:
        logger.error(f"Error in LLM router: {e}")
        logger.info("Defaulting to agent mode")
        return "agent"