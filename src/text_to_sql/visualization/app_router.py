import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from text_to_sql.agent.router import route_query
from text_to_sql.visualization.app_with_agents import AgentBasedTextToSQLApp

logger = logging.getLogger(__name__)

class RouterBasedTextToSQLApp(AgentBasedTextToSQLApp):
    """
    A variant of the agent-based Gradio app that, before processing a natural language query,
    calls the LLM router (route_query) to choose which mode to use.
    """

    def handle_query(
        self, 
        query: str
    ) -> List[Union[str, pd.DataFrame, plt.Figure, Dict[str, Any], None]]:
        """
        Route the query using the LLM router and then process it accordingly.
        """
        if not query.strip():
            return [
                "", None, "", "Please enter a query.", 
                gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), "Table", None
            ]
        
        # Use the router to decide mode: agent, standard, or dynamic.
        try:
            mode = route_query(query)
            logger.info(f"Router selected mode: {mode}")
        except Exception as e:
            logger.error(f"Error during routing: {e}")
            mode = "agent"
        
        if mode == "agent" or mode == "dynamic":
            # For 'agent' and 'dynamic', we use the existing agent-based processing.
            return super().handle_query(query)
        elif mode == "standard":
            # For standard mode, use a direct text-to-SQL transformation.
            try:
                sql_query, confidence, metadata = self.llm_engine.generate_sql_no_agents(query)
                
                # Execute the SQL query.
                results, error = self.db_manager.execute_query(sql_query)
                
                # Convert results to DataFrame.
                df = pd.DataFrame(results) if results else pd.DataFrame()
                
                explanation = "Standard mode: Query converted using direct text-to-SQL transformation."
                status = "Standard mode executed."
                columns = df.columns.tolist() if not df.empty else []
                
                # Return outputs matching the original handle_query return signature.
                return (
                    sql_query,
                    df,
                    explanation,
                    status,
                    gr.update(choices=columns),
                    gr.update(choices=columns),
                    gr.update(choices=columns),
                    "Table",
                    None
                )
            except Exception as e:
                logger.error(f"Error in standard processing: {e}")
                return [
                    "", 
                    None, 
                    "", 
                    f"❌ Error in standard mode: {str(e)}", 
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    "Table",
                    None
                ]
        else:
            return super().handle_query(query)