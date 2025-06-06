"""
Visualization Chart Functions

This module provides functions for creating different types of charts
and visualizations for the text-to-SQL application.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def validate_dataframe(data: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that the DataFrame is not empty and contains required columns.
    
    Args:
        data: DataFrame to validate.
        required_columns: List of columns that must be in the DataFrame.
    
    Raises:
        ValueError: If DataFrame is empty or any required column is missing.
    """
    if data.empty:
        raise ValueError("Provided DataFrame is empty.")
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

def create_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None
) -> plt.Figure:
    try:
        logger.info("Creating bar chart with x_column='%s', y_column='%s', color_column='%s'", 
                    x_column, y_column, color_column)
        
        # Validate the DataFrame
        validate_dataframe(data, [x_column, y_column])
        
        logger.debug("Data head:\n%s", data.head())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        if color_column:
            if color_column not in data.columns:
                logger.error("Color column '%s' not found in DataFrame.", color_column)
                raise ValueError(f"Color column '{color_column}' not found in DataFrame.")
            grouped_data = data.groupby([x_column, color_column])[y_column].sum().unstack()
            grouped_data.plot(kind='bar', ax=ax)
        else:
# Simple bar chart
            sns.barplot(x=x_column, y=y_column, data=data, ax=ax)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{y_column} by {x_column}")
        plt.xticks(rotation=45, ha='right')
        
        if color_column:
            ax.legend(title=color_column)
        
        plt.tight_layout()
        logger.info("Bar chart created successfully.")
        return fig
    except Exception as e:
        logger.error("Error creating bar chart: %s", e)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", ha='center', va='center')
        return fig

def create_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None
) -> plt.Figure:
    try:
        logger.info("Creating line chart with x_column='%s', y_column='%s', color_column='%s'", 
                    x_column, y_column, color_column)
        
        # Validate the DataFrame
        validate_dataframe(data, [x_column, y_column])
        
        logger.debug("Data head:\n%s", data.head())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        data = data.sort_values(by=x_column)
        
        if color_column:
            if color_column not in data.columns:
                logger.error("Color column '%s' not found in DataFrame.", color_column)
                raise ValueError(f"Color column '{color_column}' not found in DataFrame.")
            sns.lineplot(x=x_column, y=y_column, hue=color_column, data=data, marker='o', ax=ax)
        else:
            sns.lineplot(x=x_column, y=y_column, data=data, marker='o', ax=ax)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{y_column} by {x_column}")
        plt.xticks(rotation=45, ha='right')
        
        if color_column:
            ax.legend(title=color_column)
        
        plt.tight_layout()
        logger.info("Line chart created successfully.")
        return fig
        
    except Exception as e:
        logger.error("Error creating line chart: %s", e)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", ha='center', va='center')
        return fig

def create_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None
) -> plt.Figure:
    try:
        logger.info("Creating scatter plot with x_column='%s', y_column='%s', color_column='%s'", 
                    x_column, y_column, color_column)
        
        # Validate the DataFrame
        validate_dataframe(data, [x_column, y_column])
        
        logger.debug("Data head:\n%s", data.head())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        if color_column:
            if color_column not in data.columns:
                logger.error("Color column '%s' not found in DataFrame.", color_column)
                raise ValueError(f"Color column '{color_column}' not found in DataFrame.")
            sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=data, ax=ax)
        else:
            sns.scatterplot(x=x_column, y=y_column, data=data, ax=ax)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{y_column} vs {x_column}")
        
        if color_column:
            ax.legend(title=color_column)
        
        plt.tight_layout()
        logger.info("Scatter plot created successfully.")
        return fig
        
    except Exception as e:
        logger.error("Error creating scatter plot: %s", e)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", ha='center', va='center')
        return fig

def create_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str
) -> plt.Figure:
    try:
        logger.info("Creating heatmap with x_column='%s' and y_column='%s'", x_column, y_column)
        
        # Validate the DataFrame
        validate_dataframe(data, [x_column, y_column])
        
        logger.debug("Data head:\n%s", data.head())
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot_data = data.pivot_table(
            index=y_column,
            columns=x_column,
            aggfunc='size',
            fill_value=0
        )
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            linewidths=.5,
            ax=ax
        )
        
        ax.set_title(f"Heatmap of {y_column} vs {x_column}")
        plt.tight_layout()
        logger.info("Heatmap created successfully.")
        return fig
        
    except Exception as e:
        logger.error("Error creating heatmap: %s", e)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", ha='center', va='center')
        return fig