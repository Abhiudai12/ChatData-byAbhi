import os
import pandas as pd
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent
from langchain_experimental.utilities import PythonREPL
import plotly.express as px

# Initialize an empty DataFrame
df = pd.DataFrame()

# Initialize LLM and Agent
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)
chatmodel = ChatHuggingFace(llm=llm)

# Setup REPL environment
python_repl_env = PythonREPL()
python_repl_env.globals["df"] = df
python_repl_env.globals["df2"] = df.copy()

@tool
def python_repl(code: str):
    """
    Execute Python code on dataframe df.

    The dataframe df already exists.

    IMPORTANT:
    - Only send Python code as a string.
    - Do NOT include explanations.
    - Do NOT create charts here.
    - Always use print() to show results.

    Example:
    print(df.columns)
    print(df.head())
    """
    return python_repl_env.run(code)

@tool
def plot_chart(group_by: str, value_column: str, top_n: int = 5, chart_type: str = "bar"):
    """
    Plot aggregated data from dataframe df.

    Parameters:
    group_by: column to group by (example: brand, city)
    value_column: column to aggregate (example: revenue, quantity)
    top_n: number of top results to show
    - Never overwrite original df
    - Always create new variables like result or summary
    """
    current_df = python_repl_env.globals.get("df", pd.DataFrame())
    
    if current_df.empty:
        return "Error: Dataframe is empty. Please upload data first."

    data = (
        current_df.groupby(group_by)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    if chart_type == "bar":
        fig = px.bar(data, x=group_by, y=value_column)
    elif chart_type == "line":
        fig = px.line(data, x=group_by, y=value_column)
    elif chart_type == "scatter":
        fig = px.scatter(data, x=group_by, y=value_column)
    else:
        return "Unsupported chart type"

    # Store the figure in globals so our Streamlit app can access and display it
    python_repl_env.globals["last_fig"] = fig
    
    return f"Showing top {top_n} {group_by} by {value_column}"

tools = [
    python_repl,
    plot_chart
]

system_prompt = """
You are an AI data analyst.

You have access to a pandas dataframe named df.

When using the python_repl tool:
- Pass arguments as JSON
- The key must be 'code'
- The value must be a string of Python code

Example tool call:
{"code": "print(df.head())"}
"""

agent = create_react_agent(
    chatmodel,
    tools=tools,
    prompt=system_prompt
)
