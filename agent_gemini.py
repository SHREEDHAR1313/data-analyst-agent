"""Gemini-powered data analyst agent using Google's Interactions API."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from google import genai

from tools import TOOL_FUNCTIONS

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_TOOL_ROUNDS = 6

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "profile_dataset",
        "description": "Get the shape, column names/types, and a preview of the dataset. Call this first when you need to understand the dataset.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "missingness_report",
        "description": "Find which columns have missing data and how much.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "compute_summary_stats",
        "description": "Get mean, median, std, min, max, and quartiles for one numeric column.",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Column name"}},
            "required": ["column"],
        },
    },
    {
        "type": "function",
        "name": "detect_outliers",
        "description": "Find outliers in one numeric column using the IQR method.",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Column name"}},
            "required": ["column"],
        },
    },
    {
        "type": "function",
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation between two numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_a": {"type": "string", "description": "First numeric column"},
                "column_b": {"type": "string", "description": "Second numeric column"},
            },
            "required": ["column_a", "column_b"],
        },
    },
    {
        "type": "function",
        "name": "group_by_aggregate",
        "description": "Group by a categorical column and aggregate a numeric column.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_column": {"type": "string"},
                "value_column": {"type": "string"},
                "agg": {"type": "string", "enum": ["mean", "sum", "median", "count", "min", "max"]},
            },
            "required": ["group_column", "value_column"],
        },
    },
    {
        "type": "function",
        "name": "plot_histogram",
        "description": "Generate and save a histogram of one numeric column's distribution.",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string"}},
            "required": ["column"],
        },
    },
    {
        "type": "function",
        "name": "plot_correlation_heatmap",
        "description": "Generate and save a correlation heatmap across all numeric columns.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "plot_bar_by_group",
        "description": "Generate and save a bar chart of an aggregated value by group.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_column": {"type": "string"},
                "value_column": {"type": "string"},
                "agg": {"type": "string", "enum": ["mean", "sum", "median", "count", "min", "max"]},
            },
            "required": ["group_column", "value_column"],
        },
    },
]

SYSTEM_PROMPT = """You are a data analyst agent. You have tools to profile, compute statistics on, and visualize a dataset that is already loaded.

You do NOT have the raw dataframe in your context. Use the tools rather than guessing values.
When a question requires analysis, use the minimum set of tools needed. Profile the dataset when column names/types are not yet known.
Only report numerical findings that came from tool results. If a tool returns an error, use the error to correct the column name or arguments and retry.
Do not claim correlation proves causation. Mention sample size or important data-quality caveats when relevant.
When a chart is generated, say that it was generated; the application will display it.
Keep the final answer concise, clear, and grounded in the returned tool results."""


class DataAnalystAgent:
    """Run deterministic dataframe tools through Gemini function calling."""

    def __init__(self, df: pd.DataFrame, api_key: str | None = None):
        if df is None or df.empty:
            raise ValueError("The dataset is empty. Please upload a CSV containing at least one row.")

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to .env or enter it in the app.")

        self.df = df
        self.client = genai.Client(api_key=key)

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOL_FUNCTIONS:
            return {"error": f"Unknown tool '{tool_name}'"}
        try:
            result = TOOL_FUNCTIONS[tool_name](self.df, **tool_input)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            return {"error": f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}"}

    def ask(self, question: str, verbose: bool = True) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Please enter a question about the dataset.", "tool_trace": [], "charts": [], "rounds": 0}

        trace: list[dict[str, Any]] = []
        charts: list[str] = []

        interaction = self.client.interactions.create(
            model=MODEL,
            system_instruction=SYSTEM_PROMPT,
            input=question,
            tools=TOOL_SCHEMAS,
        )

        for round_num in range(MAX_TOOL_ROUNDS):
            function_calls = [step for step in (interaction.steps or []) if getattr(step, "type", None) == "function_call"]

            if not function_calls:
                answer = interaction.output_text or "I could not produce a final answer. Please try the question again."
                return {
                    "answer": answer,
                    "tool_trace": trace,
                    "charts": charts,
                    "rounds": round_num + 1,
                }

            results_input = []
            for call in function_calls:
                tool_name = call.name
                tool_input = dict(call.arguments or {})
                if verbose:
                    print(f"  [tool call] {tool_name}({tool_input})")

                result = self._execute_tool(tool_name, tool_input)
                trace.append({"tool": tool_name, "input": tool_input, "result": result})

                if isinstance(result, dict) and result.get("file_path"):
                    charts.append(result["file_path"])

                results_input.append(
                    {
                        "type": "function_result",
                        "name": tool_name,
                        "call_id": call.id,
                        "result": [{"type": "text", "text": json.dumps(result, default=str)}],
                    }
                )

            interaction = self.client.interactions.create(
                model=MODEL,
                previous_interaction_id=interaction.id,
                input=results_input,
                tools=TOOL_SCHEMAS,
            )

        return {
            "answer": "I reached the analysis step limit before producing a final answer. Please try a more focused question.",
            "tool_trace": trace,
            "charts": charts,
            "rounds": MAX_TOOL_ROUNDS,
        }


if __name__ == "__main__":
    df = pd.read_csv("sample_data/sales_data.csv")
    agent = DataAnalystAgent(df)
    result = agent.ask("Give me an overview of this dataset and flag any data quality issues.")
    print("\n--- FINAL ANSWER ---")
    print(result["answer"])
