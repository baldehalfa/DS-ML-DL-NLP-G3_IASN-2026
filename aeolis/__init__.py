"""AEOLIS - shared helpers for the G3 / IASN Data Science & AI course.

Students import from this package but never edit it. Its whole purpose is to absorb the
code that costs typing without teaching anything (loading, plotting, boilerplate) so that
scarce lab time is spent on decisions instead of syntax.

    from aeolis import data, plots, evaluate, llm

Modules
    data      load the AEOLIS datasets (one function per subsystem)
    plots     every figure the notebooks need, one call each
    evaluate  metrics, cost-based thresholds, temporal backtests, RAG evaluation
    llm       ask(prompt, backend=cache|local|ollama|api) - no API key required
"""

__version__ = "0.1.0"

__all__ = ["data", "plots", "evaluate", "llm"]
