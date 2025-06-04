"""Core functionality for AutoOllamaCoder."""
from .core import (
    run_code_with_shell_commands,
    run_python_code,
    chat,
    main,
    ExecutionError,
)

__all__ = [
    "run_code_with_shell_commands",
    "run_python_code",
    "chat",
    "main",
    "ExecutionError",
]
