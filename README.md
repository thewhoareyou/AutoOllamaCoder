# AutoOllamaCoder

AutoOllamaCoder provides a minimal framework for generating and executing Python
code using an Ollama language model running locally. The original prototype was
implemented in a Jupyter notebook. The logic has now been refactored into a
reusable Python package with basic error handling and tests.

## Installation

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt  # currently only `requests` and `pytest` for testing
```

Make sure an Ollama server is running locally (`ollama serve`) and that the
model configured in `autoollamacoder.core.MODEL` is available.

## Usage

Run the interactive prompt:

```bash
python -m autoollamacoder
```

Enter prompts and the model will respond with code blocks. Any lines beginning
with an exclamation point (`!`) are executed as shell commands; the rest is
executed as Python code. Errors are caught and reported without terminating the
program.

## Development

Tests can be run with `pytest`:

```bash
pytest
```

The package exposes the following functions:

```python
from autoollamacoder import chat, run_python_code, main
```

See `autoollamacoder/core.py` for implementation details.
