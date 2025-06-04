import json
import subprocess
import time
from typing import List, Dict

try:
    import requests
except ImportError:  # pragma: no cover - handled in chat()
    requests = None


class ExecutionError(Exception):
    """Raised when executing generated code or shell commands fails."""


MODEL = "llama3.1"


def run_code_with_shell_commands(code_string: str, messages: List[Dict[str, str]]) -> None:
    """Execute generated Python code and shell commands.

    Lines starting with `!` are executed in the shell. Remaining lines are
    executed as Python code in the current interpreter.
    """
    code_to_run = ""
    lines = code_string.splitlines()
    for line in lines:
        if line.strip().startswith("!"):
            shell_command = line.strip()[1:].strip()
            print("EXECUTE GENERATED SHELL CODE")
            print(shell_command)
            try:
                subprocess.run(shell_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                raise ExecutionError(f"Shell command failed: {shell_command}") from e
        else:
            code_to_run += "\n" + line

    if code_to_run.strip():
        print("EXECUTE GENERATED CODE")
        print(code_to_run)
        try:
            exec(code_to_run, globals())
        except BaseException as e:
            raise ExecutionError(f"Generated code failed: {e}") from e


def run_python_code(message_data: str, messages: List[Dict[str, str]]) -> None:
    """Extract and execute Python code blocks from a message."""
    python_code_blocks = message_data.count("```python")
    start_string_num = 0
    for _ in range(python_code_blocks):
        start_index = message_data.find("```python", start_string_num) + 9
        end_index = message_data.find("```", start_index)
        code_string = message_data[start_index:end_index]
        start_string_num = end_index + 3
        run_code_with_shell_commands(code_string, messages)


def chat(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Send a chat completion request to the local Ollama server."""
    if requests is None:
        raise ImportError("The 'requests' package is required to use chat().")
    start_time = time.time()
    response = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": MODEL, "messages": messages, "stream": True},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    output = ""
    message = {}
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", {})
            content = message.get("content", "")
            if output == "":
                print(time.time() - start_time)
                start_time = time.time()
            output += content
            print(content, end="", flush=True)
        if body.get("done", False):
            message["content"] = output
            return message
    return message


def main() -> None:
    messages: List[Dict[str, str]] = []
    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            break
        print()
        messages.append({"role": "user", "content": user_input})
        message = chat(messages)
        messages.append(message)
        try:
            run_python_code(message["content"], messages)
        except ExecutionError as e:
            print(f"Error during execution: {e}")
        print("\n\n")


__all__ = [
    "run_code_with_shell_commands",
    "run_python_code",
    "chat",
    "main",
    "ExecutionError",
]
