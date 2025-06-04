import builtins
import types
from unittest import mock

import pytest

from autoollamacoder.core import run_code_with_shell_commands, run_python_code, chat, ExecutionError


def test_run_code_with_shell_commands_executes_python_and_shell(tmp_path):
    script = """!echo hello > {file}\nprint('world')""".format(file=tmp_path/'out.txt')
    with mock.patch('builtins.print') as mprint:
        run_code_with_shell_commands(script, [])
    # Ensure python print executed
    mprint.assert_any_call('world')
    # Ensure file created by shell command
    assert (tmp_path / 'out.txt').read_text().strip() == 'hello'


def test_run_python_code_parses_blocks(tmp_path):
    message = 'Text before```python\n!echo hi > {file}\n```more text'.format(file=tmp_path/'f.txt')
    run_python_code(message, [])
    assert (tmp_path / 'f.txt').read_text().strip() == 'hi'


def test_chat_handles_error_response():
    fake_resp = types.SimpleNamespace(
        iter_lines=lambda: [b'{"error": "bad"}'],
        raise_for_status=lambda: None,
    )
    fake_requests = types.SimpleNamespace(post=lambda *a, **k: fake_resp)
    with mock.patch('autoollamacoder.core.requests', fake_requests):
        with pytest.raises(Exception):
            chat([])

