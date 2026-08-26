"""Regression tests for executing multiline LLM-generated Python code."""

from app.code_sandbox import CodeSandbox


def test_execute_multiline_program_with_trailing_comment():
    result = CodeSandbox().execute(
        "for value in range(3):\n"
        "    print(value)\n"
        "# The model may append a comment after the program"
    )

    assert result["error"] == ""
    assert result["output"] == "0\n1\n2\n"


def test_execute_function_definition_and_final_expression():
    result = CodeSandbox().execute(
        "def greet(name):\n"
        "    return f'Hello {name}'\n"
        "greet('Sir')"
    )

    assert result["error"] == ""
    assert result["result"] == "'Hello Sir'"
