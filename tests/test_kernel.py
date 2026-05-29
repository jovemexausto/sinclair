"""Tests for PythonKernel behavior."""

from __future__ import annotations


from sinclair import PythonKernel


def test_basic_execution():
    k = PythonKernel()
    r = k.execute("x = 1 + 1")
    assert r.error is None
    assert "x" in r.variables_added


def test_stdout_captured():
    k = PythonKernel()
    r = k.execute("print('hello world')")
    assert "hello world" in r.stdout


def test_error_captured():
    k = PythonKernel()
    r = k.execute("1 / 0")
    assert r.error is not None
    assert "ZeroDivisionError" in r.error


def test_namespace_persists():
    k = PythonKernel()
    k.execute("x = 42")
    r = k.execute("print(x)")
    assert "42" in r.stdout


def test_variables_added_tracking():
    k = PythonKernel()
    r = k.execute("a = 1\nb = 2")
    assert "a" in r.variables_added
    assert "b" in r.variables_added


def test_variables_modified_tracking():
    k = PythonKernel(env={"x": 1})
    r = k.execute("x = 999")
    assert "x" in r.variables_modified


def test_reset_restores_initial_env():
    k = PythonKernel(env={"x": 10})
    k.execute("x = 999\ny = 1")
    k.reset()
    r = k.execute("print(x)")
    assert "10" in r.stdout
    r2 = k.execute("print('y' in dir())")
    assert "False" in r2.stdout


def test_restricted_allows_import_temporarily():
    k = PythonKernel(restricted=True, allowed_modules=["math"])
    r = k.execute("import os")
    assert r.error is None
    assert "os" in r.variables_added


def test_restricted_allows_listed_module():
    k = PythonKernel(restricted=True, allowed_modules=["math"])
    r = k.execute("import math; print(math.pi)")
    assert r.error is None
    assert "3.14" in r.stdout


def test_unrestricted_allows_any_import():
    k = PythonKernel(restricted=False)
    r = k.execute("import os; print(os.sep)")
    assert r.error is None


def test_snapshot_skips_non_serializable():
    k = PythonKernel(env={"obj": object(), "x": 42})
    snap = k.snapshot()
    assert "x" in snap
    assert "obj" not in snap


def test_as_tool_returns_tool():
    k = PythonKernel()
    tool = k.as_tool()
    assert tool.name == "run_python"


def test_as_tool_executes():
    k = PythonKernel()
    tool = k.as_tool()
    result = tool.invoke(
        {
            "code": "print('via tool')",
            "intent": "Estou validando a execução da sandbox.",
        }
    )
    assert "via tool" in result


def test_as_tool_requires_progressive_first_person_intent():
    k = PythonKernel()
    tool = k.as_tool()
    try:
        tool.invoke(
            {"code": "print('via tool')", "intent": "Validar a sandbox"}
        )
    except Exception as exc:
        assert "high-level and product-facing" in str(exc)
    else:
        raise AssertionError(
            "run_python should reject non-first-person intent"
        )


def test_str_representation():
    k = PythonKernel()
    r = k.execute("print('test')")
    assert "test" in str(r)


def test_execution_time_recorded():
    k = PythonKernel()
    r = k.execute("x = 1")
    assert r.execution_time_s >= 0


def test_timeout_interrupts_execution():
    k = PythonKernel(timeout=0.01, restricted=False)
    r = k.execute("import time; time.sleep(1)")
    assert r.error is not None
    assert "TimeoutError" in r.error
