import asyncio
import contextlib
import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints


def _python_type_to_json_type(py_type: Any) -> str:
    """Maps a Python type annotation to a JSON Schema primitive type."""
    if py_type in (str,):
        return "string"
    if py_type in (int,):
        return "integer"
    if py_type in (float,):
        return "number"
    if py_type in (bool,):
        return "boolean"
    if py_type in (list, tuple, Sequence):
        return "array"
    if py_type in (dict,):
        return "object"
    return "string"


def function_to_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generates an OpenAI-compatible function tool definition from a Python function."""
    if hasattr(func, "__tool_schema__"):
        return func.__tool_schema__

    name = getattr(func, "__name__", "unnamed_function")
    doc = inspect.getdoc(func) or f"Function {name}"

    sig = inspect.signature(func)
    type_hints = {}
    with contextlib.suppress(Exception):
        type_hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        py_type = type_hints.get(param_name, str)
        json_type = _python_type_to_json_type(py_type)
        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}",
        }
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc.split("\n")[0] if doc else f"Execute {name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that marks a Python function as an Omni-Local-LLM tool with auto-generated schema."""
    schema = function_to_tool_schema(func)
    func.__tool_schema__ = schema  # type: ignore[attr-defined]
    return func


class ToolRegistry:
    """Registry that manages tool definitions and executes function calls."""

    def __init__(self, tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None):
        self._functions: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

        if tools:
            for item in tools:
                if callable(item):
                    schema = function_to_tool_schema(item)
                    func_name = schema["function"]["name"]
                    self._functions[func_name] = item
                    self._schemas.append(schema)
                elif isinstance(item, dict):
                    self._schemas.append(item)
                    # If dict has a callable attached or purely schema
                    func_info = item.get("function", {})
                    name = func_info.get("name") if isinstance(func_info, dict) else None
                    if name and callable(item.get("callable")):
                        self._functions[name] = item["callable"]

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def has_tool(self, name: str) -> bool:
        return name in self._functions

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Executes a tool call synchronously and returns the result as a string."""
        if name not in self._functions:
            return f"Error: Tool '{name}' is not registered."

        func = self._functions[name]
        parsed_args: dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except Exception as e:
                return f"Error: Failed to parse arguments JSON: {e}"
        elif isinstance(arguments, dict):
            parsed_args = arguments

        try:
            result = func(**parsed_args)
            if inspect.iscoroutine(result):
                return asyncio.run(result)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    async def aexecute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Executes a tool call asynchronously."""
        if name not in self._functions:
            return f"Error: Tool '{name}' is not registered."

        func = self._functions[name]
        parsed_args: dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except Exception as e:
                return f"Error: Failed to parse arguments JSON: {e}"
        elif isinstance(arguments, dict):
            parsed_args = arguments

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**parsed_args)
            else:
                result = await asyncio.to_thread(func, **parsed_args)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
