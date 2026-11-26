import subprocess
import sys

import omnillm


def test_core_import_does_not_need_optional_dependencies():
    """`pip install omni-local-llm` (no extras) must be importable: block every optional dependency."""
    blocked = [
        "pydantic",
        "fastapi",
        "uvicorn",
        "ollama",
        "llama_cpp",
        "huggingface_hub",
        "httpx",
        "psutil",
        "jsonschema",
    ]
    code = (
        "import sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        "import omnillm\n"
        "from omnillm.core.hardware import recommend_model_fit\n"
        "from omnillm.harness import EvalHarness, BenchmarkHarness\n"
        "print(omnillm.__version__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_version_comes_from_package_metadata():
    from importlib.metadata import version

    assert omnillm.__version__ == version("omni-local-llm")
