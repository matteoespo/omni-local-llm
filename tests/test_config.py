from pathlib import Path

import pytest

from omnillm import LocalLLMManager, OpenAICompatibleAdapter
from omnillm.adapters.ollama_adapter import OllamaAdapter
from omnillm.core.config import ConfigError, find_config_file, load_config, parse_config
from omnillm.core.errors import BackendUnavailableError

FULL_CONFIG = """
default_backend = "lmstudio"

[backends.lmstudio]
type = "openai"
base_url = "http://localhost:1234"
api_key_env = "TEST_LMSTUDIO_KEY"
timeout = 30
headers = { X-Team = "ml" }
capabilities = { embeddings = false }

[backends.gpu-box]
type = "ollama"
host = "http://gpu-box:11434"

[models.coder]
backend = "llama.cpp"
model = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
n_ctx = 8192
temperature = 0.2
stop = ["<|im_end|>"]
fallbacks = ["gpu-box/qwen2.5-coder:7b", "small"]

[models.small]
model = "gpu-box/llama3.2:3b"
"""


def write(tmp_path: Path, text: str, name: str = "omnillm.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_full_config_registers_backends_aliases_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_LMSTUDIO_KEY", "sk-local")
    manager = LocalLLMManager.from_config(write(tmp_path, FULL_CONFIG))

    assert {"ollama", "llama.cpp", "lmstudio", "gpu-box"} <= set(manager.backend_names)
    assert manager.default_backend == "lmstudio"

    coder = manager.resolve("coder")
    assert (coder.backend, coder.model, coder.alias) == (
        "llama.cpp",
        "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "coder",
    )
    assert coder.options == {
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "n_ctx": 8192,
        "temperature": 0.2,
        "stop": ["<|im_end|>"],
    }
    assert [t.ref for t in manager.candidates("coder")] == [
        "llama.cpp/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "gpu-box/qwen2.5-coder:7b",
        "gpu-box/llama3.2:3b",
    ]
    assert manager.resolve("qwen3-8b").ref == "lmstudio/qwen3-8b"  # bare name -> default backend

    lmstudio = manager._get_backend("lmstudio")
    assert isinstance(lmstudio, OpenAICompatibleAdapter)
    assert lmstudio.base_url == "http://localhost:1234/v1"
    assert lmstudio._headers["Authorization"] == "Bearer sk-local"
    assert lmstudio._headers["X-Team"] == "ml"
    assert lmstudio.capabilities.embeddings is False

    gpu_box = manager._get_backend("gpu-box")
    assert isinstance(gpu_box, OllamaAdapter)
    assert gpu_box.host == "http://gpu-box:11434"


def test_missing_api_key_env_makes_backend_unavailable_not_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEST_LMSTUDIO_KEY", raising=False)
    manager = LocalLLMManager.from_config(write(tmp_path, FULL_CONFIG))

    with pytest.raises(BackendUnavailableError, match="TEST_LMSTUDIO_KEY"):
        manager._get_backend("lmstudio")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[models.a]\nmodel = 'x'\ntemprature = 0.1\n", "unknown key(s) ['temprature']"),
        ("[models.a]\ntemperature = 0.1\n", "'model' is required"),
        ("[models.a]\nmodel = 'x'\nn_ctx = 'big'\n", "'n_ctx' must be an integer"),
        ("[models.a]\nmodel = 'x'\nmax_tokens = true\n", "not a boolean"),
        ("[models.a]\nmodel = 'x'\nfallbacks = [1]\n", "'fallbacks' must be a list"),
        ("[models.a]\nmodel = 'x'\nstop = [1]\n", "'stop' must be a string or a list"),
        ("[models.'a/b']\nmodel = 'x'\n", "must not contain '/'"),
        ("[backends.x]\nbase_url = 'http://h'\n", "'type' is required"),
        ("[backends.x]\ntype = 'vllm'\n", "unknown type 'vllm'"),
        ("[backends.x]\ntype = 'openai'\n", "missing required key(s) ['base_url']"),
        ("[backends.x]\ntype = 'openai'\nbase_url = 'h'\napi_key = 'a'\napi_key_env = 'B'\n", "not both"),
        ("[backends.x]\ntype = 'openai'\nbase_url = 'h'\ncapabilities = { telepathy = true }\n", "unknown capability"),
        ("[backends.x]\ntype = 'ollama'\nbase_url = 'h'\n", "unknown key(s) ['base_url']"),
        ("servers = 1\n", "Unknown top-level key(s) ['servers']"),
    ],
)
def test_invalid_configs_fail_with_actionable_messages(tmp_path: Path, text: str, fragment: str):
    path = write(tmp_path, text)

    with pytest.raises(ConfigError) as info:
        load_config(path)

    assert fragment in str(info.value)
    assert str(path) in str(info.value)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[models.a]\nbackend = 'nope'\nmodel = 'x'\n", "unknown backend 'nope'"),
        ("[models.a]\nmodel = 'x'\nfallbacks = ['a']\n", "cannot fall back to itself"),
        ("[models.a]\nmodel = 'x'\nfallbacks = ['nope/x']\n", "fallback 'nope/x'"),
        ("default_backend = 'nope'\n", "default_backend 'nope'"),
    ],
)
def test_cross_reference_errors(tmp_path: Path, text: str, fragment: str):
    with pytest.raises(ConfigError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        LocalLLMManager.from_config(write(tmp_path, text))


def test_invalid_toml(tmp_path: Path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[models.a\nmodel = 'x'"))


def test_backend_type_aliases_are_accepted():
    config = parse_config({"backends": {"x": {"type": "openai-compatible", "base_url": "http://h"}}})
    assert config.backends[0].type == "openai"


def test_config_file_search_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project, home, explicit = tmp_path / "project", tmp_path / "xdg", tmp_path / "explicit.toml"
    project.mkdir()
    (home / "omnillm").mkdir(parents=True)
    explicit.write_text("")
    monkeypatch.chdir(project)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.delenv("OMNILLM_CONFIG", raising=False)

    assert find_config_file() is None

    user_config = home / "omnillm" / "config.toml"
    user_config.write_text("")
    assert find_config_file() == user_config

    local = project / "omnillm.toml"
    local.write_text("")
    assert find_config_file() == local

    monkeypatch.setenv("OMNILLM_CONFIG", str(explicit))
    assert find_config_file() == explicit
    assert find_config_file(search=False) == explicit

    monkeypatch.delenv("OMNILLM_CONFIG")
    assert find_config_file(search=False) is None
    with pytest.raises(ConfigError, match="not found"):
        find_config_file(tmp_path / "missing.toml")


def test_from_config_without_any_file_is_a_default_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OMNILLM_CONFIG", raising=False)

    manager = LocalLLMManager.from_config()

    assert manager.backend_names == ("ollama", "llama.cpp")
    assert manager.alias_names == ()


def test_alias_table_round_trips_through_toml(tmp_path: Path):
    from omnillm.core.config import alias_table, append_alias

    table = alias_table(
        "my.model",
        'org/"quoted"\\name',
        backend="ollama",
        options={"stop": ["</s>", "\n"], "local_files_only": True, "temperature": 0.2, "n_ctx": 4096, "seed": None},
    )
    assert table.startswith('[models."my.model"]')  # a dotted name stays one key, not a nested table
    path = append_alias(tmp_path / "omnillm.toml", table, "my.model")
    [alias] = load_config(path).aliases
    assert (alias.name, alias.backend, alias.model) == ("my.model", "ollama", 'org/"quoted"\\name')
    assert dict(alias.options) == {"stop": ["</s>", "\n"], "local_files_only": True, "temperature": 0.2, "n_ctx": 4096}
