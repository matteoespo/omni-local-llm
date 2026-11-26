"""TOML configuration: named backends (OpenAI-compatible servers, remote Ollama hosts) and model aliases.

Example ``omnillm.toml``::

    default_backend = "ollama"

    [backends.lmstudio]
    type = "openai"                      # any OpenAI-compatible server
    base_url = "http://localhost:1234/v1"

    [backends.gpu-box]
    type = "ollama"
    host = "http://gpu-box:11434"

    [models.coder]
    backend = "llama.cpp"
    model = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
    filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    n_ctx = 8192
    temperature = 0.2
    fallbacks = ["gpu-box/qwen2.5-coder:7b"]
"""

import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omnillm.core.errors import BackendUnavailableError, InvalidRequestError, OmniLLMError
from omnillm.core.types import BackendCapabilities

if TYPE_CHECKING:
    from omnillm.core.base import LLMBackend
    from omnillm.core.manager import LocalLLMManager

CONFIG_ENV_VAR = "OMNILLM_CONFIG"
LOCAL_CONFIG_NAME = "omnillm.toml"

_NUMBER = (int, float)
_BACKEND_KEYS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": str,
        "api_key": str,
        "api_key_env": str,
        "headers": dict,
        "timeout": _NUMBER,
        "connect_timeout": _NUMBER,
        "stream_usage": bool,
        "capabilities": dict,
    },
    "ollama": {"host": str, "timeout": _NUMBER},
    "llama.cpp": {},
}
_REQUIRED_BACKEND_KEYS = {"openai": {"base_url"}}
_BACKEND_TYPE_ALIASES = {"openai-compatible": "openai", "openai_compatible": "openai", "llamacpp": "llama.cpp"}
_ALIAS_KEYS: dict[str, Any] = {
    "backend": str,
    "model": str,
    "fallbacks": list,
    "temperature": _NUMBER,
    "max_tokens": int,
    "top_p": _NUMBER,
    "stop": (str, list),
    "seed": int,
    "filename": str,
    "revision": str,
    "cache_dir": str,
    "local_files_only": bool,
    "n_gpu_layers": int,
    "n_ctx": int,
    "mmproj_filename": str,
    "clip_model_path": str,
}
_CAPABILITY_NAMES = {f.name for f in fields(BackendCapabilities)}


class ConfigError(OmniLLMError):
    """Raised when a configuration file is missing, unreadable, or invalid."""


@dataclass(frozen=True, slots=True)
class BackendConfig:
    name: str
    type: str
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AliasConfig:
    name: str
    model: str
    backend: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    fallbacks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OmniConfig:
    path: Path | None = None
    default_backend: str | None = None
    backends: tuple[BackendConfig, ...] = ()
    aliases: tuple[AliasConfig, ...] = ()


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "omnillm" / "config.toml"


def find_config_file(path: str | Path | None = None, *, search: bool = True) -> Path | None:
    """Locates the config file: explicit path, then $OMNILLM_CONFIG, then ./omnillm.toml, then the user config."""
    explicit = path if path is not None else os.environ.get(CONFIG_ENV_VAR) or None
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"Config file not found: {candidate}")
        return candidate
    if not search:
        return None
    for candidate in (Path.cwd() / LOCAL_CONFIG_NAME, user_config_path()):
        if candidate.is_file():
            return candidate
    return None


def _type_name(expected: Any) -> str:
    names = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "table"}
    if isinstance(expected, tuple):
        name = " or ".join(dict.fromkeys(names.get(t, t.__name__) for t in expected))
    else:
        name = names.get(expected, getattr(expected, "__name__", str(expected)))
    return f"an {name}" if name[0] in "aeiou" else f"a {name}"


def _check_table(where: str, table: Mapping[str, Any], allowed: Mapping[str, Any]) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}")
    for key, value in table.items():
        expected = allowed[key]
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if isinstance(value, bool) and bool not in expected_types:
            raise ConfigError(f"{where}: '{key}' must be {_type_name(expected)}, not a boolean.")
        if not isinstance(value, expected_types):
            raise ConfigError(f"{where}: '{key}' must be {_type_name(expected)}.")


def _parse_backend(name: str, table: Any) -> BackendConfig:
    where = f"[backends.{name}]"
    if "/" in name:
        raise ConfigError(f"{where}: backend names must not contain '/'.")
    if not isinstance(table, dict):
        raise ConfigError(f"{where} must be a table.")
    settings = dict(table)
    raw_type = settings.pop("type", None)
    if not isinstance(raw_type, str):
        raise ConfigError(f"{where}: 'type' is required (one of {sorted(_BACKEND_KEYS)}).")
    backend_type = _BACKEND_TYPE_ALIASES.get(raw_type, raw_type)
    if backend_type not in _BACKEND_KEYS:
        raise ConfigError(f"{where}: unknown type '{raw_type}'. Choose from {sorted(_BACKEND_KEYS)}.")
    _check_table(where, settings, _BACKEND_KEYS[backend_type])
    missing = sorted(_REQUIRED_BACKEND_KEYS.get(backend_type, set()) - set(settings))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}.")
    if "api_key" in settings and "api_key_env" in settings:
        raise ConfigError(f"{where}: set either 'api_key' or 'api_key_env', not both.")
    for key, value in dict(settings.get("headers", {})).items():
        if not isinstance(value, str):
            raise ConfigError(f"{where}: header '{key}' must be a string.")
    for key, value in dict(settings.get("capabilities", {})).items():
        if key not in _CAPABILITY_NAMES:
            raise ConfigError(f"{where}: unknown capability '{key}'. Choose from {sorted(_CAPABILITY_NAMES)}.")
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: capability '{key}' must be true or false.")
    return BackendConfig(name=name, type=backend_type, settings=settings)


def _parse_alias(name: str, table: Any) -> AliasConfig:
    where = f"[models.{name}]"
    if "/" in name:
        raise ConfigError(f"{where}: alias names must not contain '/'.")
    if not isinstance(table, dict):
        raise ConfigError(f"{where} must be a table.")
    _check_table(where, table, _ALIAS_KEYS)
    options = dict(table)
    model = options.pop("model", None)
    if not model:
        raise ConfigError(f"{where}: 'model' is required.")
    backend = options.pop("backend", None)
    fallbacks = options.pop("fallbacks", [])
    if any(not isinstance(item, str) or not item for item in fallbacks):
        raise ConfigError(f"{where}: 'fallbacks' must be a list of model names.")
    if isinstance(options.get("stop"), list) and any(not isinstance(s, str) for s in options["stop"]):
        raise ConfigError(f"{where}: 'stop' must be a string or a list of strings.")
    return AliasConfig(name=name, model=model, backend=backend, options=options, fallbacks=tuple(fallbacks))


def parse_config(data: Mapping[str, Any], path: Path | None = None) -> OmniConfig:
    """Validates already-parsed TOML data."""
    unknown = sorted(set(data) - {"default_backend", "backends", "models"})
    if unknown:
        raise ConfigError(f"Unknown top-level key(s) {unknown}. Allowed: ['backends', 'default_backend', 'models']")
    default_backend = data.get("default_backend")
    if default_backend is not None and not isinstance(default_backend, str):
        raise ConfigError("'default_backend' must be a string.")
    backends_table = data.get("backends", {})
    models_table = data.get("models", {})
    if not isinstance(backends_table, dict) or not isinstance(models_table, dict):
        raise ConfigError("[backends] and [models] must be tables.")
    return OmniConfig(
        path=path,
        default_backend=default_backend,
        backends=tuple(_parse_backend(name, table) for name, table in backends_table.items()),
        aliases=tuple(_parse_alias(name, table) for name, table in models_table.items()),
    )


def load_config(path: str | Path) -> OmniConfig:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{config_path}: invalid TOML: {error}") from error
    try:
        return parse_config(data, config_path)
    except ConfigError as error:
        raise ConfigError(f"{config_path}: {error}") from None


def _backend_factory(config: BackendConfig) -> "Callable[[], LLMBackend]":
    settings = dict(config.settings)

    if config.type == "openai":

        def create_openai() -> "LLMBackend":
            from omnillm.adapters.openai_adapter import OpenAICompatibleAdapter

            api_key = settings.get("api_key")
            if "api_key_env" in settings:
                api_key = os.environ.get(settings["api_key_env"])
                if not api_key:
                    raise BackendUnavailableError(
                        f"Backend '{config.name}' needs the environment variable {settings['api_key_env']}."
                    )
            capabilities = settings.get("capabilities")
            return OpenAICompatibleAdapter(
                settings["base_url"],
                api_key=api_key,
                headers=settings.get("headers"),
                timeout=float(settings.get("timeout", 600.0)),
                connect_timeout=float(settings.get("connect_timeout", 5.0)),
                stream_usage=settings.get("stream_usage", True),
                capabilities=BackendCapabilities(**capabilities) if capabilities else None,
                name=config.name,
            )

        return create_openai

    if config.type == "ollama":

        def create_ollama() -> "LLMBackend":
            from omnillm.adapters.ollama_adapter import OllamaAdapter

            timeout = settings.get("timeout")
            return OllamaAdapter(host=settings.get("host"), timeout=float(timeout) if timeout is not None else None)

        return create_ollama

    def create_llamacpp() -> "LLMBackend":
        from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter

        return LlamaCPPAdapter()

    return create_llamacpp


def apply_config(manager: "LocalLLMManager", config: OmniConfig) -> None:
    """Registers the config's backends and aliases on a manager, validating cross-references."""
    where = f"{config.path}: " if config.path else ""
    for backend in config.backends:
        manager.register_backend(backend.name, _backend_factory(backend))
    if config.default_backend is not None:
        if config.default_backend not in manager.backend_names:
            raise ConfigError(f"{where}default_backend '{config.default_backend}' is not a known backend.")
        manager.default_backend = config.default_backend
    alias_names = {alias.name for alias in config.aliases} | set(manager.alias_names)
    for alias in config.aliases:
        if alias.backend is not None and alias.backend not in manager.backend_names:
            raise ConfigError(
                f"{where}[models.{alias.name}]: unknown backend '{alias.backend}'. "
                f"Known backends: {list(manager.backend_names)}"
            )
        for fallback in alias.fallbacks:
            if fallback == alias.name:
                raise ConfigError(f"{where}[models.{alias.name}]: an alias cannot fall back to itself.")
            prefix, sep, _ = fallback.partition("/")
            if fallback not in alias_names and sep and prefix not in manager.backend_names:
                raise ConfigError(
                    f"{where}[models.{alias.name}]: fallback '{fallback}' is neither an alias nor a "
                    f"'backend/model' reference to a known backend."
                )
        manager.register_alias(
            alias.name, model=alias.model, backend=alias.backend, fallbacks=alias.fallbacks, **alias.options
        )


_BARE_KEY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'


def toml_key(name: str) -> str:
    """`name` as a TOML key: bare when possible, quoted otherwise (so ``a.b`` isn't read as a nested table)."""
    return name if name and set(name) <= _BARE_KEY_CHARS else _toml_string(name)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot write {type(value).__name__} values to omnillm.toml.")


def alias_table(name: str, model: str, *, backend: str | None = None, options: Mapping[str, Any] | None = None) -> str:
    """The ``[models.<name>]`` TOML table for an alias."""
    lines = [f"[models.{toml_key(name)}]"]
    if backend is not None:
        lines.append(f"backend = {_toml_string(backend)}")
    lines.append(f"model = {_toml_string(model)}")
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in (options or {}).items() if value is not None)
    return "\n".join(lines)


def append_alias(config_path: str | Path, table: str, name: str) -> Path:
    """Appends an alias table to a config file (created if missing), validating before and after.

    Raises `InvalidRequestError` for a bad or existing name, and restores the file (raising `ConfigError`) if
    the result doesn't load.
    """
    if not name or "/" in name:
        raise InvalidRequestError("Alias names must be non-empty and must not contain '/'.")
    path = Path(config_path).expanduser()
    original: str | None = None
    if path.exists():
        existing = load_config(path)
        if any(alias.name == name for alias in existing.aliases):
            raise InvalidRequestError(f"Alias '{name}' already exists in {path}.")
        original = path.read_text()
        content = original.rstrip("\n") + "\n\n" + table + "\n"
    else:
        content = table + "\n"
    path.write_text(content)
    try:
        load_config(path)
    except ConfigError:
        if original is None:
            path.unlink()
        else:
            path.write_text(original)
        raise
    return path
