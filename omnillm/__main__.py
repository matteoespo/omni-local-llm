import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omnillm import ChatResponse, LocalLLMManager
from omnillm.core.config import ConfigError
from omnillm.harness.benchmark import BenchmarkHarness, BenchmarkTarget
from omnillm.harness.eval import EvalHarness, EvalTestCase
from omnillm.harness.needle import NeedleHarness


def _describe_target(manager: LocalLLMManager, args: argparse.Namespace) -> str:
    target = manager.resolve(args.model, args.backend)
    return f"{target.alias} → {target.ref}" if target.alias else target.ref


def _handle_chat(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    session = manager.create_session(
        backend=args.backend,
        model=args.model,
        system_prompt=args.system,
        max_turns=args.max_turns,
        max_tokens_budget=args.max_tokens,
    )

    print(f"🤖 Omni-Local-LLM Interactive Chat [{_describe_target(manager, args)}]")
    print("Commands: /clear, /reset [prompt], /tokens, /system <prompt>, /exit\n")

    options: dict[str, Any] = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.filename:
        options["filename"] = args.filename

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
                print("Goodbye!")
                return
            if user_input == "/clear":
                session.clear()
                print("🧹 Chat history cleared (system prompt preserved).\n")
                continue
            if user_input.startswith("/reset"):
                parts = user_input.split(" ", 1)
                new_sys = parts[1] if len(parts) > 1 else None
                session.reset(new_sys)
                print("🔄 Session reset.\n")
                continue
            if user_input == "/tokens":
                print(f"📊 Stored Turns: {session.turn_count} | Estimated Tokens: {session.estimated_tokens}\n")
                continue
            if user_input.startswith("/system"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    session.reset(parts[1])
                    print(f"⚙️ System prompt updated to: {parts[1]}\n")
                else:
                    print(f"Current system prompt: {session.system_prompt}\n")
                continue
            if user_input == "/help":
                print("Available commands:")
                print("  /clear          - Clear history, keep system prompt")
                print("  /reset [prompt] - Reset session completely with optional new system prompt")
                print("  /tokens         - Show active turns and estimated token count")
                print("  /system <text>  - Update or display system prompt")
                print("  /exit, /quit    - Exit the chat\n")
                continue

            print("AI: ", end="", flush=True)
            if not args.no_stream:
                for chunk in session.send(user_input, stream=True, **options):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = session.send(user_input, **options)
                print(f"{response.content}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            return
        except Exception as error:
            print(f"\n❌ Error: {error}\n", file=sys.stderr)


def _handle_run(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: No prompt provided. Specify a prompt argument or pipe via stdin.", file=sys.stderr)
        sys.exit(1)

    options: dict[str, Any] = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.filename:
        options["filename"] = args.filename

    if args.stream:
        stream = manager.chat(
            backend=args.backend,
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **options,
        )
        if isinstance(stream, Iterator):
            for chunk in stream:
                content = getattr(chunk, "content", "")
                if content:
                    print(content, end="", flush=True)
            print()
    else:
        response = manager.chat(
            backend=args.backend,
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        if isinstance(response, ChatResponse):
            print(response.content)


def _handle_models(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    try:
        models = manager.list_models(backend=args.backend)
        aliases = {} if args.backend else manager.aliases
    except Exception as error:
        print(f"Error listing models: {error}", file=sys.stderr)
        sys.exit(1)

    if aliases:
        print(f"{'ALIAS':<15} {'TARGET':<45} {'FALLBACKS'}")
        print("-" * 78)
        for name, target in aliases.items():
            print(f"{name:<15} {target.ref:<45} {', '.join(target.fallbacks) or '-'}")
        print()

    if not models:
        print("No models detected.")
        return

    print(f"{'BACKEND':<15} {'MODEL'}")
    print("-" * 50)
    for backend, model in models:
        print(f"{backend:<15} {model}")


def _handle_bench(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    harness = BenchmarkHarness(manager)
    targets: list[BenchmarkTarget] = []

    if args.targets:
        for t in args.targets.split(","):
            t = t.strip()
            if t:
                targets.append(BenchmarkTarget.from_spec(t))
    elif args.backend is None:
        targets.append(harness._target(args.model))
    else:
        targets.append(BenchmarkTarget(backend=args.backend, model=args.model))

    print(f"Running benchmark across {len(targets)} target(s) ({args.runs} runs each)...", file=sys.stderr)
    suite = harness.run(
        targets=targets,
        prompt=args.prompt,
        max_tokens=args.tokens,
        runs=args.runs,
        warmup=not args.no_warmup,
    )

    if args.json:
        print(json.dumps(suite.to_dict(), indent=2))
    else:
        print("\n" + suite.format_table() + "\n")


def _handle_eval(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    if args.needle:
        needle_harness = NeedleHarness(manager)
        depths = [int(d.strip()) for d in args.depths.split(",")] if args.depths else (0, 25, 50, 75, 100)
        print(f"Running Needle in a Haystack evaluation on {_describe_target(manager, args)}...", file=sys.stderr)
        result = needle_harness.run(
            backend=args.backend,
            model=args.model,
            depths=depths,
            secret_code=args.secret_code,
        )
        print("\n" + result.format_table() + "\n")
        return

    eval_harness = EvalHarness(manager)
    if args.suite:
        test_cases = eval_harness.load_jsonl_suite(args.suite)
    else:
        # Default built-in reasoning & instruction following test suite
        test_cases = [
            EvalTestCase(
                name="Capital of France",
                prompt="What is the capital of France? Reply in one word.",
                expected_pattern=r"Paris",
            ),
            EvalTestCase(
                name="Simple Arithmetic",
                prompt="Compute 45 + 55. Reply with just the numeric answer.",
                expected_pattern=r"100",
            ),
            EvalTestCase(
                name="JSON Object adherence",
                prompt="Return a JSON object with keys 'status' (value 'ok') and 'code' (value 200).",
                expected_pattern=r'"status":\s*"ok"',
            ),
        ]

    print(
        f"Running evaluation suite ({len(test_cases)} tests) on {_describe_target(manager, args)}...",
        file=sys.stderr,
    )
    suite_result = eval_harness.run_suite(backend=args.backend, model=args.model, test_cases=test_cases)

    if args.json:
        print(json.dumps(suite_result.to_dict(), indent=2))
    else:
        print("\n" + suite_result.format_table() + "\n")


def _handle_serve(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    try:
        import uvicorn

        from omnillm.server import create_app
    except ImportError:
        print("The API server requires optional dependencies: install omni-local-llm[server]", file=sys.stderr)
        sys.exit(1)

    api_key = _serve_api_key(args)
    allow_downloads = True if getattr(args, "allow_downloads", False) else None  # None: $OMNILLM_ALLOW_DOWNLOADS
    app = create_app(
        manager=manager,
        allowed_hosts=_serve_allowed_hosts(args),
        api_key=api_key,
        allow_downloads=allow_downloads,
    )
    print(f"Starting Omni-Local-LLM OpenAI-Compatible API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def _serve_api_key(args: argparse.Namespace) -> str | None:
    """The key clients must send: --api-key-file, else $OMNILLM_API_KEY (None: no key). Warns when exposed without one."""
    import os

    from omnillm.server import API_KEY_ENV, LOOPBACK_HOSTS

    key_file = getattr(args, "api_key_file", None)
    if key_file:
        try:
            key = Path(key_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            print(f"❌ Cannot read --api-key-file: {error}", file=sys.stderr)
            sys.exit(2)
        if not key:
            print(f"❌ --api-key-file {key_file} is empty.", file=sys.stderr)
            sys.exit(2)
        return key
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return key
    if args.host not in LOOPBACK_HOSTS:
        print(
            f"⚠️  No API key set while listening on {args.host}: anyone who can reach this port can use your models. "
            f"Set {API_KEY_ENV} or pass --api-key-file.",
            file=sys.stderr,
        )
    return None


def _serve_allowed_hosts(args: argparse.Namespace) -> list[str] | None:
    """Host headers to accept: --allowed-host, else $OMNILLM_ALLOWED_HOSTS, else loopback plus the bind address."""
    import os
    import socket

    from omnillm.server import ALLOWED_HOSTS_ENV, LOOPBACK_HOSTS

    if getattr(args, "allowed_host", None):
        return list(args.allowed_host)
    if os.environ.get(ALLOWED_HOSTS_ENV):
        return None
    host = args.host
    if host in LOOPBACK_HOSTS:
        return list(LOOPBACK_HOSTS)
    if host in ("0.0.0.0", "::", ""):
        names = [*LOOPBACK_HOSTS, socket.gethostname()]
        print(
            f"⚠️  Listening on all interfaces. Accepting Host headers: {', '.join(names)}. "
            "Add --allowed-host <name-or-IP> for each address clients use (or '*' to accept any).",
            file=sys.stderr,
        )
        return names
    return [*LOOPBACK_HOSTS, host]


def _handle_fit(args: argparse.Namespace, manager: LocalLLMManager | None = None) -> None:
    from omnillm.core.errors import OmniLLMError
    from omnillm.core.fit import alias_snippet, analyze_model_fit, pull_quant, save_alias, select_result
    from omnillm.core.hardware import QuantFitStatus, detect_hardware, suggest_models_for_hardware

    hardware = detect_hardware()

    if not args.model:
        suggestions = suggest_models_for_hardware(hardware)
        if args.json:
            print(json.dumps({"hardware": hardware.to_dict(), "suggestions": suggestions}, indent=2))
            return

        print("🖥️  Detected Hardware Profile")
        print("-" * 78)
        print(
            f"OS:           {hardware.os_name} {hardware.os_release} ({hardware.architecture}, {hardware.cpu_count} CPUs)"
        )
        print(f"System RAM:   {hardware.total_ram_gb:.2f} GB Total ({hardware.available_ram_gb:.2f} GB Available)")
        if hardware.has_cuda and hardware.gpu_name:
            print(
                f"GPU:          {hardware.gpu_name} ({hardware.total_vram_gb:.2f} GB VRAM, {hardware.free_vram_gb:.2f} GB Free)"
            )
        elif hardware.is_apple_silicon:
            print(f"GPU:          {hardware.gpu_name or 'Apple Silicon Metal'} (Unified Memory Architecture)")
        else:
            print("Accelerator:  CPU Only (No dedicated GPU detected)")
        print("-" * 78)
        print("\n🚀 Model Compatibility Matrix for Your Machine:")
        print(f"{'MODEL':<20} {'SIZE':<8} {'FIT TARGET':<34} {'QUANT':<10} {'EST. MEMORY'}")
        print("-" * 78)
        for s in suggestions:
            target_str = str(s["target"])
            if target_str == QuantFitStatus.PARTIAL.value:
                target_str = f"GPU+CPU ({s['n_gpu_layers']} layers on GPU)"
            icon = (
                "✅"
                if s["target"] in (QuantFitStatus.GPU_FULL.value, QuantFitStatus.METAL_UNIFIED.value)
                else ("⚠️" if s["can_fit"] else "❌")
            )
            print(
                f"{s['model']:<20} {s['params']:<8} {icon + ' ' + target_str:<34} {s['recommended_quant'] or 'N/A':<10} {s['required_gb']:.2f} GB"
            )
        print("-" * 78)
        print("\n💡 Tip: Run 'omnillm fit <model>' with a Hugging Face repo (e.g. unsloth/Qwen3-8B-GGUF), an installed")
        print("   Ollama model, a config alias, or a .gguf file for an exact per-quant breakdown.")
        return

    try:
        rec = analyze_model_fit(
            args.model,
            context_length=args.context,
            manager=manager,
            hardware=hardware,
            kv_cache_type=getattr(args, "kv_cache", "f16"),
            offline=getattr(args, "offline", False),
            param_count_b=args.params,
        )
    except (OmniLLMError, ValueError) as error:
        print(f"❌ {error}", file=sys.stderr)
        sys.exit(1)

    pull = getattr(args, "pull", False)
    quant = getattr(args, "quant", None)
    alias_name = getattr(args, "save_alias", None)
    if not args.json:
        print("\n" + rec.format_table() + "\n")
    if alias_name and not pull:
        try:
            result = select_result(rec, quant)
            saved = save_alias(rec, result, alias_name, _alias_config_path(args))
        except (OmniLLMError, ValueError) as error:
            print(f"❌ {error}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Saved alias '{alias_name}' ({result.quant}) to {_display_path(saved)}. It downloads on first use:")
        print(f'   omnillm run --model {alias_name} "Hello!"')
        return
    if not pull:
        if args.json:
            print(json.dumps(rec.to_dict(), indent=2))
        elif rec.source == "huggingface-gguf" and rec.recommended is not None:
            print(
                f"⬇️  Download it with: omnillm fit {args.model} --pull --save-alias <name>"
                + (f" --context {rec.context_length}" if args.context else "")
            )
        return

    try:
        path, result = pull_quant(
            rec, quant, progress=lambda name: print(f"⬇️  Downloading {name} ...", file=sys.stderr)
        )
    except (OmniLLMError, ValueError) as error:
        print(f"❌ {error}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps({**rec.to_dict(), "downloaded": {"quant": result.quant, "path": str(path)}}, indent=2))
        return
    print(f"✅ {result.quant} downloaded to {_display_path(path)}\n")
    if alias_name:
        try:
            saved = save_alias(rec, result, alias_name, _alias_config_path(args))
        except (OmniLLMError, ValueError) as error:
            print(f"❌ {error}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Saved alias '{alias_name}' to {_display_path(saved)}. Run it with:")
        print(f'   omnillm run --model {alias_name} "Hello!"')
        return
    print("Use it:")
    print(f"  omnillm chat --backend llama.cpp --model {rec.repo} --filename {result.files[0]}\n")
    print("Or add this alias to omnillm.toml (or pass --save-alias NAME) and run `omnillm chat --model <alias>`:\n")
    print("  " + alias_snippet(rec, result).replace("\n", "\n  ") + "\n")


def _display_path(path: Path) -> str:
    """Shortens paths for display: ./relative under the working directory, ~/... under the home directory."""
    resolved = Path(path).expanduser().resolve()
    for base, prefix in ((Path.cwd().resolve(), "./"), (Path.home().resolve(), "~/")):
        if resolved.is_relative_to(base):
            return prefix + str(resolved.relative_to(base))
    return str(path)


def _alias_config_path(args: argparse.Namespace) -> Path:
    """The config file in effect (so a user config isn't shadowed), else ./omnillm.toml."""
    from omnillm.core.config import LOCAL_CONFIG_NAME, find_config_file

    return find_config_file(getattr(args, "config", None)) or Path.cwd() / LOCAL_CONFIG_NAME


def build_parser() -> argparse.ArgumentParser:
    # Shared options. SUPPRESS keeps a subcommand from overwriting a value given before the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Path to an omnillm.toml config (default: $OMNILLM_CONFIG, ./omnillm.toml, ~/.config/omnillm/config.toml)",
    )
    common.add_argument(
        "--base-url",
        default=argparse.SUPPRESS,
        help="Use an OpenAI-compatible server (e.g. http://localhost:1234/v1 for LM Studio) without a config file",
    )
    common.add_argument("--api-key", default=argparse.SUPPRESS, help="API key for --base-url")
    parser = argparse.ArgumentParser(
        prog="omnillm",
        description="Omni-Local-LLM: Unified local LLM CLI, server, and evaluation harness.",
        parents=[common],
    )

    backend_help = "Backend name (default: taken from the model alias or 'backend/model' prefix, else ollama)"
    model_help = "Model: an alias, 'backend/model', or a model name"
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Chat
    chat_p = subparsers.add_parser("chat", help="Start an interactive chat session", parents=[common])
    chat_p.add_argument("--backend", default=None, help=backend_help)
    chat_p.add_argument("--model", required=True, help=model_help)
    chat_p.add_argument("--filename", help="GGUF filename for llama.cpp")
    chat_p.add_argument("--system", help="System prompt")
    chat_p.add_argument("--temperature", type=float, help="Sampling temperature")
    chat_p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    chat_p.add_argument("--max-turns", type=int, help="Maximum conversational turns to retain")
    chat_p.add_argument("--max-tokens", type=int, help="Context token budget for sliding-window pruning")

    # 2. Run (one-shot)
    run_p = subparsers.add_parser("run", help="Execute a one-shot prompt from argument or stdin", parents=[common])
    run_p.add_argument("prompt", nargs="?", default="", help="Prompt text")
    run_p.add_argument("--backend", default=None, help=backend_help)
    run_p.add_argument("--model", required=True, help=model_help)
    run_p.add_argument("--filename", help="GGUF filename for llama.cpp")
    run_p.add_argument("--temperature", type=float, help="Sampling temperature")
    run_p.add_argument("--stream", action="store_true", help="Stream the response")

    # 3. Models
    models_p = subparsers.add_parser("models", help="List aliases and models from all backends", parents=[common])
    models_p.add_argument("--backend", help="Only list models from this backend")

    # 4. Benchmark
    bench_p = subparsers.add_parser("bench", help="Benchmark performance (TTFT and throughput)", parents=[common])
    bench_p.add_argument(
        "--targets", help="Comma-separated aliases or 'backend/model' refs (e.g. 'coder,ollama/llama3')"
    )
    bench_p.add_argument("--backend", default=None, help=backend_help)
    bench_p.add_argument("--model", default="llama3", help=model_help)
    bench_p.add_argument("--prompt", default="Explain the theory of relativity in 3 sentences.", help="Prompt")
    bench_p.add_argument("--tokens", type=int, default=128, help="Maximum tokens to generate")
    bench_p.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    bench_p.add_argument("--no-warmup", action="store_true", help="Skip initial warmup run")
    bench_p.add_argument("--json", action="store_true", help="Output raw JSON metrics")

    # 5. Eval
    eval_p = subparsers.add_parser(
        "eval", help="Run model evaluations, test suites, or needle-in-haystack", parents=[common]
    )
    eval_p.add_argument("--backend", default=None, help=backend_help)
    eval_p.add_argument("--model", required=True, help=model_help)
    eval_p.add_argument("--suite", help="Path to JSONL evaluation test suite file")
    eval_p.add_argument("--needle", action="store_true", help="Run Needle-In-A-Haystack context evaluation")
    eval_p.add_argument("--secret-code", default="BLUE-TITAN-42", help="Secret needle password")
    eval_p.add_argument("--depths", default="0,25,50,75,100", help="Comma-separated depths for needle test")
    eval_p.add_argument("--json", action="store_true", help="Output raw JSON results")

    # 6. Serve
    serve_p = subparsers.add_parser("serve", help="Launch the OpenAI-compatible FastAPI server", parents=[common])
    serve_p.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_p.add_argument("--port", type=int, default=8000, help="Port number")
    serve_p.add_argument(
        "--allowed-host",
        action="append",
        metavar="HOST",
        help="Accept requests with this Host header (repeatable; '*.domain' wildcards, '*' for any). "
        "Defaults to loopback names plus the bind address, to block DNS-rebinding attacks.",
    )
    serve_p.add_argument(
        "--api-key-file",
        metavar="PATH",
        help="Require clients to send the key in this file as 'Authorization: Bearer <key>' "
        "(default: $OMNILLM_API_KEY; no key if unset). /health stays public.",
    )
    serve_p.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Let API requests download models that aren't installed (Ollama pulls, Hugging Face GGUFs). "
        "Off by default; also enabled by OMNILLM_ALLOW_DOWNLOADS=1.",
    )

    # 7. Fit (Hardware & Quantization Recommender)
    fit_p = subparsers.add_parser(
        "fit", help="Check which quantization of a model fits this machine (and download it)", parents=[common]
    )
    fit_p.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Hugging Face repo (unsloth/Qwen3-8B-GGUF), installed Ollama model (llama3.2:3b), config alias, "
            ".gguf file, or a name to estimate (llama-3.1-8b)"
        ),
    )
    fit_p.add_argument(
        "--context", type=int, default=None, help="Context window in tokens (default: alias n_ctx or 4096)"
    )
    fit_p.add_argument(
        "--kv-cache",
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="KV cache type (llama.cpp --cache-type-k/v, Ollama OLLAMA_KV_CACHE_TYPE)",
    )
    fit_p.add_argument("--pull", action="store_true", help="Download the recommended GGUF (Hugging Face repos)")
    fit_p.add_argument("--quant", help="With --pull: download this quant instead of the recommendation (e.g. Q5_K_M)")
    fit_p.add_argument("--offline", action="store_true", help="Don't query Hugging Face or Ollama; estimate only")
    fit_p.add_argument(
        "--save-alias",
        metavar="NAME",
        help="Save the recommended (or --quant) GGUF as an alias in your omnillm.toml (Hugging Face repos)",
    )
    fit_p.add_argument("--params", type=float, help="Explicit parameter count in billions (overrides auto-detection)")
    fit_p.add_argument("--json", action="store_true", help="Output raw JSON analysis")

    # Top-level fallback arguments for backwards compatibility (e.g. `python -m omnillm --model llama3`)
    parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--filename", help=argparse.SUPPRESS)
    parser.add_argument("--system", help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--no-stream", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-turns", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)

    return parser


def build_manager(args: argparse.Namespace) -> LocalLLMManager:
    """Loads the config file and applies an ad-hoc --base-url backend, if given."""
    manager = LocalLLMManager.from_config(getattr(args, "config", None))
    base_url = getattr(args, "base_url", None)
    if base_url:
        from omnillm.adapters.openai_adapter import OpenAICompatibleAdapter

        name = getattr(args, "backend", None) or "openai"
        api_key = getattr(args, "api_key", None)
        manager.register_backend(name, lambda: OpenAICompatibleAdapter(base_url, api_key=api_key, name=name))
        if hasattr(args, "backend"):
            args.backend = name
        manager.default_backend = name
    return manager


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manager = build_manager(args)
    except ConfigError as error:
        print(f"❌ Config error: {error}", file=sys.stderr)
        sys.exit(2)

    if args.subcommand == "chat":
        _handle_chat(args, manager)
    elif args.subcommand == "run":
        _handle_run(args, manager)
    elif args.subcommand == "models":
        _handle_models(args, manager)
    elif args.subcommand == "bench":
        _handle_bench(args, manager)
    elif args.subcommand == "eval":
        _handle_eval(args, manager)
    elif args.subcommand == "serve":
        _handle_serve(args, manager)
    elif args.subcommand == "fit":
        _handle_fit(args, manager)
    else:
        # Fallback to chat if --model was provided at top-level
        if getattr(args, "model", None):
            _handle_chat(args, manager)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
