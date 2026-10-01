import argparse
import json
import sys
from collections.abc import Iterator
from typing import Any

from omnillm import ChatResponse, LocalLLMManager
from omnillm.harness.benchmark import BenchmarkHarness, BenchmarkTarget
from omnillm.harness.eval import EvalHarness, EvalTestCase
from omnillm.harness.needle import NeedleHarness


def _handle_chat(args: argparse.Namespace, manager: LocalLLMManager) -> None:
    session = manager.create_session(
        backend=args.backend,
        model=args.model,
        system_prompt=args.system,
        max_turns=args.max_turns,
        max_tokens_budget=args.max_tokens,
    )

    print(f"🤖 Omni-Local-LLM Interactive Chat [{args.backend}/{args.model}]")
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
    except Exception as error:
        print(f"Error listing models: {error}", file=sys.stderr)
        sys.exit(1)

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
    else:
        targets.append(BenchmarkTarget(backend=args.backend, model=args.model))

    print(f"Running benchmark across {len(targets)} target(s) ({args.runs} runs each)...")
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
        print(f"Running Needle in a Haystack evaluation on {args.backend}/{args.model}...")
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

    print(f"Running evaluation suite ({len(test_cases)} tests) on {args.backend}/{args.model}...")
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

    app = create_app(manager=manager)
    print(f"Starting Omni-Local-LLM OpenAI-Compatible API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def _handle_fit(args: argparse.Namespace) -> None:
    from omnillm.core.hardware import (
        detect_hardware,
        recommend_model_fit,
        suggest_models_for_hardware,
    )

    hardware = detect_hardware()

    if not args.model:
        suggestions = suggest_models_for_hardware(hardware)
        if args.json:
            print(
                json.dumps(
                    {
                        "hardware": hardware.to_dict(),
                        "suggestions": suggestions,
                    },
                    indent=2,
                )
            )
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
        print(f"{'MODEL':<20} {'SIZE':<8} {'FIT TARGET':<26} {'QUANT':<10} {'EST. RAM'}")
        print("-" * 78)
        for s in suggestions:
            target_str = str(s["target"])
            icon = (
                "✅"
                if s["can_fit"] and ("GPU" in target_str or "Metal" in target_str)
                else ("⚠️" if s["can_fit"] else "❌")
            )
            print(
                f"{s['model']:<20} {s['params']:<8} {icon + ' ' + target_str:<26} {s['recommended_quant'] or 'N/A':<10} {s['required_gb']:.2f} GB"
            )
        print("-" * 78)
        print("\n💡 Tip: Run 'omnillm fit <model_name>' for a detailed breakdown of a specific model.")
        return

    rec = recommend_model_fit(
        model_name=args.model,
        context_length=args.context,
        param_count_b=args.params,
        hardware=hardware,
    )

    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print("\n" + rec.format_table() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnillm",
        description="Omni-Local-LLM: Unified local LLM CLI, server, and evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Chat
    chat_p = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_p.add_argument("--backend", default="ollama", help="Backend to use (ollama, llama.cpp)")
    chat_p.add_argument("--model", required=True, help="Model name")
    chat_p.add_argument("--filename", help="GGUF filename for llama.cpp")
    chat_p.add_argument("--system", help="System prompt")
    chat_p.add_argument("--temperature", type=float, help="Sampling temperature")
    chat_p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    chat_p.add_argument("--max-turns", type=int, help="Maximum conversational turns to retain")
    chat_p.add_argument("--max-tokens", type=int, help="Context token budget for sliding-window pruning")

    # 2. Run (one-shot)
    run_p = subparsers.add_parser("run", help="Execute a one-shot prompt from argument or stdin")
    run_p.add_argument("prompt", nargs="?", default="", help="Prompt text")
    run_p.add_argument("--backend", default="ollama", help="Backend to use (ollama, llama.cpp)")
    run_p.add_argument("--model", required=True, help="Model name")
    run_p.add_argument("--filename", help="GGUF filename for llama.cpp")
    run_p.add_argument("--temperature", type=float, help="Sampling temperature")
    run_p.add_argument("--stream", action="store_true", help="Stream the response")

    # 3. Models
    models_p = subparsers.add_parser("models", help="List detected or installed models")
    models_p.add_argument("--backend", help="Filter by backend (ollama, llama.cpp)")

    # 4. Benchmark
    bench_p = subparsers.add_parser("bench", help="Benchmark performance (TTFT and throughput)")
    bench_p.add_argument("--targets", help="Comma-separated targets (e.g. 'ollama/llama3,llama.cpp/model')")
    bench_p.add_argument("--backend", default="ollama", help="Default backend")
    bench_p.add_argument("--model", default="llama3", help="Default model")
    bench_p.add_argument("--prompt", default="Explain the theory of relativity in 3 sentences.", help="Prompt")
    bench_p.add_argument("--tokens", type=int, default=128, help="Maximum tokens to generate")
    bench_p.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    bench_p.add_argument("--no-warmup", action="store_true", help="Skip initial warmup run")
    bench_p.add_argument("--json", action="store_true", help="Output raw JSON metrics")

    # 5. Eval
    eval_p = subparsers.add_parser("eval", help="Run model evaluations, test suites, or needle-in-haystack")
    eval_p.add_argument("--backend", default="ollama", help="Backend to evaluate")
    eval_p.add_argument("--model", required=True, help="Model to evaluate")
    eval_p.add_argument("--suite", help="Path to JSONL evaluation test suite file")
    eval_p.add_argument("--needle", action="store_true", help="Run Needle-In-A-Haystack context evaluation")
    eval_p.add_argument("--secret-code", default="BLUE-TITAN-42", help="Secret needle password")
    eval_p.add_argument("--depths", default="0,25,50,75,100", help="Comma-separated depths for needle test")
    eval_p.add_argument("--json", action="store_true", help="Output raw JSON results")

    # 6. Serve
    serve_p = subparsers.add_parser("serve", help="Launch the OpenAI-compatible FastAPI server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_p.add_argument("--port", type=int, default=8000, help="Port number")

    # 7. Fit (Hardware & Quantization Recommender)
    fit_p = subparsers.add_parser("fit", help="Auto-detect hardware and recommend optimal model quantization")
    fit_p.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model name or alias to evaluate (e.g. 'llama-3.1-8b', 'mistral:7b')",
    )
    fit_p.add_argument("--context", type=int, default=4096, help="Context window token size (default: 4096)")
    fit_p.add_argument("--params", type=float, help="Explicit parameter count in billions (overrides auto-detection)")
    fit_p.add_argument("--json", action="store_true", help="Output raw JSON analysis")

    # Top-level fallback arguments for backwards compatibility (e.g. `python -m omnillm --model llama3`)
    parser.add_argument("--backend", default="ollama", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--filename", help=argparse.SUPPRESS)
    parser.add_argument("--system", help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--no-stream", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-turns", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manager = LocalLLMManager()

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
        _handle_fit(args)
    else:
        # Fallback to chat if --model was provided at top-level
        if getattr(args, "model", None):
            _handle_chat(args, manager)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
