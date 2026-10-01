import argparse
import sys
from unittest.mock import patch

from omnillm import ChatChunk, ChatResponse, LocalLLMManager
from omnillm.__main__ import (
    _handle_bench,
    _handle_eval,
    _handle_models,
    _handle_run,
    build_parser,
    main,
)
from tests.fakes import RecordingBackend


def test_build_parser_subcommands():
    parser = build_parser()

    # Chat
    args = parser.parse_args(["chat", "--model", "llama3", "--max-turns", "5"])
    assert args.subcommand == "chat"
    assert args.model == "llama3"
    assert args.max_turns == 5

    # Run
    args = parser.parse_args(["run", "--model", "llama3", "Hello world"])
    assert args.subcommand == "run"
    assert args.prompt == "Hello world"

    # Bench
    args = parser.parse_args(["bench", "--targets", "ollama/llama3", "--runs", "2"])
    assert args.subcommand == "bench"
    assert args.targets == "ollama/llama3"
    assert args.runs == 2

    # Eval
    args = parser.parse_args(["eval", "--model", "llama3", "--needle"])
    assert args.subcommand == "eval"
    assert args.needle is True

    # Models
    args = parser.parse_args(["models", "--backend", "ollama"])
    assert args.subcommand == "models"
    assert args.backend == "ollama"

    # Serve
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.subcommand == "serve"
    assert args.port == 9000

    # Fit
    args = parser.parse_args(["fit", "llama-3.1-8b", "--context", "8192"])
    assert args.subcommand == "fit"
    assert args.model == "llama-3.1-8b"
    assert args.context == 8192


def test_cli_handle_run(capsys):
    backend = RecordingBackend(response=ChatResponse(content="One-shot response"))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        backend="fake",
        model="test",
        prompt="Tell me something",
        temperature=None,
        filename=None,
        stream=False,
    )
    _handle_run(args, manager)

    captured = capsys.readouterr()
    assert "One-shot response" in captured.out


def test_cli_handle_run_stream(capsys):
    backend = RecordingBackend(chunks=(ChatChunk(content="Streamed "), ChatChunk(content="chunk")))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        backend="fake",
        model="test",
        prompt="Stream this",
        temperature=0.7,
        filename=None,
        stream=True,
    )
    _handle_run(args, manager)

    captured = capsys.readouterr()
    assert "Streamed chunk" in captured.out


def test_cli_handle_models(capsys):
    backend = RecordingBackend(models=("m1", "m2"))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(backend="fake")
    _handle_models(args, manager)

    captured = capsys.readouterr()
    assert "fake" in captured.out
    assert "m1" in captured.out
    assert "m2" in captured.out


def test_cli_handle_bench(capsys):
    backend = RecordingBackend(chunks=(ChatChunk(content="Tok 1 "), ChatChunk(content="Tok 2")))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        targets=None,
        backend="fake",
        model="test",
        prompt="Test prompt",
        tokens=32,
        runs=1,
        no_warmup=True,
        json=False,
    )
    _handle_bench(args, manager)

    captured = capsys.readouterr()
    assert "Tokens/sec" in captured.out
    assert "fake" in captured.out


def test_cli_handle_eval(capsys):
    backend = RecordingBackend(response=ChatResponse(content="The answer is Paris."))
    manager = LocalLLMManager({"fake": backend})

    args = argparse.Namespace(
        needle=False,
        suite=None,
        backend="fake",
        model="test",
        json=False,
    )
    _handle_eval(args, manager)

    captured = capsys.readouterr()
    assert "Model Evaluation" in captured.out
    assert "PASS" in captured.out


def test_cli_handle_fit_with_model(capsys):
    from omnillm.__main__ import _handle_fit

    args = argparse.Namespace(
        model="llama-3.2-3b",
        context=4096,
        params=None,
        json=False,
    )
    _handle_fit(args)
    captured = capsys.readouterr()
    assert "llama-3.2-3b" in captured.out
    assert "Detected Hardware Profile" in captured.out
    assert "QUANT" in captured.out


def test_cli_handle_fit_matrix(capsys):
    from omnillm.__main__ import _handle_fit

    args = argparse.Namespace(
        model=None,
        context=4096,
        params=None,
        json=False,
    )
    _handle_fit(args)
    captured = capsys.readouterr()
    assert "Model Compatibility Matrix" in captured.out
    assert "smollm:135m" in captured.out


def test_cli_main_help_output(capsys):
    with patch.object(sys, "argv", ["omnillm"]):
        main()
    captured = capsys.readouterr()
    assert "Omni-Local-LLM" in captured.out
    assert "chat" in captured.out
    assert "bench" in captured.out
    assert "fit" in captured.out
