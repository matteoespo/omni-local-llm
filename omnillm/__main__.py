import argparse
import sys

from omnillm import LocalLLMManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Omni-Local-LLM CLI Interface")
    parser.add_argument("--backend", default="ollama", help="Backend to use (ollama, llama.cpp)")
    parser.add_argument("--model", required=True, help="Model name (for example llama3 or a Hugging Face repository)")
    parser.add_argument("--filename", help="GGUF filename required by the llama.cpp backend")
    parser.add_argument("--system", help="Optional system prompt")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    manager = LocalLLMManager()
    try:
        session = manager.create_session(backend=args.backend, model=args.model, system_prompt=args.system)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Started chat session with {args.backend} using model {args.model}.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                return
            if not user_input.strip():
                continue

            options = {"temperature": args.temperature}
            if args.filename:
                options["filename"] = args.filename

            print("AI: ", end="", flush=True)
            if not args.no_stream:
                for chunk in session.send(user_input, stream=True, **options):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = session.send(user_input, **options)
                print(response.content)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return
        except Exception as error:
            print(f"\nAn error occurred: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
