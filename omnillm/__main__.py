import argparse
import sys
from omnillm import LocalLLMManager

def main():
    parser = argparse.ArgumentParser(description="Omni-LLM CLI Interface")
    parser.add_argument("--backend", type=str, default="ollama", help="Backend to use (ollama, llama.cpp)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llama3, unsloth/llama-3-8b-GGUF)")
    parser.add_argument("--system", type=str, help="Optional system prompt")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    
    args = parser.parse_args()

    manager = LocalLLMManager()
    
    try:
        session = manager.create_session(backend=args.backend, model=args.model, system_prompt=args.system)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Started chat session with {args.backend} using model {args.model}.")
    print("Type 'exit' or 'quit' to end the session.\n")

    stream = not args.no_stream

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            print("AI: ", end="", flush=True)
            if stream:
                for chunk in session.send(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = session.send(user_input, stream=False)
                print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
