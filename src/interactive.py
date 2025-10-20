import os
import json
import argparse
import logging
from copy import copy

from generate import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    args.config_path = cfg
    if "use_counter" not in args:
        args.use_counter = True
    return args


def choose_model(args):
    if args.method == "non-retrieval":
        return BasicRAG(args)
    elif args.method == "single-retrieval":
        return SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        return FixLengthRAG(args)
    elif args.method == "token":
        return TokenRAG(args)
    elif args.method == "entity":
        return EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        return AttnWeightRAG(args)
    else:
        raise NotImplementedError(f"Unknown method: {args.method}")


def interactive_loop(model, args):
    print("Interactive session started. Type /exit to quit. /help for commands.")
    history = []
    session_counter = copy(model.counter)

    def print_help():
        print("Commands:")
        print("  /exit            - exit the session")
        print("  /reset           - reset conversation history and counters")
        print("  /history         - show past Q/A pairs in this session")
        print("  /save <path>     - save session history to a jsonl file")
        print("  /show_counter    - show generation/retrieval counters")
        print("  /help            - show this help")

    print_help()

    try:
        while True:
            user = input("User: ").strip()
            if len(user) == 0:
                continue
            if user.startswith("/"):
                parts = user.split()
                cmd = parts[0]
                if cmd == "/exit":
                    break
                elif cmd == "/help":
                    print_help()
                    continue
                elif cmd == "/reset":
                    history = []
                    model.counter = copy(session_counter)
                    print("Session reset.")
                    continue
                elif cmd == "/history":
                    if not history:
                        print("No history yet.")
                    else:
                        for i, h in enumerate(history):
                            print(f"[{i+1}] Q: {h['question']}\n    A: {h['answer']}")
                    continue
                elif cmd == "/save":
                    if len(parts) < 2:
                        print("Usage: /save <path>")
                        continue
                    path = parts[1]
                    try:
                        with open(path, "w") as f:
                            for h in history:
                                f.write(json.dumps(h) + "\n")
                        print(f"Saved {len(history)} entries to {path}")
                    except Exception as e:
                        print(f"Failed to save: {e}")
                    continue
                elif cmd == "/show_counter":
                    if hasattr(model, "counter"):
                        print(json.dumps(model.counter.calc(session_counter), indent=2))
                    else:
                        print("No counter available on model.")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Normal question flow. We keep the demo and case empty by default
            question = user
            demo = []
            case = ""  # no fixed case; user can include instruction in the question

            print("Thinking...")
            try:
                last_counter = copy(model.counter)
                answer = model.inference(question, demo, case)
            except Exception as e:
                print(f"Error during model inference: {e}")
                continue

            answer = answer.strip()
            print(f"Assistant: {answer}\n")

            rec = {"question": question, "answer": answer}
            if args.use_counter:
                rec.update(model.counter.calc(last_counter))
            history.append(rec)

    except KeyboardInterrupt:
        print("\nExiting interactive session.")

    return history


def main():
    args = get_args()
    logger.info(f"Loaded config from {args.config_path}")

    model = choose_model(args)

    history = interactive_loop(model, args)

    # On exit, offer to save history
    if history:
        save_path = os.path.join(os.getcwd(), "interactive_history.jsonl")
        try:
            with open(save_path, "w") as f:
                for h in history:
                    f.write(json.dumps(h) + "\n")
            print(f"Session history saved to {save_path}")
        except Exception as e:
            print(f"Failed to save session history: {e}")


if __name__ == "__main__":
    main()
