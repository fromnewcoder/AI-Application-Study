#!/usr/bin/env python3
"""
Multi-turn CLI Chatbot with Streaming LLM API
Uses raw HTTP requests — no SDKs or frameworks.
"""

import os
import sys
import json
import time
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
MODEL = "MiniMax-M2.5"
MAX_TOKENS = 1024


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("\033[91m[ERROR] ANTHROPIC_API_KEY not set. Export it or add it to a .env file.\033[0m")
        sys.exit(1)
    return key


def build_headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def build_payload(history: list) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": history,
    }


# ── Spinner ────────────────────────────────────────────────────────────────────

class Spinner:
    _frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking"):
        self._label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self._frames[i % len(self._frames)]
            sys.stdout.write(f"\r\033[36m{frame} {self._label}...\033[0m")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()


# ── Streaming ─────────────────────────────────────────────────────────────────

def send_message(history: list, api_key: str) -> str:
    """
    Send conversation history to the API with streaming enabled.
    Returns the fully assembled assistant reply.
    """
    headers = build_headers(api_key)
    payload = build_payload(history)

    spinner = Spinner()
    spinner.start()
    first_token_received = False
    assembled = []

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        spinner.stop()
        raise RuntimeError("Network error: could not reach the API. Check your connection.")
    except requests.exceptions.Timeout:
        spinner.stop()
        raise RuntimeError("Request timed out. The server took too long to respond.")

    # ── HTTP-level errors ──────────────────────────────────────────────────────
    if response.status_code == 401:
        spinner.stop()
        print("\033[91m[AUTH ERROR] Invalid API key (401). Check ANTHROPIC_API_KEY.\033[0m")
        sys.exit(1)

    if response.status_code == 429:
        spinner.stop()
        retry_after = int(response.headers.get("retry-after", 10))
        print(f"\033[93m[RATE LIMIT] Too many requests (429). Retrying in {retry_after}s...\033[0m")
        time.sleep(retry_after)
        return send_message(history, api_key)  # one retry

    if response.status_code != 200:
        spinner.stop()
        raw = response.text[:300]
        raise RuntimeError(f"API error {response.status_code}: {raw}")

    # ── Parse SSE stream ───────────────────────────────────────────────────────
    print("\033[32mAssistant:\033[0m ", end="", flush=True)

    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

            if not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"\n\033[93m[WARN] Could not parse chunk: {data_str[:80]}\033[0m")
                continue

            event_type = event.get("type", "")

            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    token = delta.get("text", "")
                    if token:
                        if not first_token_received:
                            spinner.stop()
                            first_token_received = True
                            print("\033[32mAssistant:\033[0m ", end="", flush=True)
                        print(token, end="", flush=True)
                        assembled.append(token)

            elif event_type == "message_stop":
                break

    except requests.exceptions.ChunkedEncodingError:
        partial = "".join(assembled)
        spinner.stop()
        print(f"\n\033[93m[WARN] Stream interrupted mid-response. Partial reply shown above.\033[0m")
        return partial

    finally:
        spinner.stop()

    print()  # newline after streamed response
    return "".join(assembled)


# ── Commands ───────────────────────────────────────────────────────────────────

COMMANDS = {
    "/clear":   "Clear conversation history and start fresh",
    "/history": "Print all conversation turns so far",
    "/help":    "Show this help message",
    "/exit":    "Quit the chatbot",
}


def print_help():
    print("\n\033[36m── Commands ───────────────────────────────\033[0m")
    for cmd, desc in COMMANDS.items():
        print(f"  \033[33m{cmd:<12}\033[0m {desc}")
    print()


def print_history(history: list):
    if not history:
        print("\033[90m(No conversation history yet)\033[0m")
        return
    print("\n\033[36m── Conversation History ────────────────────\033[0m")
    for i, msg in enumerate(history, 1):
        role_color = "\033[32m" if msg["role"] == "assistant" else "\033[34m"
        print(f"{role_color}[{i}] {msg['role'].capitalize()}:\033[0m {msg['content'][:200]}")
        if len(msg["content"]) > 200:
            print("    \033[90m... (truncated)\033[0m")
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────

BANNER = """
\033[36m╔══════════════════════════════════════════╗
║          CLI Chatbot  ·  Claude API       ║
║   Type /help for commands  ·  /exit quit  ║
╚══════════════════════════════════════════╝\033[0m
"""


def main():
    print(BANNER)
    api_key = get_api_key()
    history: list = []

    while True:
        try:
            user_input = input("\033[34mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[90mGoodbye.\033[0m")
            sys.exit(0)

        # ── Empty input ────────────────────────────────────────────────────────
        if not user_input:
            print("\033[90m(Please type a message or /help)\033[0m")
            continue

        # ── Built-in commands ──────────────────────────────────────────────────
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("\033[90mGoodbye.\033[0m")
            sys.exit(0)

        if user_input.lower() in ("/clear", "/reset"):
            history.clear()
            print("\033[90m✓ Conversation cleared.\033[0m")
            continue

        if user_input.lower() == "/history":
            print_history(history)
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        # ── Send to API ────────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        try:
            reply = send_message(history, api_key)
        except RuntimeError as err:
            print(f"\033[91m[ERROR] {err}\033[0m")
            # Remove the user turn we just appended so the history stays clean
            history.pop()
            print("\033[90m(Your message was not sent — please try again)\033[0m")
            continue

        if reply:
            history.append({"role": "assistant", "content": reply})
        else:
            print("\033[93m[WARN] Received an empty reply from the API.\033[0m")
            history.pop()  # remove dangling user turn


if __name__ == "__main__":
    main()
