#!/usr/bin/env python3
"""Talk to a model through the LLM-GC proxy, and watch the dashboard while you do.

    python scripts/run_proxy.py --serve --dashboard       # terminal 1
    python scripts/chat.py                                # terminal 2
    open http://127.0.0.1:9901                            # the dashboard

It is the official Anthropic SDK pointed at the proxy - exactly what an
application does - so everything the proxy does to a real client's traffic
happens here too: the full history is sent on every request, replies come back
streamed, and thinking blocks are returned with the history unchanged.

Commands: /new starts a fresh conversation; /quit (or Ctrl-D) leaves.

Credentials: ANTHROPIC_API_KEY, or any credential the SDK resolves on its own.
None is needed when the proxy runs with --stub-upstream.

Developer tooling, not library code, which is why it lives in `scripts/`.
"""

import argparse
import sys

import anthropic

PROXY_URL = "http://127.0.0.1:9900"
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 64_000       # streamed, so a large ceiling costs nothing unless used
SESSION_HEADER = "x-llm-gc-session"

# Re-runs a declined request on a fallback model inside the same call, routed by
# the refusal's category. Off with --no-fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Sent when no credential was found, so a stub upstream - which checks none - still
# answers. A real provider rejects it with a 401, reported with what to do.
PLACEHOLDER_KEY = "no-credential-found"

NO_CREDENTIAL_HINT = (
    "no Anthropic credential was found, so a placeholder key was sent.\n"
    "  for a real model:  export ANTHROPIC_API_KEY=... and start chat.py again\n"
    "  to run offline:    restart the proxy with --stub-upstream"
)

_DIM, _RESET = "\033[2m", "\033[0m"


def build_client(proxy_url: str) -> tuple[anthropic.Anthropic, bool]:
    """An SDK client aimed at the proxy, and whether it found a credential of its own.

    The SDK resolves its credential when it is built - an API key, an auth token,
    or a saved profile - but raises only when a request is sent, and then a
    `TypeError` rather than an API error. So what it resolved is checked here,
    before any request.
    """
    client = anthropic.Anthropic(base_url=proxy_url)
    if client.api_key or client.auth_token or client.credentials:
        return client, True
    return anthropic.Anthropic(base_url=proxy_url, api_key=PLACEHOLDER_KEY), False


def send(client: anthropic.Anthropic, args: argparse.Namespace, history: list[dict]):
    """Stream one reply to the terminal and return the final message."""
    request: dict = {"model": args.model, "max_tokens": args.max_tokens, "messages": history}
    if args.system:
        request["system"] = args.system
    if args.effort:
        request["output_config"] = {"effort": args.effort}
    if args.show_thinking:
        request["thinking"] = {"type": "adaptive", "display": "summarized"}
    if args.session:
        request["extra_headers"] = {SESSION_HEADER: args.session}
    if not args.no_fallback:
        request["betas"] = [FALLBACK_BETA]
        request["extra_body"] = {"fallbacks": "default"}

    # The request is sent on entering the block; announce the reply only once it is under way.
    with client.beta.messages.stream(**request) as stream:
        print("\nassistant> ", end="", flush=True)
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
            elif event.delta.type == "thinking_delta" and args.show_thinking:
                print(f"{_DIM}{event.delta.thinking}{_RESET}", end="", flush=True)
        final = stream.get_final_message()
    print()

    if final.stop_reason == "refusal":
        print(f"{_DIM}(the model declined this request){_RESET}")
    elif final.stop_reason == "max_tokens":
        print(f"{_DIM}(cut off at --max-tokens){_RESET}")
    return final


def chat(args: argparse.Namespace) -> None:
    client, has_credential = build_client(args.proxy)
    history: list[dict] = []
    print(f"talking to {args.model} through {args.proxy}   (/new, /quit)")
    if not has_credential:
        print(f"{_DIM}no Anthropic credential found - fine if the proxy runs with --stub-upstream{_RESET}")

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/new":
            history.clear()
            print(f"{_DIM}(new conversation){_RESET}")
            continue

        history.append({"role": "user", "content": line})
        try:
            reply = send(client, args, history)
        except anthropic.APIConnectionError:
            print(f"\ncannot reach the proxy at {args.proxy} - is `run_proxy.py --serve` running?")
        except anthropic.AuthenticationError as err:
            print(f"\nthe provider answered {err.status_code}: {err.message}")
            if not has_credential:
                print(NO_CREDENTIAL_HINT)
        except anthropic.APIStatusError as err:
            print(f"\nthe provider answered {err.status_code}: {err.message}")
        else:
            # The whole content, thinking blocks included and unchanged: the next
            # request must carry the reply exactly as it was produced.
            history.append({"role": "assistant", "content": reply.content})
            continue
        # Nothing was answered, so the turn is not part of the conversation; the
        # proxy records nothing for a failed request either.
        history.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", default=PROXY_URL, help=f"the proxy's base URL (default {PROXY_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="the model's effort level; its own default when omitted")
    parser.add_argument("--system", help="a system prompt for the conversation")
    parser.add_argument("--show-thinking", action="store_true", help="ask for and print a summary of the reasoning")
    parser.add_argument("--session", help=f"send {SESSION_HEADER} instead of letting the proxy match by content")
    parser.add_argument("--no-fallback", action="store_true", help="do not re-run declined requests on another model")
    try:
        chat(parser.parse_args())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
