# Security policy

## Reporting a vulnerability

Please report a vulnerability privately, not as a public issue. On GitHub, open this repository's **Security** tab and choose **Report a vulnerability**. Only the maintainers can read the report.

Please include:
- what the problem is, and what an attacker gains from it;
- the steps to reproduce it, and the output of `llm-gc --version`;
- whether it exposes conversation text or an API key.

Do not put a real conversation or a real API key in a report. A made-up conversation that shows the problem is enough.

The maintainer will acknowledge a report within a week, say whether they consider it a vulnerability, and tell you when a fix is released. LLM-GC has one maintainer, so these are best-effort times and not a guarantee.

## Supported versions

LLM-GC is before 1.0. Only the latest release gets security fixes.

## What LLM-GC keeps

[doc/data.md](doc/data.md) says which of your data LLM-GC holds, where, for how long, what leaves your machine, and how to delete it.

## What counts as a vulnerability

LLM-GC sits between a person's AI client and the model provider, so it sees every conversation and forwards the person's API key. These are the promises it makes, and a way to break one is a vulnerability:

- **The proxy and the dashboard are reachable from this machine only.** They listen on `127.0.0.1`, and the proxy has no login, because it is meant for one person on their own computer. A way for another machine, or a website open in the person's browser, to reach either one is in scope.
- **A client's API key is forwarded and never kept.** It is not stored, logged, put in an event or shown on the dashboard. LLM-GC's own key is held as a secret and is never printed. A way to make either appear is in scope.
- **Conversation text is never written to a log line or put in an event.** The dashboard shows it, because showing it is the dashboard's job, and the dashboard is off unless the person starts it. A way to make the text leave through a log line, an event, an error body or any other response that should not carry it is in scope.
- **Text inside a conversation is data, never instructions to LLM-GC.** Conversation text is quoted inside LLM-GC's own prompts, and text that closes the quote early or forges a speaker is a vulnerability. Text that merely persuades the summarising model inside an intact quote is a known limit of using a model to summarise, so please report it as a normal issue.

## What does not count

- Anything that needs an attacker who can already run code as the person, or read their `llm-gc.toml` or `.env` file.
- A flaw in the model provider's service, or in a gateway placed in front of it.
- Starting the proxy on an address other than `127.0.0.1`: LLM-GC does not support that, because it has no authentication to protect it.
