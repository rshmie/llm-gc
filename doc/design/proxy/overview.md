# The Proxy

## How it works

An application points its SDK at the proxy instead of the provider and changes
nothing else. The proxy is LLM-GC's application face: it holds no GC logic of its
own. It finds which conversation a request belongs to, keeps that conversation's
record true, asks the engine for the collected context, and forwards.

```
client ──► POST /v1/messages (the whole conversation, every time)
             │
             ├─ 1. split:      history = all but the last turn      new turn = the last
             ├─ 2. reconcile:  find the session this history continues; record what it
             │                 has not seen; rewind or start over if the history diverged
             ├─ 3. collect:    engine returns the session's context
             │                 (identical to the history? forward the client's own bytes)
             ├─ 4. forward ──────────────────────────────────────────────► provider
             │   ◄── response streamed or returned to the client unchanged ◄─┘
             └─ 5. age:        in the background, from the recorded conversation only
```

### The client's history is the truth

A chat API is stateless: every request carries the entire conversation. That
history is the most reliable record that exists of what the user saw — every
earlier reply exactly as delivered (text, tool calls, thinking blocks), and
nothing the user never saw. So the proxy never builds its record from responses.
It **reconciles its record to the client's history on every request**: the reply
to request N is recorded when request N+1 shows it.

Several cases fall out of that one rule without code of their own:

| What the client did | What the history shows | What the proxy does |
|---|---|---|
| Continued the conversation | Our record plus the last exchange | Records the exchange; collects; forwards |
| Retried a failed request | Exactly what the failed attempt sent | Nothing new to record; forwards the same thing again |
| Pressed regenerate | Our record, minus nothing; the last reply never shown again | Records nothing; the replaced reply is never forwarded |
| Edited an earlier turn | Our record up to the edit, then something else | Rewinds the session to the edit (header path), or starts a fresh session |
| Reached a restarted proxy | A conversation we hold nothing for | Seeds a new session from the history |
| A stream died mid-reply | The reply is absent (the client never had it) | Never recorded |

### Finding the conversation

Every turn gets a **chain hash**: a hash of the previous turn's hash plus this
turn's role and its content exactly as sent, in canonical JSON. The hash at turn
*k* therefore identifies the whole conversation up to *k*, and two histories share
it only if they are byte-identical that far.

```
client sends:  t0 t1 t2 t3 t4 │ t5 (new)        chain hashes: h0 h1 h2 h3 h4
session S has recorded up to t2   → S.tip == h2  → continue: record t3, t4 into S
```

- **With the `x-llm-gc-session` header**, the header names the session, and the
  proxy compares that session's recorded chain with the client's to find exactly
  where they diverge.
- **Without it**, the proxy looks for the session whose last recorded hash (its
  *tip*) matches the longest prefix of the client's history.

Two conversations can share a session only while their histories are
byte-identical, so everything the session holds was derived from content both of
them contain. The moment one differs, its hashes stop matching and it continues in
its own session.

**Rewinding.** When a header session's history diverges at turn *k*, the session
is rewound to *k* if every turn from *k* on is still verbatim; otherwise it starts
over, because a summary or a fact derived from the abandoned branch cannot be
separated from what came before it. Starting over costs one uncompressed request;
keeping the abandoned branch could put it in the model's memory. Either way the
session gets a new state object, so an aging pass still running on the old one
discards its result instead of committing changes planned against a conversation
that no longer exists.

### Forwarding

- **Untouched when nothing has aged.** If the collected context is exactly the
  client's history, the client's own request bytes are forwarded. Images, tool
  calls, thinking blocks, and the provider's prompt cache all survive intact.
- **Streaming is relayed chunk by chunk** as it arrives. Nothing in the stream is
  parsed: the reply is learnt from the next request, so a stream needs no
  observer.
- **The provider's answer and errors pass through verbatim**, status codes
  included: the client's error handling was written against the provider's format.

### LLM-GC never makes a conversation worse

Using LLM-GC must never leave a user worse off than not using it. Only the
forwarding itself may fail a request:

| Where something fails | What the user gets | What is lost |
|---|---|---|
| Anything before forwarding (parsing, reconciling, collecting) | The original request, forwarded | GC for that one request |
| Background aging, or scheduling it | The full answer | That aging pass; logged |
| The provider | The provider's own error | Nothing of ours |

A malformed body is forwarded untouched, because the provider is the authority on
its own API. Guards catch `Exception`, never `BaseException`, so cancellation on
client disconnect still propagates and the upstream connection is closed.

### What `/stats` reports

Aggregate counters only, never conversation content: requests, GC applied and
bypassed, tokens in client histories and tokens forwarded, and the
reconciliations — `history_rewinds`, `history_restarts`, `sessions_seeded`. A
restart or a seed costs one uncompressed request, so each is counted rather than
hidden.

## Why it exists

The engine is a library; the proxy is how a real application uses it without
changing its code. It is also the boundary where LLM-GC meets a conversation it
does not control — a client that retries, regenerates, edits, streams, attaches
images, or calls tools — so it is where the engine's assumptions about a
conversation are tested against what clients actually do.

## Why this shape

- **Record from the client's history, not from responses.** Recording replies
  parsed out of responses failed in every case clients actually produce: a
  regenerate forwarded the answer being replaced, a retry duplicated turns, a
  tool-call reply had no text and was lost, a streamed reply needed a parser, and
  the record could drift from what the user saw. *Accepted cost:* a one-exchange
  lag — the newest exchange reaches the record, and aging, one request later.
  Measured on a 12-request demo the forwarded share rose from 0.917 to 0.965 of the
  client's history; over a long conversation the lag is the last two turns of
  many, which aging would keep verbatim anyway. Removing it would mean recording a
  turn before knowing the request succeeded, which breaks retries.
- **Identify a conversation by chain hashes, not by fingerprinting its opening.**
  Hashing the first two turns gave a conversation's first and second requests
  different identities, so the second request of every headerless conversation
  reached the model without its history. A chain hash identifies a conversation by
  everything up to a point, works from the second request, and cannot merge two
  conversations that differ. *Rejected: requiring the session header* — correct,
  but almost no client sends it.
- **Canonical JSON when hashing.** An SDK that serialises the same content with its
  keys in another order must not look like an edited conversation, which would cost
  a needless restart.
- **Forward the client's bytes whenever nothing changed.** Rebuilding every turn as
  plain text dropped attached images and tool calls, and any rewrite resets the
  provider's prompt cache.

### After a cut

Once the engine has cut a session, the forwarded conversation is the front (one
user-role message) followed by every later turn as the client's **original
content** — images, tool calls and all. Thinking blocks are removed from every
turn sent with the first request that carries a new front, because each was
produced against a prompt the cut replaced; the proxy tells the engine when it
sends a front for the first time, and that request's history length becomes the
boundary. Replies produced from then on keep theirs. Between cuts the forwarded
conversation only grows at the end, so the provider's prompt cache holds. See
[the cut](../generational-memory/the-cut.md).

### Idle sessions

A session holds its conversation's content, so the proxy runs the session idle
sweeper for as long as it serves: a session untouched for 30 minutes is closed
and its data dropped. A client that returns later is seeded afresh from its own
history, at the cost of one uncompressed request.

### The dashboard

The live dashboard runs in the same process on its own localhost port, reading
the same sessions; it is opt-in because it shows conversation content. See
[visibility](../visibility/overview.md).

### Request size

A body larger than the provider's own request limit (32 MB) is refused with a 413
before it is buffered. The provider would reject it anyway, so refusing it costs no
user anything; the limit is enforced while reading, because `content-length` may be
absent or wrong.

## Known limitations (planned, not yet built)

- **A cut turn's non-text content** survives in the front only as text.
- **Anthropic Messages only.** The OpenAI-compatible endpoint is not built.
