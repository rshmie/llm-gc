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

The provider is Anthropic's API unless the proxy is given an upstream. Behind an
organisation's gateway, the gateway is the upstream: requests go there with the
client's own headers, and LLM-GC's summary and fact calls go there too, so every
call reaches Anthropic the one way the organisation allows. An upstream that is the
proxy's own address is refused at startup, since every request would come back to it.

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
turn's role and its content, in canonical form. The hash at turn *k* therefore
identifies the whole conversation up to *k*, and two histories share it only if
they say the same thing that far. The canonical form leaves out what is not said:
key order, a plain string against one text block, delivery hints such as
`cache_control`, and the client's instructions inside a turn (below). Claude Code
marks its newest turns for caching and moves the markers on with every request; a
hash that counted them saw every request as a new conversation, so nothing was ever
collected. It also rewrites the instructions inside earlier turns — the first
message's grew from 263 tokens to 4,303 mid-conversation — and a hash that counted
them seeded the same conversation as a second session, which collected again and
read its facts again from nothing. Leaving them out merges nothing that was said
differently: the engine never reads them, and every forwarded request carries them
as the client sends them in that request.

```
client sends:  t0 t1 t2 t3 t4 │ t5 (new)        chain hashes: h0 h1 h2 h3 h4
session S has recorded up to t2   → S.tip == h2  → continue: record t3, t4 into S
```

- **With the `x-llm-gc-session` header**, the header names the session, and the
  proxy compares that session's recorded chain with the client's to find exactly
  where they diverge.
- **Without it**, the proxy looks for the session whose last recorded hash (its
  *tip*) matches the longest prefix of the client's history.

Two conversations can share a session only while their histories say the same
thing, so everything the session holds was derived from content both of them
contain. The moment one differs, its hashes stop matching and it continues in
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
- **Streaming is relayed chunk by chunk** as it arrives. The record never reads
  the stream: the reply is learnt from the next request. Only while the dashboard
  runs is a copy kept, one append per chunk, and read once the client has the
  whole reply, to show it (below).
- **The provider's answer and errors pass through verbatim**, status codes
  included: the client's error handling was written against the provider's format.
- **Recent turns go as the client sent them in this request**, not as recorded
  when they arrived. The client moves its cache markers onto its newest turns, and
  markers kept from earlier requests would pile up past the provider's limit of
  four.
- **Instructions inside the conversation are never condensed.** A client can put
  system messages among its turns: Claude Code adds its environment and its tools'
  instructions after a user turn. They are set aside, word for word, each with its
  place; the engine sees only the conversation, and every forwarded request puts
  each instruction back — after the turn it followed, or right after the front when
  that turn was collected. Condensed, Claude Code's own instructions could be
  summarised away; passed over, every Claude Code request went untouched.
- **So are instructions inside a message.** Claude Code also writes instructions
  into the user's messages and tool results, each wrapped in a `<system-reminder>`
  element: CLAUDE.md, memory and the user's context arrive in the first message
  this way. Each element is lifted out of its turn's text, so the engine never
  summarises, scores or quotes it, and kept on the turn. A recent turn goes as the
  client sent it, elements in place; once a turn is collected, its elements open
  the front, word for word, in order, each said once — Claude Code repeats some
  after every file it reads. Summarised with their message, the model read a
  paraphrase of the project's rules, or nothing of them.

  ```
  message 0 as sent:   <system-reminder>CLAUDE.md…</system-reminder>  Help me design…
  engine reads:        Help me design…
  after a collection:  front = [ <system-reminder>CLAUDE.md…</system-reminder> , recap ]
  ```

  *Rejected: never collecting the first message* — it protects one message's
  instructions and none of the later ones. *Rejected: keeping only standing
  instructions* — telling a standing rule from a passing note needs each client's
  meaning. *Rejected: sending them as system messages* — it changes how the client's
  words reach the model. *Accepted cost:* a long session's instructions all stay in
  every request, as they would without LLM-GC; another client's instructions inside
  a message, in another wrapper, are condensed like conversation.
- **Every other Anthropic endpoint passes through untouched.** A client such as
  Claude Code also counts tokens and lists models through the same base URL; those
  requests are forwarded with their method, path, query, headers and body, and
  nothing is recorded. A 404 from the proxy would fail the client because of LLM-GC.
  The query string is kept on Messages too.

### LLM-GC never makes a conversation worse

Using LLM-GC must never leave a user worse off than not using it. Only the
forwarding itself may fail a request:

| Where something fails | What the user gets | What is lost |
|---|---|---|
| Anything before forwarding (parsing, reconciling, collecting) | The original request, forwarded | GC for that one request |
| Background aging, or scheduling it | The full answer | That aging pass; logged |
| The history observer (the dashboard's transcript) | The full answer, with GC | Nothing; logged |
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
- **Read a turn as text, with its tool calls and results written out.** The engine
  sizes, scores, summarises, reads facts from and quotes a turn by its text alone.
  A tool call reads as `[tool call: name] {input}` and its output as
  `[tool result] ...` (`[tool result, error] ...` when the tool failed); an
  attachment reads as `[image]` or `[document]`; thinking has no text form. In an
  agent's conversation most turns are tool rounds, so reading only the words
  counted them as empty: the prompt looked smaller than it was, summaries were
  written without what the tools did, and a span kept word for word lost every
  tool output while being shown as complete. *Rejected: counting the blocks'
  tokens separately* — the sizes come right, but summaries and quotes still lose
  the output. *Rejected: a second text field for tool content* — every reader
  needs it, and a second field invites the next reader to read the wrong one.
  *Accepted cost:* a large tool output (a file read in full) makes the summary
  and facts calls for its span as large.

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
the same sessions; it is opt-in because it shows conversation content. To show
what was said beside what the model still has, the proxy hands each request's
conversation to an optional observer before forwarding, and each finished reply
to another once the client has it all; both are attached only when the dashboard
runs, and neither can fail a request. A reply is read for showing only, in the
same text form its turn will have once recorded; one that stopped before its end,
or longer than 4 MB, is not shown. See [visibility](../visibility/overview.md).

```
request N ──► forwarded ──► reply streams to the client ──► copy read: shown as "newest reply"
request N+1 carries the reply in its history ──► recorded; the shown copy is dropped
```

### Request size

A body larger than the provider's own request limit (32 MB) is refused with a 413
before it is buffered. The provider would reject it anyway, so refusing it costs no
user anything; the limit is enforced while reading, because `content-length` may be
absent or wrong.

## Known limitations (planned, not yet built)

- **A cut turn's attachments** reach the front as a placeholder, `[image]` or
  `[document]`, and are counted as one: the front is text, so a collected image
  is named, not shown, and its size is undercounted.
- **Anthropic Messages only.** The OpenAI-compatible endpoint is not built.
