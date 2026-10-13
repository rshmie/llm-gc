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

**One client session, several conversations.** A conversation here is one message
list, and a client can run several under one session of its own. Claude Code sends its
main conversation, one per sub-agent, and, after `/compact` or an edited message, a new
main conversation that continues nothing recorded. Each is its own prompt, so each is its
own LLM-GC session, collected on its own. Claude Code names its session on every request
(`x-claude-code-session-id`) and a sub-agent's requests name the agent too
(`x-claude-code-agent-id`); the proxy reads both as a label, short plain ids only, and
never as identity.

```
Claude Code session 6d84277c
  ├─ main conversation          s_0676…   (after /compact: the one in use)
  ├─ sub-agent afc5b6eb         s_591e…
  ├─ sub-agent a4e43c0c         s_143a…
  └─ earlier main conversation  s_2cb5…   (replaced; closes once idle)
```

The proxy logs each conversation it starts, with the history it came with and the
client session and agent it names, and each answer names its conversation, so a log
line and a dashboard entry can be matched. *Rejected: keying the session by Claude
Code's session id* — the main conversation and its sub-agents interleave under one id,
so every request would read as an edit of the last and restart the session, and nothing
would ever be collected. *Rejected: recognising a compacted conversation by its opening
words* — tied to one client's current text; the headers are structure. *Accepted cost:*
the label comes from one client's header names; a client that sends none is shown
ungrouped, and a renamed header turns the grouping off rather than grouping wrongly.

**A subagent's conversation is recorded, not collected.** It passes through the proxy,
is counted and shown like any other, and is forwarded exactly as the client sent it, so
its prompt cache stays whole. It is never collected, unless `collect_subagents` is on.

| | Main conversation | Subagent |
|---|---|---|
| After a collection | many more requests, each carrying the saving | often none: it answers and stops ([not yet sent](#what-is-measured)) |
| What a collection costs | summary calls, and a cache rewrite from the recap on | the same, seldom paid back |
| If condensing loses a detail | the next reply shows it, to the person reading | the subagent's answer is quietly worse, and only the answer is read |

The client's agent header only turns collection *off*. Without it - another client, or a
renamed header - a conversation is collected like any other: the worst a wrong reading
can do is what LLM-GC already does. That is why the header may decide this when it may
not decide which conversation a request belongs to. The engine holds the choice per
session, so a collection that will not run is never forecast, and *Run GC now* says so
instead of collecting. *Rejected: collecting every conversation* — subagent collections
were measured landing after their last request, paid for and never sent. *Rejected: a
rule without the header, such as collecting only after a conversation has run N
requests* — it would still collect the long subagents, whose answers carry the risk.
*Revisit when* real runs show a subagent near its own window: then collecting above a
high share of it would be worth its risk.

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
- **A subagent's report is marked, not lifted.** When a background subagent
  finishes, Claude Code adds its report to the main conversation as a user turn
  that opens with an `<agent-message>` element, sometimes after a reminder in the
  same turn. A turn whose text, once its reminders are lifted, opens with that
  element is marked *relayed*. It stays conversation — scored, condensed when it
  goes cold, quoted as a `subagent report` — but nothing reads it as the person:
  it opens no exchange, is not what the user is discussing now, and its facts are
  the assistant's side. *Rejected: lifting it like a reminder* — instructions are
  never condensed, so a long report would ride in every request for the rest of the
  session. *Rejected: recognising it by matching the subagent's handback text* — in
  a live trace one report in two was wrapped so its text no longer matched.
  *Accepted cost:* the element is Claude Code's; another client's relayed report is
  read as the user's, as before, and a person who starts a message with
  `<agent-message` is read as a report.

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
| The upstream rate-limits or is overloaded (429, 529) | The provider's answer; later requests forwarded as before | LLM-GC's own model calls, and so collections, until the pause ends |

**LLM-GC yields to the upstream's limits.** Its summaries and fact reads sometimes can go to the
same upstream as the client's requests, often under the same credential, so they share
its rate limit. When a client request comes back 429 (rate limited) or 529 (overloaded),
or one of LLM-GC's own calls is rate-limited, LLM-GC makes no model call of its own for
as long as the upstream's `Retry-After` asks, or a minute when it gives none; a later
refusal extends the pause, up to ten minutes. No collection starts meanwhile, and each
check skipped is recorded as *paused*, with the reason. The client's requests are never
held back: they are forwarded exactly as before, with the recap already made. The
dashboard says so on every tab while it lasts.

```
client request ──► upstream ──► 429, retry-after: 30
                                   │
                                   ▼
                      LLM-GC pauses its own calls 30 s ──► checks skipped: "paused"
client requests keep going ──► forwarded as before, recap unchanged
```

*Rejected: retrying LLM-GC's own calls with backoff* — it keeps competing with the
client for the same limit, only more slowly. *Rejected: leaving it to the summary
calls' own errors* — a refused summary is recorded, but the next collection tries
again at once, still competing. *Accepted cost:* while paused, the conversation grows
uncondensed; a collection follows once the pause ends.

A malformed body is forwarded untouched, because the provider is the authority on
its own API. Guards catch `Exception`, never `BaseException`, so cancellation on
client disconnect still propagates and the upstream connection is closed.

### What is measured

Every request is counted where it belongs: its conversation, the client session that ran
it, and the proxy as a whole. Counts, ids and model names only, never conversation
content. Each figure happened — a request sent, a reply's reported usage, a collection's
own model calls. What the *next* request will carry is the engine's forecast, shown as
one, never added in.

```
The proxy                          every request, from every client
└─ Claude Code session 6d84277c    every request that named it
   ├─ Main                         requests · tokens sent and forwarded · the provider's usage
   ├─ Subagent 1                     · LLM-GC's own calls · collections, and those not yet sent
   ├─ Earlier main                   · the model, and its window
   └─ Other requests               first messages no conversation claimed; side requests
```

Each number comes from one place, at one moment in a request's life:

```
 client ──► request ──► count it whole ───────────────────────► requests,
            system,     system + tools + instructions + messages  tokens sent,
            tools,      as sent, and again as forwarded           tokens forwarded
            messages            │
                                ▼
                    forward ──► provider ──► reply ──► read its usage ──► the provider's usage
                                                       (the body, or a      (input, cache read,
                                                        stream's events      cache write, output)
                                                        as they pass)
 after the reply:
   collection ──► summary and fact calls, metered ──► gc_finished ──► LLM-GC's own calls,
                                                                       collections, not yet sent
```

- **The whole request counts, on both sides.** Tokens sent are the system prompt, the
  tool definitions, the client's instructions and the messages; tokens forwarded are the
  same request with the recap. GC never changes the system prompt or the tools, but left
  out, the share saved would be measured against only the part GC can shrink, and read
  larger than what the model was spared:

  ```
                      system + tools   conversation        share saved
  sent                    20,000          40,000
  forwarded               20,000          10,000
  counted whole           60,000  →  30,000                  50%   ← what the model was spared
  conversation only       40,000  →  10,000                  75%   ← overstated
  ```

  Both sides are counted with LLM-GC's own tokenizer, so the ratio holds while the
  absolute numbers differ from the provider's. The system prompt and tools repeat on
  every request, so their count is cached by text.
- **The provider's own count rides beside it.** Each reply's `usage` — uncached input,
  cache read, cache write, output — is added wherever its request was counted: from the
  body of an unstreamed reply, or read from a stream's `message_start` and
  `message_delta` events as they pass, holding only an unfinished line between chunks. A
  stream cut short still reports the prompt, which was billed. The provider never sees
  the uncondensed prompt, so it cannot count a saving: its numbers say what was billed,
  LLM-GC's compare before and after.
- **LLM-GC's own model calls are counted per collection.** The summary provider is
  wrapped in a meter ([the provider layer](../llm-provider.md)), so every call a
  collection makes — the compactor's and the fact reader's, concurrent or not — is
  counted against that collection and no other. The result carries the calls and
  tokens, and the proxy adds them to the conversation. A collection that bought summaries
  and then abandoned its cut still reports what it spent.
- **Bounded.** 1,024 conversations, 256 client sessions and 256 unclaimed first requests
  are kept; above that the least recently active is dropped, and its requests stay
  counted in the totals above it.

**A conversation's first request is claimed on its second.** With no history, a first
request belongs to no conversation yet. It is held under its opening turn's hash and the
client session and agent it named, and the conversation that opens the same way claims
it — the same object, so a reply to it still on its way lands there too. Two subagents
given the same task are told apart by their agent. A request nothing claims, such as a
side request, stays in its client session as *other requests*.

```
request 1     [m0]            no history: no conversation yet
                              held as  opens with #m0 · session cc-1 · main
request 2     [m0, a0, m1]    its history starts conversation s_0676
                              s_0676   opens with #m0 · session cc-1 · main  → claims request 1
subagent 1    [task]          held as  opens with #task · session cc-1 · agent afc5
subagent 2    [task]          held as  opens with #task · session cc-1 · agent a4e4  ← same words,
                                                                                    another agent
side request  [x]             held as  opens with #x · session cc-1 · main
                              never claimed → "other requests" of cc-1
```

**A collection not yet sent is said to be.** A collection is made after a reply, for
the conversation's *next* request, so a request must come after it before it saves
anything. Until one does, the conversation counts it as *not yet sent*. A finished
subagent, or a main conversation `/compact` replaced, keeps it for good: that collection
cost its model calls and saved nothing. Whether such collections are frequent enough to
change when a collection runs is measured here before it is decided.

```
main conversation   … req 9 ─► reply ─► collection ─► req 10 carries the recap
                                         not yet sent 1    not yet sent 0, saved grows

subagent            … req 5 ─► reply ─► collection ─► (the subagent has answered:
                                         not yet sent 1    no request 6) saved stays 0
```

**Each request is also recorded, on its own.** The ledger keeps sums; the trace needs
each request as a row. A record opens just before forwarding and closes when the reply
ends: its conversation, its client session and agent, when it went and came back, the
tokens it carried, what kind its newest message is — typed by the person, tool results,
a relayed report, or the assistant's — and the reply's outline: status, stop reason,
the provider's usage, and the tools it called, by name and call id. Structure only,
never content: the outline is read from the stream as it passes, and a reply cut short,
refused or never answered still closes its record. 10,000 records are kept; the oldest
is dropped first, and its counts stay in the ledger.

A record names its newest message by the conversation's chain hash through it, so a
later request of the same conversation settles it: *kept* when its history says the
same through that message, *replaced* when it says something else there. Two requests
at the same position are not compared with each other: until one is carried on,
neither is known to be kept.

```
pos 1..5  [ask][Agent ×2][results][running]
req A     ……………………………………………………………… + [suggest a next message]   side request      kept? waits
req B     ……………………………………………………………… + [agent report]            relayed report    kept? waits
req C     ……………………………………………………………… + [agent report][reply][thanks]
                                     history through position 6 = req B's → B kept, A replaced
```

A first request is settled the same way: kept, and claimed by the conversation, when
that conversation's history carries it. A request no conversation ever carries on —
a title, a quota check — stays unsettled; the trace shows it as a side request.
*Accepted cost:* only the newest 64 unsettled requests per client session and agent are
watched, so an old one stays unsettled rather than being compared for ever.

**A subagent's first request names the call that started it.** A tool call that starts a
subagent gives it a prompt, and the subagent's first message holds that prompt inside the
context the client adds around it. So each reply's tool calls that carry a `prompt` input
are offered to a linker, and so is each subagent's first message; a pair is made when the
prompt is found in the message, and the link is stored on the subagent's first record.

```
main reply      → Agent toolu_a {prompt: "Count the test files…"}      ┐ offered when the reply ends
                → Agent toolu_b {prompt: "Summarise README.md…"}       ┘
subagent a1     first message: [context] "Count the test files…" [context]  → spawned by toolu_a
subagent a2     first message: [context] "Summarise README.md…" [context]   → spawned by toolu_b
                (either side may reach the proxy first; each waits for the other)
```

- **Read by the input's field, not the tool's name.** Claude Code has named the tool
  `Task` and `Agent`; both take `prompt`. *Rejected: matching every tool input* — in a
  live trace a subagent's own file path was found inside its opening, a false link.
- **Either side may come first.** The reply ends and the subagent starts within
  milliseconds, in no fixed order, so a call and an opening each wait for the other, for
  up to ten minutes. When several could pair, the longest prompt wins - a short one can
  sit inside a longer one's subagent by chance - and of equal prompts the earliest call,
  so two subagents given the same task start in order.
- **The prompt is content, held only for this.** It is read from the reply as it
  streams, kept apart from the outline, held in memory only until matched, never
  logged, and only while the dashboard runs. An input larger than any prompt is let go
  while it streams. A subagent no call matches is shown as started by a call not
  determined, never guessed.
- **Not yet linked: a report to its subagent.** A relayed report is shown as a
  subagent report, in its place in time; which subagent sent it is not named, because
  a live trace matched only one report in two by its words.

*Rejected: keeping the counts in each session's state* — sessions close when idle, and
their requests would vanish from the client session's total. *Rejected: asking the
provider to count each request (`count_tokens`)* — a network call per request for a
number the reply's `usage` already gives, and the uncondensed prompt is never sent, so
it could not count the saving either. *Rejected: one process-wide counter for LLM-GC's
own calls* — two conversations collected at once would each be credited with the
other's. *Accepted cost:* grouping by client session relies on Claude Code's headers; a
client that names no session has each conversation listed on its own.

`/stats` returns the proxy's totals — requests, GC applied and bypassed,
`tokens_from_client`, `tokens_forwarded`, `tokens_saved` and their ratio, the provider's
reported usage, LLM-GC's own calls, and the reconciliations `history_rewinds`,
`history_restarts` and `sessions_seeded` (a restart or a seed costs one uncompressed
request, so each is counted rather than hidden) — then `client_sessions`, each with its
conversations: their part (main, subagent, earlier main), model, window and counts; and
`other_conversations`, for those no client session named.

`llm-gc stats` reads these from a running proxy, on the port the settings name, and
validates the answer before using it, since whatever is on that port may not be an
LLM-GC proxy. It prints the totals, what the provider reported and what LLM-GC's own calls
spent, then one table per client session: each conversation's requests, tokens sent and
saved, collections (and those not yet sent) and window. "Applied" counts requests that
went through LLM-GC's path, which includes the early ones where nothing has been
condensed yet; "bypassed" counts requests forwarded untouched because they could not be
read or the engine failed on the request path. With no requests it says so instead of
printing zeros, because a ratio with nothing behind it is not a measurement. Rewinds and
restarts appear only once they have happened: only a client sending the session header
can cause them, so for Claude Code they would be a permanent row of zeros. `--json`
always has every counter.

### Each conversation's window

Each conversation names its model on every request, and one session runs several — a
main conversation on one model, its subagents on another. The window of each model is
asked of the upstream's Models API (`GET /v1/models/{model}`, its `max_input_tokens`)
with LLM-GC's own Anthropic credential: once per model, in the background, never on the
request path. Until the answer arrives, or when there is none — a gateway that lists no
models, a model it names its own way, no Anthropic credential — the configured
`context_window` is used, and the dashboard and `llm-gc stats` say it is assumed. The
conversation's recap budget is its share of that window
([detail levels](../generational-memory/detail-levels.md)), read at its next collection.
At most 64 model names are asked about, so a client cannot make LLM-GC call the upstream
without end.

```
request names model M
        │
        ▼
 window of M known? ──── yes: the upstream reported it ─────► M's window           "1M"
        │
        ├─────────────── yes: asked, and none came back ─────► configured window   "200k assumed"
        │
        └─ no ─► configured window for now ──────────────────► configured window   "200k assumed"
                 and, once, in the background:
                 GET /v1/models/M ─► max_input_tokens ─► remembered for M

the conversation's recap budget = its share (30%) of that window, read at its next collection
```

A larger window cannot push a request past the model's: the recap is the conversation's
own older turns, condensed or quoted, so even with every span restored it is about the
size of the turns it stands for, which the client would send anyway. A larger budget
only means a long conversation's recap is squeezed later. *Rejected: a table of windows
in code* — stale with each model release, and blind to a gateway's names. *Rejected:
asking on the request path* — the first request of every new model would wait on it.
*Accepted cost:* a model whose larger window needs a header the client did not send is
sized for the larger window; the bound above still holds.

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
