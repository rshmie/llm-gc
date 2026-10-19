# The Proxy

## How it works

An application points its SDK at the proxy instead of the provider and changes nothing else. The proxy is LLM-GC's application face: it holds no GC logic of its own. It finds which conversation a request belongs to, keeps that conversation's record true, asks the engine for the collected context, and forwards.

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

The provider is Anthropic's API unless the proxy is given an upstream. Behind an organisation's gateway, the gateway is the upstream: requests go there with the client's own headers, and LLM-GC's summary and fact calls go there too, so every call reaches Anthropic the one way the organisation allows. An upstream that is the proxy's own address is refused at startup, since every request would come back to it.

### The client's history is the truth

A chat API is stateless: every request carries the entire conversation. That history is the most reliable record that exists of what the user saw — every earlier reply exactly as delivered (text, tool calls, thinking blocks), and nothing the user never saw. So the proxy never builds its record from responses. It **reconciles its record to the client's history on every request**: the reply to request N is recorded when request N+1 shows it.

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

Every turn gets a **chain hash**: a hash of the previous turn's hash plus this turn's role and its content, in canonical form. The hash at turn *k* therefore identifies the whole conversation up to *k*, and two histories share it only if they say the same thing that far. The canonical form leaves out what is not said: key order, a plain string against one text block, delivery hints such as `cache_control`, the client's instructions inside a turn (below), and the model's thinking. Claude Code marks its newest turns for caching and moves the markers on with every request; a hash that counted them saw every request as a new conversation, so nothing was ever collected. It also rewrites the instructions inside earlier turns — the first message's grew from 263 tokens to 4,303 mid-conversation — and a hash that counted them seeded the same conversation as a second session, which collected again and read its facts again from nothing. And when the conversation moves to another model, Claude Code sends the earlier replies without their thinking, which the new model could not read; a hash that counted thinking started the conversation over, losing a collection already applied. Leaving them out merges nothing that was said differently: the engine never reads them, and every forwarded request carries them as the client sends them in that request.

```
client sends:  t0 t1 t2 t3 t4 │ t5 (new)        chain hashes: h0 h1 h2 h3 h4
session S has recorded up to t2   → S.tip == h2  → continue: record t3, t4 into S
```

- **With the `x-llm-gc-session` header**, the header names the session, and the proxy compares that session's recorded chain with the client's to find exactly where they diverge. The name must be a short plain id, 1 to 64 letters, digits, `_` or `-`, the same rule as the client's own ids, because it becomes a log field, a dashboard label and part of the dashboard's addresses. Any other value is ignored with a warning, and the conversation is found as if no header was sent.
- **Without it**, the proxy looks for the session whose last recorded hash (its *tip*) matches the longest prefix of the client's history.

Two conversations can share a session only while their histories say the same thing, so everything the session holds was derived from content both of them contain. The moment one differs, its hashes stop matching and it continues in its own session.

**Rewinding.** When a header session's history diverges at turn *k*, the session is rewound to *k* if every turn from *k* on is still verbatim; otherwise it starts over, because a summary or a fact derived from the abandoned branch cannot be separated from what came before it. Starting over costs one uncompressed request; keeping the abandoned branch could put it in the model's memory. Either way the session gets a new state object, so an aging pass still running on the old one discards its result instead of committing changes planned against a conversation that no longer exists.

**One client session, several conversations.** A conversation here is one message list, and a client can run several under one session of its own. Claude Code sends its main conversation, one per sub-agent, and, after `/compact` or an edited message, a new main conversation that continues nothing recorded. Each is its own prompt, so each is its own LLM-GC session, collected on its own. Claude Code names its session on every request (`x-claude-code-session-id`) and a sub-agent's requests name the agent too (`x-claude-code-agent-id`); the proxy reads both as a label, short plain ids only, and never as identity.

```
Claude Code session 6d84277c
  ├─ main conversation          s_0676…   (after /compact: the one in use)
  ├─ sub-agent afc5b6eb         s_591e…
  ├─ sub-agent a4e43c0c         s_143a…
  └─ earlier main conversation  s_2cb5…   (replaced; closes once idle)
```

The proxy logs each conversation it starts, with the history it came with and the client session and agent it names, and each answer names its conversation, so a log line and a dashboard entry can be matched. *Rejected: keying the session by Claude Code's session id* — the main conversation and its sub-agents interleave under one id, so every request would read as an edit of the last and restart the session, and nothing would ever be collected. *Rejected: recognising a compacted conversation by its opening words* — tied to one client's current text; the headers are structure. *Accepted cost:* the label comes from one client's header names; a client that sends none is shown ungrouped, and a renamed header turns the grouping off rather than grouping wrongly.

**A subagent's conversation is recorded, not collected.** It passes through the proxy, is counted and shown like any other, and is forwarded exactly as the client sent it, so its prompt cache stays whole. It is never collected, unless `collect_subagents` is on.

| | Main conversation | Subagent |
|---|---|---|
| After a collection | many more requests, each carrying the saving | often none: it answers and stops ([not yet sent](#what-is-measured)) |
| What a collection costs | summary calls, and a cache rewrite from the recap on | the same, seldom paid back |
| If condensing loses a detail | the next reply shows it, to the person reading | the subagent's answer is quietly worse, and only the answer is read |

The client's agent header only turns collection *off*. Without it - another client, or a renamed header - a conversation is collected like any other: the worst a wrong reading can do is what LLM-GC already does. That is why the header may decide this when it may not decide which conversation a request belongs to. The engine holds the choice per session, so a collection that will not run is never forecast, and *Run GC now* says so instead of collecting. *Rejected: collecting every conversation* — subagent collections were measured landing after their last request, paid for and never sent. *Rejected: a rule without the header, such as collecting only after a conversation has run N requests* — it would still collect the long subagents, whose answers carry the risk. *Revisit when* real runs show a subagent near its own window: then collecting above a high share of it would be worth its risk.

### Forwarding

- **Untouched when nothing has aged.** If the collected context is exactly the client's history, the client's own request bytes are forwarded. Images, tool calls, thinking blocks, and the provider's prompt cache all survive intact.
- **Streaming is relayed chunk by chunk** as it arrives. The record never reads the stream: the reply is learnt from the next request. Only while the dashboard runs is a copy kept, one append per chunk, and read once the client has the whole reply, to show it (below).
- **The provider's answer and errors pass through verbatim**, status codes included: the client's error handling was written against the provider's format.
- **Recent turns go as the client sent them in this request**, not as recorded when they arrived. The client moves its cache markers onto its newest turns, and markers kept from earlier requests would pile up past the provider's limit of four.
- **Instructions inside the conversation are never condensed.** A client can put system messages among its turns: Claude Code adds its environment and its tools' instructions after a user turn. They are set aside, word for word, each with its place; the engine sees only the conversation, and every forwarded request puts each instruction back — after the turn it followed, or right after the front when that turn was collected. Condensed, Claude Code's own instructions could be summarised away; passed over, every Claude Code request went untouched.
- **So are instructions inside a message.** Claude Code also writes instructions into the user's messages and tool results, each wrapped in a `<system-reminder>` element: CLAUDE.md, memory and the user's context arrive in the first message this way. Each element is lifted out of its turn's text, so the engine never summarises, scores or quotes it, and kept on the turn. A recent turn goes as the client sent it, elements in place; once a turn is collected, its elements open the front, word for word, in order, each said once — Claude Code repeats some after every file it reads. Summarised with their message, the model read a paraphrase of the project's rules, or nothing of them.
- **A subagent's report is marked, not lifted.** When a background subagent finishes, Claude Code adds its report to the main conversation as a user turn that opens with an `<agent-message>` element, sometimes after a reminder in the same turn, and sometimes after the one line Claude Code writes before it (`Another Claude session sent a message:`). A turn whose text, once its reminders are lifted, opens with that element, alone or after that exact line, is marked *relayed*. It stays conversation — scored, condensed when it goes cold, quoted as a `subagent report` — but nothing reads it as the person: it opens no exchange, is not what the user is discussing now, and its facts are the assistant's side. *Rejected: lifting it like a reminder* — instructions are never condensed, so a long report would ride in every request for the rest of the session. *Rejected: recognising it by matching the subagent's handback text* — in a live trace one report in two was wrapped so its text no longer matched. *Rejected: accepting any opening line before the element* — a person's own "look at this:" before a pasted report would be filed as the subagent's, and their facts would leave "You said" with nothing to show it; a line matched exactly fails the other way, as a turn nobody typed on the trace, which can be seen. *Accepted cost:* the element and the line are Claude Code's; another client's relayed report, or one after a reworded line, is read as the user's, and a person who starts a message with `<agent-message` is read as a report.
- **Text Claude Code writes into a message is marked as its own.** Claude Code merges some of its own text into the person's next message: the record of a command they ran (`<command-name>`, `<command-message>`, `<command-args>`), the text a skill or prompt command expands to, a local command's caveat (`<local-command-caveat>`) and output (`<local-command-stdout>`), and the notice that a background task finished (`<task-notification>`). One turn can hold several of these and the person's own words, so each part Claude Code wrote is recorded on the turn as a range of its text (`client_written`). The command stays the person's, because they typed it. The text right after a command is its expansion, unless a caveat came before the command (it ran on the person's machine and expands to nothing) or the person's own words did (Claude Code writes a command first, so one after the person's words was pasted). As with reminders, only an element that opens at the start of a line and closes at the end of one counts, and tool output is never examined: reading a transcript file returns these elements as output. The parts stay conversation, sent, scored and condensed as before; only who is credited changes. A turn Claude Code wrote all of is the client's: it opens no exchange, its facts are the assistant's side, and the trace shows its request as a *Claude Code notice*. A turn it wrote part of is still the person's, but only their words are read as theirs (`generational-memory/overview.md`).

  ```
  message as sent:  <local-command-caveat>…   <command-name>/model…   <local-command-stdout>…   Now run the tests.
  written by:       Claude Code               the person              Claude Code               the person
  ```

  *Rejected: marking the whole turn, as a relayed report is marked* — a typed message and the output before it share one turn, so a whole-turn mark either files a skill's rules as the person's decision or takes the person's decision away from them. *Rejected: lifting the parts like reminders* — lifted text is sent whole in every request and never condensed, so a 2,500-token `/context` output would stay for the rest of the session. *Rejected: splitting the turn into one message per part* — turn numbers would stop matching the client's messages, which the conversation's identity, verbatim forwarding and cut points rely on. *Accepted cost:* the elements are Claude Code's and undocumented; when they change, the text is credited to the person, as without this, and is never lost. Text Claude Code writes without an element, such as the summary a conversation opens with after `/compact` or a note about an attached image, is read as the person's, and so are bash mode's elements, which no recorded session has used.
- **Words typed after an interrupt are the person's, beside the tool output.** A person can speak during a tool round only by interrupting it. Claude Code then sends the round's results, its mark `[Request interrupted by user]` as a text block of its own, and what they type, all in one user turn. Such a turn's text is built block by block, so each block's place in it is known. The tool output is recorded as ranges of its own (`tool_output`). The mark, and any other text beside tool results, is recorded as Claude Code's. The text after the mark is the person's, less what the element rules above find in it; those rules read the blocks after the mark together, so a command typed there keeps its expansion apart. A turn of tool results whose other text is all Claude Code's — a skill's instructions after the call that loaded it, `Tool loaded.`, an image's size — holds no words of the person's. The person's words count as theirs for facts and for what they are discussing now, but the turn opens no exchange (`generational-memory/overview.md`).

  ```
  blocks as sent:  tool_result 42 passed   [Request interrupted by user]   Stop, use MySQL.
  written by:      a tool                  Claude Code                      the person
  ```

  *Rejected: crediting every text block beside tool results to the person* — in one machine's Claude Code transcripts, 8 of about 60 such turns held a person's words, each after the mark; the rest were Claude Code's own text, which this would have filed as the person's. *Rejected: recognising each of Claude Code's notices by its words* — a list that grows with every release, and an unknown notice would be credited to the person. *Accepted cost:* the mark is Claude Code's and undocumented; if it changes, the person's words beside tool results are read as Claude Code's, as before this, and are never lost.

  ```
  message 0 as sent:   <system-reminder>CLAUDE.md…</system-reminder>  Help me design…
  engine reads:        Help me design…
  after a collection:  front = [ <system-reminder>CLAUDE.md…</system-reminder> , recap ]
  ```

  *Rejected: never collecting the first message* — it protects one message's instructions and none of the later ones. *Rejected: keeping only standing instructions* — telling a standing rule from a passing note needs each client's meaning. *Rejected: sending them as system messages* — it changes how the client's words reach the model. *Accepted cost:* a long session's instructions all stay in every request, as they would without LLM-GC; another client's instructions inside a message, in another wrapper, are condensed like conversation.
- **Every other Anthropic endpoint passes through untouched.** A client such as Claude Code also counts tokens and lists models through the same base URL; those requests are forwarded with their method, path, query, headers and body, and nothing is recorded. A 404 from the proxy would fail the client because of LLM-GC. The query string is kept on Messages too.

### LLM-GC never makes a conversation worse

Using LLM-GC must never leave a user worse off than not using it. Only the forwarding itself may fail a request:

| Where something fails | What the user gets | What is lost |
|---|---|---|
| Anything before forwarding (parsing, reconciling, collecting) | The original request, forwarded | GC for that one request |
| Background aging, or scheduling it | The full answer | That aging pass; logged |
| The history observer (the dashboard's transcript) | The full answer, with GC | Nothing; logged |
| The provider | The provider's own error | Nothing of ours |
| The upstream rate-limits or is overloaded (429, 529) | The provider's answer; later requests forwarded as before | LLM-GC's own model calls, and so collections, until the pause ends |
| Every one of LLM-GC's own calls in a check fails in a way only a person can fix (a refused key, an account out of credit, a spent quota) | Nothing; requests forwarded as before | LLM-GC's own model calls, and so collections, for ten minutes at a time, shown first on the dashboard until it passes |

**LLM-GC yields to the upstream's limits.** Its summaries and fact reads sometimes can go to the same upstream as the client's requests, often under the same credential, so they share its rate limit. When a client request comes back 429 (rate limited) or 529 (overloaded), or one of LLM-GC's own calls is rate-limited, LLM-GC makes no model call of its own for as long as the upstream's `Retry-After` asks, or a minute when it gives none; a later refusal extends the pause, up to ten minutes. No collection starts meanwhile, and each check skipped is recorded as *paused*, with the reason. The client's requests are never held back: they are forwarded exactly as before, with the recap already made. The dashboard says so on every tab while it lasts.

```
client request ──► upstream ──► 429, retry-after: 30
                                   │
                                   ▼
                      LLM-GC pauses its own calls 30 s ──► checks skipped: "paused"
client requests keep going ──► forwarded as before, recap unchanged
```

**LLM-GC also stops calling when its calls cannot succeed.** When every model call of a check fails with an error that retrying will not fix — the provider refused the key, refused the request without running it for a reason other than its size, as when the account is out of credit, or reported a rate limit it says will not lift, as when a quota is spent — LLM-GC pauses its own calls for ten minutes and tries again after that. The dashboard shows it first, impossible to dismiss, with the provider's own message and what to check. Nothing is collected meanwhile, so nothing loses detail, and the warning says so ([the dashboard](../visibility/overview.md)). A check in which any call worked, or any failure could pass on retry, does not pause. *Rejected: recognising the provider's wording for an exhausted account* — vendor text that changes without notice. *Rejected: pausing after any failed call* — one exchange the provider refuses would stop every collection. *Accepted cost:* a check whose only exchange is one the provider refuses pauses collection for ten minutes, with the reason shown.

*Rejected: retrying a rate-limited call of LLM-GC's own with backoff* — it keeps competing with the client for the same limit, only more slowly. *Rejected: leaving it to the summary calls' own errors* — a refused summary is recorded, but the next collection tries again at once, still competing. *Accepted cost:* while paused, the conversation grows uncondensed; a collection follows once the pause ends.

**A request the provider refuses as LLM-GC rewrote it goes again as the client wrote it.** A 400 to a request LLM-GC changed may be LLM-GC's own doing, and the client must never see it: the client's request is what it would have sent without LLM-GC, so that goes in its place, once. A 400 is decided before any output and is not billed, so the cost is one round trip. The collections the refused request was to carry still wait for one that reaches the model, and both requests are recorded, the refused one marked in the trace and counted as `rewrites_refused`. The provider's reason is logged at WARNING, its error type and message cut to 300 characters: nobody else records it, as a live run found when a client retried a refused request silently. The message names the request's structure, a message index or a block type, not its content; it is logged only for requests LLM-GC changed, and request headers never are. *Rejected: retrying on 429 or 5xx too* — those are the upstream's state, not LLM-GC's rewrite, and the client's own request would meet the same limit. *Accepted cost:* a rewrite the provider keeps refusing pays one refused round trip per request, each logged.

A malformed body is forwarded untouched, because the provider is the authority on its own API. Guards catch `Exception`, never `BaseException`, so cancellation on client disconnect still propagates and the upstream connection is closed.

**Nor a more expensive one, nor a less grounded one.** Never worse has two meanings, and the engine holds both ([when-to-collect.md](../generational-memory/when-to-collect.md)):

- **What the model reasons over.** While a prepared cut waits, the model is sent the client's own turns word for word, as without LLM-GC; a switch never makes the prompt larger; and what has gone quiet is held under a bound.
- **What the conversation costs.** A prepared cut is switched in when the provider's cache is already lost, or when the saving repays the rewrite. Only the bound, and the developer's own *Run GC now*, can switch in a cut that costs more than it saves, and the dashboard labels it.

The proxy supplies the one thing the engine cannot see: whether the conversation's cache is warm. It reads the cached tokens each reply reports and the cache lifetime a request states, and when unsure it reports warm, the direction whose error costs only an opportunity.

### What is measured

Every request is counted where it belongs: its conversation, the client session that ran it, and the proxy as a whole. Counts, ids and model names only, never conversation content. Each figure happened — a request sent, a reply's reported usage, a collection's own model calls. What the *next* request will carry is the engine's forecast, shown as one, never added in.

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

- **The whole request counts, on both sides.** Tokens sent are the system prompt, the tool definitions, the client's instructions and the messages; tokens forwarded are the same request with the recap. GC never changes the system prompt or the tools, but left out, the share saved would be measured against only the part GC can shrink, and read larger than what the model was spared:

  ```
                      system + tools   conversation        share saved
  sent                    20,000          40,000
  forwarded               20,000          10,000
  counted whole           60,000  →  30,000                  50%   ← what the model was spared
  conversation only       40,000  →  10,000                  75%   ← overstated
  ```

  Both sides are counted with LLM-GC's own tokenizer, so the ratio holds while the absolute numbers differ from the provider's. Every text is counted once and remembered, however many requests carry it (see *Why this shape*). Each side also holds the model's earlier thinking as that request sends it ([the model's hidden thinking](#the-models-hidden-thinking)).
- **The provider's own count rides beside it.** Each reply's `usage` — uncached input, cache read, cache write, output — is added wherever its request was counted: from the body of an unstreamed reply, or read from a stream's `message_start` and `message_delta` events as they pass, holding only an unfinished line between chunks. A stream cut short still reports the prompt, which was billed. The provider never sees the uncondensed prompt, so it cannot count a saving: its numbers say what was billed, LLM-GC's compare before and after.
- **LLM-GC's own model calls are counted per collection.** The summary provider is wrapped in a meter ([the provider layer](../llm-provider.md)), so every call a collection makes — the compactor's and the fact reader's, concurrent or not — is counted against that collection and no other. The result carries the calls and tokens, and the proxy adds them to the conversation. A collection that bought summaries and then abandoned its cut still reports what it spent.
- **Bounded.** 1,024 conversations, 256 client sessions and 256 unclaimed first requests are kept; above that the least recently active is dropped, and its requests stay counted in the totals above it.

**A conversation's first request is claimed on its second.** With no history, a first request belongs to no conversation yet. It is held under its opening turn's hash and the client session and agent it named, and the conversation that opens the same way claims it — the same object, so a reply to it still on its way lands there too. Two subagents given the same task are told apart by their agent. A request nothing claims, such as a side request, stays in its client session as *other requests*.

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

**A collection not yet sent is said to be.** A collection is made after a reply, for the conversation's *next* request, so a request must come after it before it saves anything. Until one does, the conversation counts it as *not yet sent*. A finished subagent, or a main conversation `/compact` replaced, keeps it for good: that collection cost its model calls and saved nothing. Whether such collections are frequent enough to change when a collection runs is measured here before it is decided.

```
main conversation   … req 9 ─► reply ─► collection ─► req 10 carries the recap
                                         not yet sent 1    not yet sent 0, saved grows

subagent            … req 5 ─► reply ─► collection ─► (the subagent has answered:
                                         not yet sent 1    no request 6) saved stays 0
```

**Each request is also recorded, on its own.** The ledger keeps sums; the trace needs each request as a row. A record opens just before forwarding and closes when the reply ends: its conversation, its client session and agent, when it went and came back, the tokens it carried, what kind its newest message is — typed by the person, tool results, a relayed report, or the assistant's — and the reply's outline: status, stop reason, the provider's usage, and the tools it called, by name and call id. Structure only, never content: the outline is read from the stream as it passes, and a reply cut short, refused or never answered still closes its record. 10,000 records are kept; the oldest is dropped first, and its counts stay in the ledger.

A record names its newest message by the conversation's chain hash through it, so a later request of the same conversation settles it: *kept* when its history says the same through that message, *replaced* when it says something else there. Two requests at the same position are not compared with each other: until one is carried on, neither is known to be kept.

```
pos 1..5  [ask][Agent ×2][results][running]
req A     ……………………………………………………………… + [suggest a next message]   side request      kept? waits
req B     ……………………………………………………………… + [agent report]            relayed report    kept? waits
req C     ……………………………………………………………… + [agent report][reply][thanks]
                                     history through position 6 = req B's → B kept, A replaced
```

A first request is settled the same way: kept, and claimed by the conversation, when that conversation's history carries it. A request no conversation ever carries on — a title, a quota check — stays unsettled; the trace shows it as a side request. *Accepted cost:* only the newest 64 unsettled requests per client session and agent are watched, so an old one stays unsettled rather than being compared for ever.

**A subagent's first request names the call that started it.** A tool call that starts a subagent gives it a prompt, and the subagent's first message holds that prompt inside the context the client adds around it. So each reply's tool calls that carry a `prompt` input are offered to a linker, and so is each subagent's first message; a pair is made when the prompt is found in the message, and the link is stored on the subagent's first record.

```
main reply      → Agent toolu_a {prompt: "Count the test files…"}      ┐ offered when the reply ends
                → Agent toolu_b {prompt: "Summarise README.md…"}       ┘
subagent a1     first message: [context] "Count the test files…" [context]  → spawned by toolu_a
subagent a2     first message: [context] "Summarise README.md…" [context]   → spawned by toolu_b
                (either side may reach the proxy first; each waits for the other)
```

- **Read by the input's field, not the tool's name.** Claude Code has named the tool `Task` and `Agent`; both take `prompt`. *Rejected: matching every tool input* — in a live trace a subagent's own file path was found inside its opening, a false link.
- **Either side may come first.** The reply ends and the subagent starts within milliseconds, in no fixed order, so a call and an opening each wait for the other, for up to ten minutes. When several could pair, the longest prompt wins - a short one can sit inside a longer one's subagent by chance - and of equal prompts the earliest call, so two subagents given the same task start in order.
- **The prompt is content, held only for this.** It is read from the reply as it streams, kept apart from the outline, held in memory only until matched, never logged, and only while the dashboard runs. An input larger than any prompt is let go while it streams. A subagent no call matches is shown as started by a call not determined, never guessed.
- **Not yet linked: a report to its subagent.** A relayed report is shown as a subagent report, in its place in time; which subagent sent it is not named, because a live trace matched only one report in two by its words.

*Rejected: keeping the counts in each session's state* — sessions close when idle, and their requests would vanish from the client session's total. *Rejected: asking the provider to count each request (`count_tokens`)* — a network call per request for a number the reply's `usage` already gives, and the uncondensed prompt is never sent, so it could not count the saving either. *Rejected: one process-wide counter for LLM-GC's own calls* — two conversations collected at once would each be credited with the other's. *Accepted cost:* grouping by client session relies on Claude Code's headers; a client that names no session has each conversation listed on its own.

`/stats` returns the proxy's totals — requests, GC applied and bypassed, `tokens_from_client`, `tokens_forwarded`, `tokens_saved` and their ratio, the provider's reported usage, the cache written on requests that carried a collection (`collection_cache_write_tokens`), LLM-GC's own calls, and the reconciliations `history_rewinds`, `history_restarts` and `sessions_seeded` (a restart or a seed costs one uncompressed request, so each is counted rather than hidden) — then `client_sessions`, each with its conversations: their part (main, subagent, earlier main), model, window, counts and newest reply counted both ways, with its rate; and `other_conversations`, for those no client session named.

`llm-gc stats` reads these from a running proxy, on the port the settings name, and validates the answer before using it, since whatever is on that port may not be an LLM-GC proxy. It prints the totals grouped by whose count they are, each group headed by that name rather than explained in a closing note:

```
LLM-GC's own count         an estimate, the same on both sides  →  sent, forwarded, saved (share)
The provider's own count   what it bills                        →  the replies' usage by kind,
                                                                    LLM-GC's own calls
```

then what the proxy handled (requests, conversations), then one table per client session: each conversation's requests, tokens sent and saved, collections (and those not yet sent), window, and the rate its newest reply measured ("Model count"). Sent and saved add up every request, each whole, so they grow far past any one conversation; the share is the figure to read. A share rounds halves up, as the dashboard does, and a saving too small to round to 1% reads "<1%", never "0%".

**LLM-GC's own calls sit beside the saving, never taken from it.** A token saved and a token spent are not worth the same: with a warm cache a saved token would have been a cache read on the proxied model, the cheapest kind, while a spent one is input or output on the summary model. And neither figure holds what a collection costs on the proxied model: the prompt after the recap written to the cache again. *Rejected: a net saving (saved minus spent), or a saved-per-spent ratio* — both subtract or divide different kinds of token on different models, and both leave the cache writes out, so they read as a gain on a session that costs more than it would without LLM-GC. The provider's group names the cache written on requests that carried a collection instead, the cost to set the saving against, as the Session tab does. *Rejected: re-counting LLM-GC's calls with LLM-GC's tokenizer so both sit on one ruler* — that makes the counts alike, not the tokens. "Applied" counts requests that went through LLM-GC's path, which includes the early ones where nothing has been condensed yet; "bypassed" counts requests forwarded untouched because they could not be read or the engine failed on the request path. With no requests it says so instead of printing zeros, because a ratio with nothing behind it is not a measurement. Rewinds and restarts appear only once they have happened: only a client sending the session header can cause them, so for Claude Code they would be a permanent row of zeros. `--json` always has every counter.

### The model's hidden thinking

A model that thinks before it answers puts that thinking in its reply as a block, the client sends the block back with every later request, and the provider bills it as input. The block carries no text LLM-GC can count: its readable part is empty or a short summary, and the thinking itself travels encrypted in the block's signature. Counted as text alone, every request looked smaller than it was, the gap grew with the conversation, and a switch's saving read as a fraction of the real one.

So LLM-GC measures each reply's thinking once, as the reply passes through, and gives it back to the turn whenever the client sends that turn again:

```
 a reply passes ──► its usage: output tokens, thinking and visible text together
                    its first thinking block ──► key: a hash of the signature's first characters
                    remembered: key → the output, the model's rate, the model

 a later request sends the turn back ──► the same block, the same key
     thinking ≈ output ÷ rate − the turn's visible text, in LLM-GC's count
     counted only when the request names the model that wrote it
```

- **Measured from the provider's own number.** A reply's output count holds its thinking and what it shows; LLM-GC counts what it shows and takes that off. Nothing extra is asked of the provider.
- **Known by its signature.** The signature is unique from its first characters and grows with the thinking, about four characters a token, so a long think streams as one line of hundreds of kilobytes. The stream reader never holds a line that long: it reads the key from the start of the line and lets the rest go. Only a hash is kept, never the signature or the thinking.
- **Counted as it is sent.** Tokens sent include every turn's thinking, because the client's request carries all of it. Tokens forwarded include only what is left once a collection has stripped it ([after a cut](#after-a-cut)); the count and the rewrite read one rule for which turns keep theirs.
- **Only for the model that wrote it.** A model that cannot read another model's thinking drops it, and the provider does not bill what it drops. Which models read whose blocks differs by pair and changes with releases; the model that wrote a block always reads it.
- **Bounded.** 20,000 replies are remembered, and the one sent back least recently goes first, so a conversation in progress keeps its own. A turn whose reply was let go counts as carrying none.
- **Provider-neutral past the translation.** Only the Anthropic translation knows which blocks are thinking and how to key them; the engine sees one number per turn. A provider whose replies send no thinking back never makes a key, and its turns carry none.

*Rejected: the provider's token-counting endpoint* — one more call for every reply, when the reply's own usage already holds the number, and a gateway may not offer it. *Rejected: reading the size from how much the next request grew* — the noise in one step is as large as the thinking. *Rejected: reading the size from the signature's length* — the signature's format is the provider's own, undocumented and free to change. *Rejected: labelling the counts as text only* — honest, but they could not be set against the provider's, and the switch policy would keep undercounting what a switch saves.

*Accepted costs:* the size is an estimate, and the dashboard shows it apart from the text, labelled as one. Replies from before LLM-GC started, or before a restart, carry none. A model that discards earlier thinking on its own, as older models do, is overcounted; that shows as the provider's count falling behind LLM-GC's over a conversation.

### Images and documents

LLM-GC's text reads an image or a document as a placeholder, `[image]` or `[document]`, and that is what is scored, summarised and quoted. The provider bills the attachment itself: a screenshot costs a recent model a few thousand tokens. Counted as a placeholder, four screenshots in one live conversation left about 13,000 of the provider's tokens out of LLM-GC's count, and the provider ÷ LLM-GC ratio stepped at each.

So the first time an attachment appears, the proxy asks the provider's token-counting endpoint what it costs the model the request names, with LLM-GC's own credential, in the background. It counts one message with the attachment and one without, because the endpoint may add tokens of its own to every message, and keeps the difference. The size is kept per model, because a newer model can cost an image more than an older one, and converted to LLM-GC's count at the model's rate when a turn holding it is read. It counts in what each request sends and forwards: a turn sent as written carries it, and a collected turn does not, since the front quotes only the placeholder. The engine weighs it in a switch's saving, what it writes again and its stale weight ([when to collect](../generational-memory/when-to-collect.md)), and the dashboard shows it apart from the text ([visibility](../visibility/overview.md)).

```
 request N    a turn brings a screenshot ──► counted as its placeholder
                                             and, once, in the background:
                                             count([image, "."]) − count(["."]) ──► (model, key) → 3,700

 request N+1  the turn is history ─────────► + 3,700 ÷ the model's rate, while the turn is sent as written

 collection   the turn is taken ───────────► the front quotes "[image]"; the image is no longer sent
```

- **Known by its ends.** The key hashes the attachment's kind, its data's length and the first and last few thousand characters of its data. A client sends every image again with every request, so hashing whole images would put megabytes on the request path each time. Two images that agree in length and at both ends have the same dimensions, so the same size.
- **Never on the request path.** At most eight counts are asked at once; beyond that, an attachment is asked about on a later request. The endpoint is free, and its rate limit is apart from the one the client's requests share.
- **Bounded.** 4,096 sizes are remembered, failures included, and the one sent back least recently goes first.

*Rejected: reading the size from how much the prompt grew on the request that brought it* — it fails on the request that carries a collection, which is a new user turn, where a person pastes a screenshot, and a conversation's first request has nothing to compare against. *Rejected: working the size out from the image's dimensions* — the provider's formula changes between model generations, and it would be kept by hand. *Rejected: adding the size to the turn's text count* — the compactor sizes a summary from that count, and would ask for a summary of a few hundred words of text as long as a screenshot.

*Accepted costs:* the request that brings an attachment counts it as its placeholder. An attachment given by URL or file, which the endpoint does not count, an upstream without the endpoint, and a proxy without a credential of its own leave it unsized, as does a count that failed, which is not asked again.

### Each conversation's window

Each conversation names its model on every request, and one session runs several — a main conversation on one model, its subagents on another. The window of each model is asked of the upstream's Models API (`GET /v1/models/{model}`, its `max_input_tokens`) with LLM-GC's own Anthropic credential: once per model, in the background, never on the request path. Until the answer arrives, or when there is none — a gateway that lists no models, a model it names its own way, no Anthropic credential — the configured `context_window` is used, and the dashboard and `llm-gc stats` say it is assumed. The conversation's recap budget is its share of that window ([detail levels](../generational-memory/detail-levels.md)), read at its next collection. At most 64 model names are asked about, so a client cannot make LLM-GC call the upstream without end.

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

the conversation's recap budget = its share (30%) of that window, read at its next collection,
                                   converted to LLM-GC's count at the rate its replies measure
```

**Each request also measures what a collection would make the provider cache again** besides the recap and the recent turns: the client's instructions, each counted once, as the recap carries them, and the part of the system prompt and tools after the client's last cache marker in them, which no cached read reaches once the messages change. The engine adds it to the cost a switch must repay ([when to collect](../generational-memory/when-to-collect.md)).

**The rate between the two counts is measured on requests LLM-GC counts exactly: no thinking, no image, no document.** Thinking is counted by estimate, and an image or a document counts as a placeholder while the provider bills it at its real size, a few thousand tokens for a screenshot, so a request holding either would skew the rate. Where a reply's usage is read and its request held none of them, the proxy divides the provider's count of that request's whole prompt (input, cache read and cache write) by LLM-GC's count of the same request, and gives the engine that rate for the conversation. The request that carries a new front sends no thinking, so the rate is measured again at each collection whose recent turns hold no attachment. A conversation's first request sends none either, but it arrives before the conversation is known, so it counts toward its model's rate, and a conversation that has not measured its own uses its model's. A model's rate is all its measured requests' provider tokens over all their LLM-GC tokens, so each request weighs by its size: a small side request, such as a title the client asks for, has a different mix of text and must not replace a rate measured on a whole conversation. The engine keeps the window in the model's tokens and takes the recap budget and the stale bound from the window converted at the rate ([detail levels](../generational-memory/detail-levels.md)). A reply that reports no usage measures nothing, and the rate stays as it was. A model's rate also converts its replies when their thinking is measured.

```
 measured on a request that sends thinking:
     rate     = the provider's prompt ÷ (text + LLM-GC's thinking estimate)
     estimate = a reply's output ÷ rate − its text
 each feeds the other. With a model that does not bill earlier thinking, a smaller rate
 makes a larger estimate, which makes a smaller rate: in a test, one reply's estimate
 grew from 397 tokens to 23.8 million in eight turns.
```

*Rejected: measuring on every request* — the loop above. *Rejected: taking the estimate off the provider's prompt before dividing* — the estimate still feeds its own rate. *Rejected: a model's latest rate in place of its weighted one* — one small request, outside any conversation, then sets the rate every later reply is converted at. *Accepted cost:* the rate changes less often, and a conversation whose recent turns always hold an attachment measures none of its own. The first request of a conversation is mostly system prompt and tool definitions, so until its first collection the rate fits those better than the conversation's text. A collection pass is scheduled as its request is sent, so it uses the rate of the reply before; the rate moves little between requests.

A larger window cannot push a request past the model's: the recap is the conversation's own older turns, condensed or quoted, so even with every span restored it is about the size of the turns it stands for, which the client would send anyway. A larger budget only means a long conversation's recap is squeezed later. *Rejected: a table of windows in code* — stale with each model release, and blind to a gateway's names. *Rejected: asking on the request path* — the first request of every new model would wait on it. *Accepted cost:* a model whose larger window needs a header the client did not send is sized for the larger window; the bound above still holds.

## Why it exists

The engine is a library; the proxy is how a real application uses it without changing its code. It is also the boundary where LLM-GC meets a conversation it does not control — a client that retries, regenerates, edits, streams, attaches images, or calls tools — so it is where the engine's assumptions about a conversation are tested against what clients actually do.

## Why this shape

- **Record from the client's history, not from responses.** Recording replies parsed out of responses failed in every case clients actually produce: a regenerate forwarded the answer being replaced, a retry duplicated turns, a tool-call reply had no text and was lost, a streamed reply needed a parser, and the record could drift from what the user saw. *Accepted cost:* a one-exchange lag — the newest exchange reaches the record, and aging, one request later. Measured on a 12-request demo the forwarded share rose from 0.917 to 0.965 of the client's history; over a long conversation the lag is the last two turns of many, which aging would keep verbatim anyway. Removing it would mean recording a turn before knowing the request succeeded, which breaks retries.
- **Identify a conversation by chain hashes, not by fingerprinting its opening.** Hashing the first two turns gave a conversation's first and second requests different identities, so the second request of every headerless conversation reached the model without its history. A chain hash identifies a conversation by everything up to a point, works from the second request, and cannot merge two conversations that differ. *Rejected: requiring the session header* — correct, but almost no client sends it.
- **Ignore a session name that is not a plain id; never refuse the request or clean the name.** Matching by content finds the conversation without the name. Only an edit loses its rewind, and it then costs one request sent whole, which `sessions_seeded` counts. *Rejected: refusing the request* — it fails a conversation that works without the name, which is worse than no LLM-GC. *Rejected: removing the characters that fail* — `chats/42` and `chats42` would become one session, and two conversations would share what either of them said. The rule starts narrow because widening it later breaks no client, and narrowing it would.
- **Canonical JSON when hashing.** An SDK that serialises the same content with its keys in another order must not look like an edited conversation, which would cost a needless restart.
- **Forward the client's bytes whenever nothing changed.** Rebuilding every turn as plain text dropped attached images and tool calls, and any rewrite resets the provider's prompt cache.
- **Read a turn as text, with its tool calls and results written out.** The engine sizes, scores, summarises, reads facts from and quotes a turn by its text alone. A tool call reads as `[tool call: name] {input}` and its output as `[tool result] ...` (`[tool result, error] ...` when the tool failed); an attachment reads as `[image]` or `[document]`; thinking has no text form, and its size is estimated apart ([the model's hidden thinking](#the-models-hidden-thinking)). In an agent's conversation most turns are tool rounds, so reading only the words counted them as empty: the prompt looked smaller than it was, summaries were written without what the tools did, and a span kept word for word lost every tool output while being shown as complete. *Rejected: counting the blocks' tokens separately* — the sizes come right, but summaries and quotes still lose the output. *Rejected: a second text field for tool content* — every reader needs it, and a second field invites the next reader to read the wrong one. *Accepted cost:* a large tool output (a file read in full) makes the summary and facts calls for its span as large.
- **Count each text once, and remember the count by a hash of the text.** A client sends its whole conversation on every request, so almost all of a request's text was counted by the one before it. Tokenising is the costliest step in reading a request, it grows with the conversation, and it runs on the event loop, where it stalls every other request and stream the proxy relays, parallel subagents included. So a text's count is remembered, and only text not seen before is tokenised. Measured on a conversation of 400 turns and 140,000 tokens, tokenising was about 30 of the 35 ms of counting in a request, and handling a whole request fell from about 46 ms to about 11 ms. The same memory serves every figure that counts a text: the turns, the client's instructions (which several figures count), the system prompt and tools, the front, each span as the planner prices it, and the dashboard's copy of the conversation.

  ```
  request N+1 ──► hash each text ──┬─► found: use the remembered count        (turns sent before)
                                   └─► not found: tokenise, then remember it  (the new turns)
  ```

  A count depends on nothing but its text, so a remembered count can never be out of date and nothing has to be invalidated. *Rejected: reusing the counts the session recorded, found by the turns' chain hashes* — the chain hash ignores on purpose what does not change what was said (the client's instructions, the model's thinking, cache markers), so two texts it calls equal can differ in size; the lookup would also need the session found before the request is read, and the instructions and system prompt would still be counted on every request. *Rejected: keeping the texts themselves as keys* — that bounds the number of entries, not their size, and keeps a second copy of the conversation alive; a tool result can run to hundreds of kilobytes. *Rejected: counting in a worker thread* — it hides the stall but keeps the work, which every request, and every parallel subagent, would still pay. *Accepted cost:* hashing each text on every request, about a millisecond per megabyte, and a text whose count was dropped (the memory holds 50,000, about 8 MB, least recently used first) is tokenised again.

### After a cut

Once the engine has switched a session to a cut, the forwarded conversation is the front (one user-role message) followed by every later turn as the client's **original content** — images, tool calls and all. Thinking blocks are removed from every turn sent with the first request that carries a new front, because each was produced against a prompt the cut replaced; that is the request that switched, and its history length becomes the boundary. Replies produced from then on keep theirs. Between switches the forwarded conversation only grows at the end, so the provider's prompt cache holds. See [the cut](../generational-memory/the-cut.md).

### Idle sessions

A session holds its conversation's content, so the proxy runs the session idle sweeper for as long as it serves: a session untouched for a day is closed and its data dropped. A client that returns later is seeded afresh from its own history, at the cost of one uncompressed request.

### The dashboard

The live dashboard runs in the same process on its own localhost port, reading the same sessions; it is opt-in because it shows conversation content. To show what was said beside what the model still has, the proxy hands each request's conversation to an optional observer before forwarding, and each finished reply to another once the client has it all; both are attached only when the dashboard runs, and neither can fail a request. A reply is read for showing only, in the same text form its turn will have once recorded; one that stopped before its end, or longer than 4 MB, is not shown. See [visibility](../visibility/overview.md).

```
request N ──► forwarded ──► reply streams to the client ──► copy read: shown as "newest reply"
request N+1 carries the reply in its history ──► recorded; the shown copy is dropped
```

### Following one request

Each request the proxy receives gets an ID. Every log line written while handling it carries the ID, and so do its rows on the dashboard's Requests tab and any error body LLM-GC writes itself. A row that looks wrong, or an error a client reports, leads to the request's lines in the log.

```
client request ──► ID 3f2a9c01b7de, held for as long as the request runs
   ├─ every log line   14:02:11 INFO llm_gc.proxy.server [3f2a9c01b7de]: claude-sonnet-5 -> 200 in s_0676…: …
   ├─ Requests tab     hover the request's row: "In the proxy's log as request 3f2a9c01b7de"
   └─ LLM-GC's errors  {"type": "error", "error": {…}, "request_id": "3f2a9c01b7de"}
```

The ID is held in a context variable that a middleware sets for the whole of the HTTP request. The engine's lines therefore carry it without the library knowing about requests, and a streamed reply, which is relayed after the route has returned, still has it when the stream drops. Work the request schedules, such as the aging pass after its answer, keeps the ID, because an asyncio task starts with a copy of the context it was created in. A filter on the application's log handler adds the ID to each line; the library sets nothing.

A request LLM-GC sends twice, because the provider refused its rewrite, is still one request of the client's, so both of its records carry the same ID. The trace tells the two attempts apart by record, not by ID.

An error LLM-GC answers itself (a 502 when the upstream cannot be reached, a 413 for a body over the limit) uses the provider's error envelope, with LLM-GC's ID where the provider puts its own: the request never reached the provider, so there is no ID of the provider's to give. An error the upstream answered passes through unchanged, carrying the provider's own ID, and the answer line names that ID, so the error a client shows leads to the line.

Each answer line also says how long LLM-GC took, from having the request's body to forwarding it, and how long the upstream took to answer. For a streamed reply that is how long it took to begin answering, because the line is written before the stream is relayed; the whole reply's time is on the Requests tab.

The dashboard gives its own requests IDs the same way, so its errors and a manual collection lead to their lines too ([visibility overview](../visibility/overview.md)). The ID sits in a small package both servers share, so the dashboard, which is built to run without a proxy, does not import one.

Logs are text by default, for the person watching the terminal. With `log_format = "json"` each line is one JSON object holding every field the line carries, for a program that reads the log. The web server's own warnings and errors, a route's traceback among them, go through the same handler, so they are in the chosen format too.

*Rejected: an ID per forward* — the refused rewrite and its resend would have two IDs for one client request, and the lines written before the first forward, such as a conversation starting, would belong to neither. *Rejected: setting the ID at the top of each route* — a streamed reply is relayed after the route returns, so a dropped stream would be logged without its ID. *Rejected: returning the ID in a response header* — no client shows one; the error body and the Requests tab are where a person looks. *Rejected: JSON by default* — it would replace the console a person reads while LLM-GC runs. *Accepted cost:* an aging pass's lines carry the ID of the request that scheduled it, which may have finished minutes earlier.

### Request size

A body larger than the provider's own request limit (32 MB) is refused with a 413 before it is buffered. The provider would reject it anyway, so refusing it costs no user anything; the limit is enforced while reading, because `content-length` may be absent or wrong. The refusal is logged, since no other line is written for the request, so the ID in the client's error leads somewhere.

## Known limitations (planned, not yet built)

- **A cut turn's attachments** reach the front as a placeholder, `[image]` or `[document]`, and are counted as one: the front is text, so a collected image is named, not shown, and its size is undercounted.
- **Anthropic Messages only.** The OpenAI-compatible endpoint is not built.
