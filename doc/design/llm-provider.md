# LLM Provider Layer


LLM-GC sometimes needs a language model to do part of its own work. The main
case is compaction: replacing a run of old conversation turns with a short
summary that keeps what mattered. Writing that summary takes a model. The
provider layer (`llm/`) is the only place in the engine that calls one. The rest
of the engine asks it one thing, "here are some messages, give me text back",
and never learns which vendor answered, what that vendor's API looks like, or
how that vendor reports failure.

## Two models, two jobs

A conversation running through LLM-GC involves two model calls that are easy to
confuse:

```
  User's app ──► LLM-GC proxy ──────────────────────► Conversation model
                     │                                (the user's chat)
                     │ after the response is sent
                     ▼
                 GC engine ── "summarise turns 5–12" ──► llm/ ──► Compaction model
                                                                  (LLM-GC's own work)
```

|                    | Conversation model                          | Compaction model                               |
|--------------------|---------------------------------------------|------------------------------------------------|
| What it does       | Answers the user                            | Writes summaries for the garbage collector     |
| Who picks it       | The user's app, on every request            | Whoever runs LLM-GC, once, at startup          |
| Whose API key      | The user's, forwarded and then dropped      | The deployment's own                           |
| When it runs       | During the request                          | After the response, in the background         |
| Where it's set     | The incoming request (`GCConfig` describes its context window) | The provider client handed to the compactor |
| Which package      | `proxy/` (inbound)                          | `llm/` (outbound)                              |


## How it works

**One small interface.** `LLMProvider` is a `Protocol` with a single async
method, `generate`. It takes messages, an optional system prompt, an output cap
and a timeout, and it returns an `LLMResponse`: the text, how many tokens went in
and came out, which model actually answered, and why generation stopped. It
offers less than any vendor API on purpose. Everything else a vendor provides is
that adapter's business.

**One adapter per vendor.** Each adapter (Anthropic, OpenAI) does three
translations so its callers don't have to:

- **Request and response shape.** Vendors differ on the details. Anthropic takes
  the system prompt as a separate field and returns a list of content blocks.
  OpenAI takes it as a message and returns a list of choices. They also name
  token counts differently.
- **Failures.** Every vendor error becomes one of LLM-GC's own exceptions:
  timeout, rate limit, auth, connection, context overflow, or bad response.
  Callers catch on the kind of failure, not on the vendor.
- **Stop reasons.** Each vendor says "I finished" or "I ran out of room" in its
  own words. The adapter maps them onto one `StopReason`: complete, truncated,
  filtered, or other.

These differences only affect the call that *produces* a summary. The summary
that comes back is plain text, wrapped in an ordinary LLM-GC `Message` that goes
into the conversation like any other turn, whichever vendor wrote it.

**A fake for tests.** `FakeProvider` implements the same interface from a script
of responses and errors. It records every call. Unit tests run against it, so
they make no network calls and can simulate a timeout on exactly the second
call. It ships inside the package, so library users can test their own code the
same way.

## Why it exists

Compaction is how LLM-GC saves context, and compaction needs a model. Without
one, the only compactor is a placeholder that copies the turns under a marker.
That adds tokens instead of saving them.

The engine still shouldn't know which model it's using. LLM-GC is a library as
well as a proxy, and people who import it already have a provider and an API key
of their own. Switching provider should be a configuration change, never a code
change.

Keeping this in `llm/` also keeps three areas apart that share vocabulary and
little else. `llm/` is the engine calling *out* to a model. `proxy/` handles
requests coming *in* from a client. `wrappers/` is a convenience layer for
library users who don't run the proxy. Merging any two would tie the engine to
the web layer.

## Why the compactor has its own model and key

The obvious shortcut is to reuse what's already there: the proxy already has the
user's model name and API key, so why not summarise with those? There are two
separate parts to that question.

### Why not the user's API key

Holding the key in memory while forwarding a request is unavoidable and fine.
That is simply what forwarding means. The trouble is *when* compaction runs. It
happens in `gc.update`, after the response has gone back to the user, as
background work. By then the request is finished. Reusing the user's key means
copying it into session state, which lives across requests and for up to half an
hour of idle time. At that point the key is stored, even though it's only in RAM.

RAM is not a safe place in itself. Keys leak when memory gets copied somewhere
else, and this codebase has several routes for that:

- **Default `repr`s.** Session state is a dataclass, and its automatic `repr`
  prints every field. One debug log line, one traceback that captures local
  variables, or one error reporter, and the key ends up in a log.
- **Events and the dashboard.** Events carry session information to the
  dashboard, which serves it over HTTP. A key held in session state is one
  careless field away from that path.
- **Persistence.** A later storage layer that saves session state to disk would
  quietly turn "in memory" into "on disk".
- **Session mix-ups.** Session identity will be inferred from requests. If two
  users' requests are ever matched to the same session, one user's compaction
  runs on the other user's key and bill.

Beyond leaks there is consent. The user handed over their key to send *their*
request. Spending it on extra calls they didn't make bills them for our work and
uses up their rate limit. A rate-limit error caused by our compaction could block
their next real message.

Library users show the same problem from the other side: a library call has no
incoming request, so there is no user key to borrow. The compactor has to be
configurable on its own either way.

### Why not the user's model

The conversation model is whatever the client asked for on that request. It
might be a vendor we have no adapter for, a small local model that summarises
badly, or a different model on the next turn. Summary quality then depends on
something LLM-GC neither chooses nor sees. One of LLM-GC's central claims is 
context savings without quality loss. That claim has to rest on a benchmark, and 
a benchmark needs a fixed, known summariser.

Cost matters too, though it isn't decisive by itself. Summarising a run means
paying to read the whole run once. With a cheaper compaction model that cost is
small next to the savings on every later request. With an expensive conversation
model it eats into them.

### What this costs, and what would change it

A deployment has to configure a second credential. That's real setup friction,
and it is the price of the points above.

There is also a privacy cost, and it runs in the opposite direction. Every
compaction sends the user's conversation turns to the compaction vendor. If that
vendor is not the one the user chose for their chat, their conversation now
reaches a second company. The deployment that picks the compaction vendor is
responsible for disclosing it. A local model keeps the text on the machine, and
turning compaction off avoids the second vendor entirely.

A "reuse the conversation's provider"
mode could still be added later as an explicit opt-in for single-user, local
setups, where the person running LLM-GC and the person whose key it is are the
same. Because the compactor receives its provider rather than building one, that
mode would be a new provider wiring, not a redesign.

## Why this shape

- **`generate` is async, and so is everything that reaches it.** The real caller,
  `GcService.gc_update`, already runs on the event loop. A blocking HTTP call
  there would freeze every other session on the server for the several seconds a
  model call takes. That looks fine in a one-person demo and falls over the first
  time two people use it. The rejected alternative was a synchronous interface
  that callers move onto a thread. It keeps a simple entry point for library
  users with no event loop, but it makes offloading a rule every caller must
  remember, and forgetting it fails silently. The cost accepted: the stateless
  `GarbageCollector.collect` becomes a coroutine, so synchronous library users
  need `asyncio.run` or a small convenience wrapper.


- **The interface is checked by the type checker.** Python doesn't enforce type
  annotations at runtime, so on its own a `Protocol` is just documentation. It is
  deliberately not `@runtime_checkable`. That check only confirms method *names*
  exist and would pass an implementation whose signature is wrong, which reads
  like verification without being any. Instead, every implementation carries a
  type-check-only assertion that it conforms, and the package is type-checked
  strictly. Without that, an adapter whose signature drifted would pass every
  test that uses the fake and fail on the real request path.


- **Two real adapters, not one.** An interface designed while looking at one
  vendor quietly takes that vendor's shape, and you can't see it by reading the
  interface. The second adapter is the test that the interface is genuinely
  shared. It isn't there as an extra feature. For a surface this small it costs
  well under a hundred lines. The rejected alternative was shipping one adapter
  now and generalising later, and in practice "later" never comes.


- **Errors are translated in the adapter, not by callers.** Request shapes are the
  obvious difference between vendors and the least dangerous one. What really
  ties code to a vendor is the failure path: what's worth retrying, how a rate
  limit arrives, how "your input is too long" shows up. An interface that covers
  only the success path pushes vendor-specific error handling into every caller.


- **Retryability is carried by the exception, not inferred from its type.** The
  obvious design — a retry policy with a list of retryable exception classes —
  does not survive real providers, because the same category spans both answers.
  A failing status arrives as one error type whether it was a 503 or a 400, and a
  rate limit arrives the same way whether the limit is a per-minute window or an
  exhausted account balance. Only the adapter saw the status code and the error
  type, so only the adapter can tell them apart. The class declares the usual
  answer and the adapter overrides it per instance. The cost of getting this wrong
  is not a crash: it is a retry budget spent arriving at a certainty, while the
  one error message that tells an operator what to fix arrives last.


- **The narrower vendor decides the interface.** Where two providers disagree, the
  interface takes the shape only one of them can express, because the wider
  vendor can always be reached from the narrower shape and not the reverse.
  System instructions are the worked example: one vendor takes them as a
  top-level field with no system role at all, the other as an ordinary message in
  the list. `generate` therefore takes `system` as its own argument, and the
  adapter that needs a message builds one. Taking the other side would have made
  the field unrepresentable for one adapter, and the interface's rules must be
  identical at both or code written against one silently breaks on the other.


- **Stop reasons are translated too, and unknown ones are never "complete".** This
  translation is the easiest to skip, because it sits on the success path: the
  call returned 200 and the text looks fine. But a summary that hit the output
  cap has silently lost the end of what it was summarising, and nothing
  downstream can tell. An unrecognised stop reason maps to `OTHER`, not
  `COMPLETE`, because "complete" would claim something we can't back up.


- **No `temperature` parameter.** Current Claude models reject it, while OpenAI
  accepts it everywhere. A parameter one adapter would have to drop silently
  isn't a shared contract, and dropping it would mislead callers into thinking
  they had asked for determinism. Compaction gets its consistency from the prompt
  and, on Anthropic, from an adapter-specific effort setting.


- **Plain HTTP (`httpx2`) instead of vendor SDKs.** Two adapters ship, so using
  SDKs would make every library user install two vendor packages to get a working
  compactor. We only need one non-streaming call, so we give up little. The cost
  we accept is owning the wire format and its changes over time. Responses are
  therefore validated with Pydantic rather than read by indexing into
  dictionaries, so a changed response body fails at the boundary with one clear
  error instead of turning up as a `None` several layers later. `httpx2` is the
  successor to `httpx` and is what the Anthropic SDK itself uses, so adopting
  that SDK later wouldn't mean running two HTTP stacks.


- **A fake instead of mocking the adapters.** One fake that implements the
  interface keeps tests honest about the interface. Mocking each adapter would let
  tests pass against a response shape that no real vendor produces.

## What the second adapter found

The second adapter is not a feature, and this is the evidence. Each of these was
invisible while only one adapter existed, and none of them is visible by reading
the interface:

- **A parameter that cannot be shared.** `temperature` was on an earlier draft of
  `generate`. One vendor rejects it outright on current models and the other
  accepts it everywhere, so an adapter would have had to discard it silently —
  the dishonest option, because the caller would believe it had asked for
  determinism it never got. Removed from the interface.
- **A retryability answer the exception could not express.** One vendor reports an
  exhausted account quota with the same status code as ordinary throttling.
  The rate-limit error declared retryability once, per class, so the permanent
  case was unrepresentable. It became per-instance, like the response error
  already was.
- **A name that stopped meaning one thing.** The default compaction model was a
  single unqualified constant, which reads fine until there are two of them.
- **A fragile match that is not a style choice.** One vendor reports context
  overflow with a machine-readable code; the other states it only in the error
  prose. The substring match in the second adapter looked like laziness until the
  first one showed what the alternative is.

## Consequences

Every vendor quirk has exactly one home. Adding a vendor means one new adapter
and its error mapping, with no changes to the engine.

The fake is load-bearing, because every unit test runs against it. A fake that's
more forgiving than reality will let a green test suite hide a broken adapter: one
that never times out, never returns a malformed body, never reports usage
differently. So its failure behaviour has to be modelled as carefully as its
success behaviour.

## Not yet built

The interface, the result type, the exception hierarchy, the fake and both
adapters exist. Still missing: the retry and backoff policy, and the prompts
directory.

The two adapters also duplicate a little genuine HTTP mechanics — reading
`Retry-After`, digging a message out of an unreliable error body, the set of
statuses worth repeating. That is deliberate for now: extracting shared code from
a single example is guessing which parts are generic, and the right time to do it
is once two implementations have shown which parts actually are. It will be free
functions rather than a shared base class, because the two adapters have HTTP
mechanics in common and no domain logic, and a common base class is how adapters
acquire a shared "almost right" method that neither vendor wants.
