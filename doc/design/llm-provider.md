# LLM Provider Layer

> **Not yet built.** This doc records the intended design so the decision is
> settled before the code exists.

## How it works

`llm/` is how the engine makes its *own* outbound LLM calls — today that means
the compactor asking a model to summarise a run of turns.

A `LLMProvider` **Protocol** defines the surface the engine needs, deliberately
smaller than any vendor's API: messages in, generated text plus token counts out,
under a caller-supplied timeout. Three implementations ship: an Anthropic client,
an OpenAI client, and a fake used by every unit test.

Clients talk to providers over `httpx` directly rather than through vendor SDKs.

Each client owns three translations, and they are the whole job:

- **Request and response shape.** The providers genuinely differ — Anthropic
  takes `system` as a top-level field and returns `content` as a list of blocks;
  OpenAI takes system as a message and returns `choices[0].message.content`.
  Token usage is `input_tokens`/`output_tokens` against
  `prompt_tokens`/`completion_tokens`.
- **Error taxonomy.** Each client maps provider failures onto LLM-GC's own
  exception hierarchy — `LLMTimeoutError`, `LLMRateLimitError`,
  `LLMResponseError` — so callers never branch on which provider they got.
- **Stop reasons.** Anthropic says `end_turn` / `max_tokens` / `stop_sequence`;
  OpenAI says `stop` / `length` / `content_filter`. Each client normalises these
  onto a `StopReason` enum, so a caller asking the only question that matters —
  *is this output whole?* — does not have to know both vocabularies.

## Why it exists

The compactor needs a model, and nothing in the engine should know which one.
Beyond that, the provider surface is the one place where "swap the model" has to
stay a configuration change rather than a code change: the library is meant to be
imported by people who already have a provider and an API key.

Keeping this in `llm/` also separates three things that are easy to conflate.
`llm/` is the engine calling *out* to a model. `proxy/` translates requests
coming *in* from a client, in the other direction. `wrappers/` is a convenience
layer for library users who do not run the proxy. They share vocabulary and
nothing else; merging any two of them couples the engine to the web layer.

## Why this shape

- **`generate` is async, and the compaction path is async with it.** The
  production caller already is: `GcService.gc_update` is a coroutine, and it is
  what reaches the compactor. A blocking HTTP call there stalls the event loop —
  and therefore every other session on it — for the multi-second duration of a
  model call, which is the kind of fault that looks fine in a single-session demo
  and collapses the first time two people use it. The rejected alternative was a
  synchronous Protocol with callers offloading to a thread: it keeps a
  synchronous entry point for library users who have no event loop, which is a
  real benefit, but it makes the offload a rule every caller must remember, and
  forgetting it fails silently under concurrency nobody reproduces locally. The
  accepted cost is that the stateless `GarbageCollector.collect` becomes a
  coroutine, so a synchronous library user needs `asyncio.run`; a convenience
  wrapper can restore that cheaply, while un-blocking a blocked event loop after
  the fact cannot.
- **The Protocol is enforced by a type checker, or it is a comment.** Python does
  not check annotations at runtime, and this Protocol is deliberately not
  `@runtime_checkable` — that would only confirm attribute *names* exist, passing
  for an implementation whose signature is wrong in every other respect, which is
  worse than no check because it reads like verification. So every implementation
  carries a `TYPE_CHECKING`-guarded conformance assertion, and the package is
  type-checked strictly. Without that, a client whose signature drifted from the
  interface would pass every test that used the fake and fail on the request path.
- **Two real clients from the start, not one.** A `Protocol` written while
  looking at a single provider silently takes that provider's shape, and the
  defect is undetectable by inspection — an interface with a `system=` parameter
  is Anthropic-shaped, one returning `.content[0].text` is Anthropic-shaped. The
  second adapter is the *test* of the abstraction, not a feature, and for a
  surface this small it costs well under a hundred lines. Rejected alternative:
  ship Anthropic only and generalise later, which is how the generalisation never
  happens honestly.
- **Error mapping belongs in the adapter, not the caller.** The request shape is
  the obvious difference and the least dangerous one. What actually couples code
  to a provider is the failure path: what counts as retryable, how a rate limit
  arrives, how context overflow surfaces. A Protocol covering only the happy path
  pushes provider-specific `except` branches up into every caller, which is the
  same coupling one layer higher.
- **`httpx`, not the vendor SDKs.** The needed surface is tiny; the SDKs carry
  their own retry and timeout behaviour that would compete with ours; and a
  library user should not have to install two vendor SDKs to get a compactor.
  Rejected alternative: the official SDKs — more convenient for streaming, but
  streaming in the proxy is pass-through, a different problem, and it should not
  drive this choice.
- **Stop reasons are normalised, not passed through.** This is the translation
  easiest to skip, because it sits on the success path: the HTTP call returned
  200 and the text field looks reasonable. But a summary that stopped at the
  output cap has silently dropped the tail of the run it was summarising, and
  nothing downstream can tell. Passing the provider's raw string through would
  push exactly the coupling the error mapping removes back into every caller,
  wearing a success status code. The enum keeps an unrecognised value as
  `OTHER` rather than mapping it onto `COMPLETE`, because the tempting default
  asserts something we cannot support.
- **A fake, not mocks of the clients.** One fake implementing the Protocol keeps
  tests honest about the interface. Mocking each client would let the suite pass
  against a shape no real provider has.

## Consequences

Every provider-specific quirk has exactly one home, and adding a third provider
is one new file plus its error mapping.

The fake is now load-bearing: it is what every unit test runs against, so a fake
that is more forgiving than reality — never timing out, never returning a
malformed body, never reporting usage differently — will let a green suite hide a
broken client. Its failure behaviour has to be modelled as deliberately as its
success behaviour.
