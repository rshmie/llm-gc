# LLM-GC Documentation

Start with **[overview.md](overview.md)** — what the problem is, why cleanup
demands transparency, and the principles the system is held to.

## Which doc answers which question

| Question | Doc |
|---|---|
| What is this, and why does it exist? | [overview.md](overview.md) |
| How does one area work, and why is it shaped that way? | the matching doc under [`design/`](design/) |
| How do I install and use it? | the project [README](../README.md) |

## `design/` — how the system works

One living doc per area. Each covers **How it works**, **Why it exists**, and
**Why this shape** — including the alternatives that were rejected and the
reason they lost. These describe the system as it is; where an area is not built
yet, the doc says so at the top.

```
design/
  scoring/                    relevance scoring — overview + one doc per scorer
  sweeper-and-composition/    classification and compaction
  generational-memory/        young / old / permanent generations, and the cut
  session/                    per-conversation state and its concurrency contract
  proxy/                      the HTTP proxy: finding a conversation, forwarding, never degrading
  events/                     the event bus components report through
  visibility/                 what the model knows, what LLM-GC did, and the dashboard
  llm-provider.md             the model calls LLM-GC makes on its own behalf
  storage.md                  schema versioning for persisted records
  failure-handling.md         the circuit breaker around engine failures
```

## What is not here

Working notes — the build log, the phase plan, the target spec, the pre-build
research — live in `doc/internal/` and are not published. They are working
material, not documentation: some of it describes the system as intended rather
than as built, and publishing that would misrepresent what actually runs today.
