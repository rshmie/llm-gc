# Events Bus — Overview

## Why This Exists

LLM-GC's central promise to the developer is *visibility*: the system must
surface what is happening to the context — what was collected, summarised,
archived, and replaced — in a form the developer can inspect. That visibility is not
a feature bolted onto the engine; it is the engine's reason for being. The
health monitor consumes one underlying stream of "what just happened" events
today, and the dashboard reads the monitor; a logger and the benchmark harness
are planned consumers of the same stream.

The events bus is that stream: the single cross-cutting channel through which
every component reports its work, without knowing or caring who is listening.

## How It Works

A single in-process publish/subscribe `EventBus` carries every observability
event:

- **Producers** (the GC service, which runs every collection, and the
  compactors) publish a typed `Event` as a side effect of doing their work.
  Emitting is fire-and-forget — a producer never waits on a consumer.
- An `Event` carries an `EventType` (which channel), a `data` payload, and a
  timestamp.
- **Consumers** subscribe a callback to the `EventType`s they care about; the bus
  invokes each matching subscriber when an event is emitted.
- The bus knows nothing about any specific consumer. The dashboard, logger, and
  benchmarks are interchangeable — adding, removing, or disabling one touches no
  producer.

The defining discipline: a component's **event surface is decided at design
time**, when the component is built — not retrofitted when a consumer later needs
the data. This is what the project calls *instrument from the inside out*.

### What is emitted, and who listens

```
  PRODUCERS                    BUS                   SUBSCRIBERS
 ┌──────────────┐  emit, fire  ┌──────────┐         ┌────────────────────────┐
 │ GcService    │ ───────────► │ EventBus │ ──────► │ Health Monitor         │
 │ compactors   │  and forget  └──────────┘         │ logger        (planned)│
 └──────────────┘                                   │ benchmarks    (planned)│
                                                    └───────────┬────────────┘
                                                                ▼
                                                     the dashboard reads the monitor
```

| Event | Emitted by | When | Monitor listens |
|---|---|---|---|
| `gc_finished` | GcService | once per pass, whatever its outcome — completed, bypassed or failed | yes |
| `message_promoted_to_old_gen` | GcService | a new span made at the summarised level | yes |
| `message_archived` | GcService | a span made archived, or lowered to archived or out of view | yes |
| `knowledge_entry_added` | GcService | a fact filed in the permanent generation | no |
| `knowledge_entry_superseded` | GcService | a newer fact replaced older ones on the same topic | yes |
| `aging_degraded` | GcService | part of a collection fell short: a summary refused or failed, facts not read, a recap over its budget | yes |
| `message_compacted` | each compactor | every summary call, whether or not its summary was used | no |

The GcService emits only **after a collection commits**, and every event names its
session, so an event always describes a change that happened, to one conversation. A
pass discarded because its session closed announces nothing but its `gc_finished`.

## Why This Shape — Design Rationale

Two ways to deliver the visibility stream were on the table:

- **Producer-driven (chosen).** Every component emits structured events as it
  works, whether or not any consumer currently exists. More work up front — each
  component must define what it emits before consumers are designed — but every
  later consumer is built without touching the producers. Visibility is correct
  *by construction*: when a new view is needed, the events it needs are already
  there.
- **Consumer-driven — direct logging/metrics calls per component (rejected).**
  Faster initially and pays only for what you use, but its failure mode is silent:
  when a new visibility need appears, the call sites are scattered or absent, and
  adding them means touching every component the view depends on. Visibility ends
  up retrofitted, with predictable gaps — unacceptable for a project whose whole
  point is visibility.

The transport choice was equally deliberate:

- **In-process bus (chosen).** The events are in-process and
  synchronous-friendly; an in-process bus has zero operational footprint and is
  trivially testable.
- **External message broker — Redis, NATS, etc. (rejected).** Adds operational
  complexity the project does not need at its current scope. If a genuine
  distributed need ever emerges, an external broker can replace the in-process bus
  behind the same publish/subscribe shape.

## Consequences

- **A small, deliberate design tax.** Every new component must decide what events
  it emits, and every new behaviour must decide whether it warrants an event. That
  tax buys visibility that is correct by construction — a consumer that needs to
  react to a behaviour finds the event already exists, because the producer was
  instrumented before the consumer was designed.
- **Event payloads are an implicit contract.** A producer that changes a payload
  can break a consumer silently. Payload discipline on `Event.data` (typed
  payloads or schema-versioned events) is a real obligation that grows with the
  number of consumers — the more views depend on the stream, the more a careless
  payload change can quietly break.
