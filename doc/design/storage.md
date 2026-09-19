# Storage — Schema Versioning

> **Not yet built.** This doc records the intended design for the storage layer
> so the decision is settled before the code exists. It describes what the
> layer will do, not what it does today.

## How it works

Two kinds of LLM-GC data outlive the running process: `KnowledgeEntry` records
in the permanent generation, and compacted summaries in the old generation.
Every persisted record carries an explicit integer `schema_version`.

The migration contract:

- **On read** — a record older than the running code's expected version is
  upgraded in-process before it reaches the caller. Migration logic lives in a
  versioned table, one function per `(v -> v+1)` step, applied in sequence.
- **On write** — records are always stamped with the current version. A newer
  binary never writes an older record.
- **Forward incompatibility fails loudly** — a record *newer* than the running
  code understands causes the read to fail with a clear error. The system never
  silently acts on data whose semantics it does not fully know.
- **One-time backup** — the first time a store is opened by a newer binary, a
  copy is taken at `<store_path>.v<old_version>.bak` before any in-place upgrade
  is written. Recovery from a bad migration is a file copy, not an
  export-and-import.

Schema versions are integers and increase monotonically. Every schema change
carries its forward and reverse migration path and a test that exercises both.

## Why it exists

Stored data is the one place where a mistake destroys user state rather than
just producing a bad answer. Without a defined versioning approach, three
failure modes appear: a newer binary crashes on a missing field and wipes
accumulated session knowledge; a newer binary reads an old record as though it
had new semantics, producing results that are wrong but plausible; or the schema
is never changed at all, and workarounds accumulate until the code is worse than
a clean evolution would have been.

Event payloads have a similar versioning concern, but events are ephemeral — a
producer/consumer mismatch shows up at runtime and gets fixed. Stored data is
not ephemeral.

The retention commitment is the deeper reason: decisions preserved earlier in a
session must survive into the rest of that session. That promise means little if
"the rest of the session" excludes "after the next upgrade".

## Why this shape

- **Integer versions, not strings.** `"v1.0"` vs `"v1.1"` forces the code to
  know which string is greater — either by parsing them as numbers, in which
  case they should have been numbers, or by maintaining a hand-written ordering,
  which is brittle. Rejected alternative: semantic-version strings.
- **In-process migration on read, not an external migration tool.** External
  tooling moves the version-aware code outside the binary, which is
  operationally heavier than this scale needs. In-process upgrade-on-read is the
  simplest design that delivers the contract; an offline migration path can be
  added later for deployments that need one.
- **Loud failure on unknown-newer records, not best-effort field skipping.** A
  binary that skips fields it does not recognise might silently ignore a new
  flag the record depends on — say an `is_critical` marker — and degrade
  behaviour without saying so. Refusing to act on data it does not fully
  understand is the honest option. Rejected alternative: tolerate and skip.
- **No versioning at all** was rejected outright: an upgrade that destroys
  retained knowledge is the loudest possible violation of the retention
  commitment.

## Consequences

Every schema change costs real work — the migration, the test, the review of the
backwards path. That tax is the price of not destroying user state, and the
alternative ("never change the schema") costs more over the project's life.

The one-time backup uses disk proportional to store size. That is trivial for a
SQLite store and wrong for a large external one; backends like qdrant or
postgres have their own backup mechanisms that should be preferred. The
file-copy fallback is the default, not the recommendation for production.

`schema_version` is part of a `KnowledgeEntry`'s public shape and is surfaced by
the visibility layer, so a user inspecting an old entry can see which version of
the system wrote it. Schema evolution stays visible rather than buried.
