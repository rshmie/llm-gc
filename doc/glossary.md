# Glossary

Some of the words LLM-GC's docs and dashboard use, each in one line. How each thing works is in the linked doc; this page only says what the word means. Terms are grouped in the order you would meet them, so reading it top to bottom also works as a short introduction.

Where the code or the design docs use a different name for the same thing, it is given in brackets.

## The conversation

| Term | Meaning | More |
|---|---|---|
| **Turn** | One message in the conversation: a user message, a reply, or a tool result. One prompt to a coding agent is many turns. | [generational memory](design/generational-memory/overview.md) |
| **Exchange** | Something the user said or asked, and everything that came back before they spoke again: replies, tool calls and their results. | [generational memory](design/generational-memory/overview.md) |
| **Tool round** | An assistant's tool calls and the results that answer them. LLM-GC never splits one, because the provider refuses a tool result cut off from its call. | [the cut](design/generational-memory/the-cut.md) |
| **Session** | One conversation as LLM-GC holds it: its turns, spans and facts. Sessions live in memory: after a restart, each is rebuilt from what the client sends next. | [session](design/session/overview.md) |
| **Snapshot** | A session's state at one moment. It is never changed in place, only replaced, so a request can read it without waiting for the background work. | [session](design/session/overview.md) |

## How the conversation is held

| Term | Meaning | More |
|---|---|---|
| **Young generation** | The newest messages, sent to the model exactly as the client wrote them. | [generational memory](design/generational-memory/overview.md) |
| **Old generation** | The older exchanges, each held as a span. | [generational memory](design/generational-memory/overview.md) |
| **Permanent generation** | Every fact read from the spans, with what later replaced it. Never discarded. | [facts](design/generational-memory/knowledge-entry-model.md) |
| **Span** | One exchange in the old generation. It keeps all its original turns, and the model is shown it at a detail level. | [generational memory](design/generational-memory/overview.md) |
| **Score** | A number from 0 to 1 for how much a span matters now: how recent it is or how much the current discussion touches it, whichever is larger, plus the decisions and specifics it states. Worked out again at every collection. | [scoring](design/scoring/overview.md) |
| **Detail level** | How much of a span the model reads. **Kept**: every word. **Summarised**: its opening line, its facts and a summary. **Archived**: its opening line and facts. **Out of view**: folded with the oldest spans behind one line. | [detail levels](design/generational-memory/detail-levels.md) |
| **Recap** (*front*) | The one message at the start of the prompt that shows every span at its level. The newest messages follow it. | [the cut](design/generational-memory/the-cut.md) |
| **Fact** (*knowledge entry*) | A decision, value or preference read from a span's own turns and filed under a topic, such as "database: Postgres". | [facts](design/generational-memory/knowledge-entry-model.md) |
| **Replaced** (*superseded*) | A fact a later turn changed on the same topic. It is kept, and its span carries a note saying it was replaced, so the model does not read it as current. | [facts](design/generational-memory/knowledge-entry-model.md) |

## When the prompt changes

| Term | Meaning | More |
|---|---|---|
| **Collection** (*cut*) | One change to what the model is sent: new spans are made from turns that have aged out, and every span's level is planned again. | [the cut](design/generational-memory/the-cut.md) |
| **Pass** | The background check after each reply: has enough gone quiet to be worth a collection? Most passes change nothing. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Prepared** or **staged** collection | A collection whose summaries and facts are ready, held in the session until it is switched in. Preparing it costs LLM-GC's own model calls and changes nothing the model is sent. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Switch in** | The moment a staged collection becomes the recap. A request does it, with no model call, when the cache is cold, when the saving repays caching the prompt again, when stale weight reaches its bound, or when the developer presses Run GC now. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Prompt cache** (*warm*, *cold*) | The provider remembers the prompt it has just read, for a few minutes. Warm: the start of the prompt is unchanged, so it is read cheaply. Cold: the memory has expired, so everything is read at full price. Switching a collection in changes the start, so the provider has to cache it again. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Stale weight** | The tokens the model is still sent that a staged collection would condense. Once it reaches 10% of the window, the collection is switched in whatever the cache costs. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Hot path**, **cold path** | The hot path builds and forwards each request and never waits for a model call. The cold path is the background work after each reply, where LLM-GC's own model calls happen. | [overview](overview.md#how-the-system-is-structured) |

## LLM-GC's own work

| Term | Meaning | More |
|---|---|---|
| **Summary model** | The model LLM-GC itself calls to write summaries and read facts, with its own key. It is separate from the model you talk to. | [LLM provider](design/llm-provider.md) |
| **Compactor** | The part that writes a span's summary. With `noop`, it makes no model call, so spans are kept or archived but never summarised. | [compaction](design/generational-memory/compaction.md) |
| **Degradation** | Part of a collection that fell short of its plan, such as a summary that failed, so its span was kept whole instead. Nothing is lost, and the dashboard shows it. | [visibility](design/visibility/overview.md) |

## Seeing what happened

| Term | Meaning | More |
|---|---|---|
| **Event bus** | The channel every component reports what it did through. Events carry counts and positions, never conversation text. | [events](design/events/overview.md) |
| **Health monitor** | Follows each conversation's events and keeps what LLM-GC has done to it, for the dashboard. | [visibility](design/visibility/overview.md) |
| **Dashboard** | The page that shows, for each conversation, what the model still receives, every fact it is told, and what LLM-GC did. | [visibility](design/visibility/overview.md) |
| **Decision-survival check** | The evaluation: scripted conversations, and questions about facts they settled, asked once through LLM-GC and once with the whole conversation sent. | [the check](../evals/decision_survival/README.md) |

## Running it

| Term | Meaning | More |
|---|---|---|
| **Provider** | The company serving the model's API, such as Anthropic or OpenAI. | [README](../README.md#get-started) |
| **Gateway** | A company's own address in front of the provider. LLM-GC sits in front of the gateway. | [README](../README.md#gateway) |
| **Upstream** | Where LLM-GC forwards requests: the provider's API, or a gateway. | [configuration](design/configuration.md) |
| **Proxy** | LLM-GC's HTTP server, between your app and the upstream. Your app sends its requests to it instead of to the provider. | [proxy](design/proxy/overview.md) |
