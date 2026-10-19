# Glossary

Some of the words LLM-GC's docs and dashboard use, each in one line. How each thing works is in the linked doc; this page only says what the word means. Terms are grouped in the order you would meet them, so reading it top to bottom also works as a short introduction.

Where the code or the design docs use a different name for the same thing, it is given in brackets.

## The basics

| Term | Meaning | More |
|---|---|---|
| **Token** | The unit a model reads text in, and is billed by: roughly a short word, or part of a longer one. Each model counts the same text a little differently. | |
| **Context window** | The most tokens a model can read in one request. Every request sends the whole conversation again, so a long conversation fills it. | |
| **Provider** | The company serving the model's API, such as Anthropic or OpenAI. | [README](../README.md#get-started) |
| **Gateway** | A server a company runs in front of the provider, so its staff reach the model through one address. LLM-GC sits in front of the gateway. | [README](../README.md#gateway) |
| **Upstream** | Where LLM-GC forwards requests: the provider's API, or a gateway. | [configuration](design/configuration.md) |
| **Proxy** | LLM-GC's HTTP server. Your app sends its requests to it instead of to the provider, and it passes each one on to the upstream. | [proxy](design/proxy/overview.md) |
| **Engine** | The part of LLM-GC that decides what the model is sent: it groups, scores and condenses the conversation. The proxy runs it, and it can also be used as a Python library. | [overview](overview.md#how-the-system-is-structured) |

## The conversation

| Term | Meaning | More |
|---|---|---|
| **Turn** | One message in the conversation: a user message, a reply, or a tool result. One message you type to a coding agent can lead to many turns, since each tool call and each tool result is a turn of its own. | [generational memory](design/generational-memory/overview.md) |
| **Exchange** | Something the user said or asked, and everything that came back before they spoke again: replies, tool calls and their results. | [generational memory](design/generational-memory/overview.md) |
| **Tool round** | An assistant's tool calls and the results that answer them. LLM-GC never condenses part of one and sends the rest as written, because the provider refuses a tool result separated from its call. | [the cut](design/generational-memory/the-cut.md) |
| **Session** | One conversation as LLM-GC holds it: its turns and everything LLM-GC has made from them. Sessions live in memory: after a restart, each is rebuilt from what the client sends next. | [session](design/session/overview.md) |
| **Snapshot** | A session's state at one moment. It is never changed in place, only replaced by a new one, so a request can read it without waiting for the work going on in the background. | [session](design/session/overview.md) |

## How the conversation is held

LLM-GC keeps a conversation in three generations, an idea borrowed from garbage collectors, which treat new and long-lived objects differently.

| Term | Meaning | More |
|---|---|---|
| **Young generation** (*tail*) | The newest messages, sent to the model exactly as the client wrote them: at least the last 5, and more build up between collections. | [generational memory](design/generational-memory/overview.md) |
| **Old generation** | Everything older, held as spans. | [generational memory](design/generational-memory/overview.md) |
| **Permanent generation** | The facts read from every exchange, and what later replaced them. Never discarded. | [facts](design/generational-memory/knowledge-entry-model.md) |
| **Span** | An exchange once it has moved to the old generation. It keeps all its original turns, and the model is shown it at a detail level. | [generational memory](design/generational-memory/overview.md) |
| **Score** | A number from 0 to 1 for how much a span matters now. It is the larger of how recent the span is and how much the current discussion touches it, plus credit for the decisions and specifics (names, numbers, code) it states. Worked out again at every collection. | [scoring](design/scoring/overview.md) |
| **Detail level** | How much of a span the model reads, set by its score and by a limit on the recap's size. **Kept**: every word (shown as *Quoted exactly*). **Summarised**: its opening line, its facts and a summary. **Archived**: its opening line and its facts. **Out of view**: only one line saying a run of the oldest spans exists. | [detail levels](design/generational-memory/detail-levels.md) |
| **Recap** (*front*) | The one message at the start of the prompt that stands in for the older conversation, showing every span at its level. The newest messages follow it. | [the cut](design/generational-memory/the-cut.md) |
| **Fact** (*knowledge entry*) | A decision, value or preference read from a span's own turns and filed under a topic, such as "database: Postgres". The summary model reads them; without one, simple patterns do. A summarised or archived span lists its facts, so a decision stays in front of the model after its words are condensed. | [facts](design/generational-memory/knowledge-entry-model.md) |
| **Replaced** (*superseded*) | A fact that a later turn changed, on the same topic. It is kept, and its span carries a note saying it was replaced, so the model does not read it as current. | [facts](design/generational-memory/knowledge-entry-model.md) |

## When the prompt changes

| Term | Meaning | More |
|---|---|---|
| **Collection** (*cut*) | One change to what the model is sent: turns that are no longer among the newest messages become spans, and every span's level is decided again. It is prepared in the background, then switched in. | [the cut](design/generational-memory/the-cut.md) |
| **Pass** | The background check after each reply: are there enough older turns, about 1,000 tokens by default, to be worth preparing a collection? Most passes change nothing. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Prepared** or **staged** collection | A collection whose summaries and facts are ready, held in the session until it is switched in. Preparing it costs LLM-GC's own model calls and changes nothing the model is sent. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Prompt cache** (*warm*, *cold*) | The provider remembers a prompt it has just read, for a few minutes, and charges much less to read it again. **Warm**: it still remembers, and the start of the prompt is unchanged. **Cold**: it has forgotten, so the whole prompt is paid for in full again. Changing the start of the prompt costs the same as a cold cache, even when the cache was warm. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Stale weight** | The tokens the model is still sent that a staged collection would condense. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Switch in** | The moment a staged collection becomes the recap. A request does it, with no model call, when the cache is cold anyway, when the saving repays caching the prompt again, when stale weight reaches 10% of the context window (by default), or when the developer presses Run GC now. | [when to collect](design/generational-memory/when-to-collect.md) |
| **Hot path**, **cold path** | The hot path is what happens while a request is on its way to the model, and it never waits for a model call. The cold path is the background work after each reply, where LLM-GC's own model calls happen. | [overview](overview.md#how-the-system-is-structured) |

## LLM-GC's own work

| Term | Meaning | More |
|---|---|---|
| **Summary model** | The model LLM-GC itself calls to write summaries and read facts, with its own key. It is separate from the model you talk to. | [LLM provider](design/llm-provider.md) |
| **Compactor** | The part that writes a span's summary. The `noop` compactor makes no model call, so spans are kept or archived but never summarised. | [compaction](design/generational-memory/compaction.md) |
| **Degradation** | Part of a collection that fell short of its plan, such as a summary that failed, so its span was kept whole instead. Nothing is deleted, and the dashboard shows it. | [visibility](design/visibility/overview.md) |

## Seeing what happened

| Term | Meaning | More |
|---|---|---|
| **Dashboard** | The page that shows, for each conversation, what the model still receives, every fact it is told, and what LLM-GC did. It runs with `llm-gc start --dashboard`. | [visibility](design/visibility/overview.md) |
| **Event bus** | The channel through which each part of LLM-GC reports what it did. Events carry counts and positions, never conversation text. | [events](design/events/overview.md) |
| **Health monitor** | Follows each conversation's events and keeps a record of what LLM-GC has done to it, for the dashboard. | [visibility](design/visibility/overview.md) |
| **Decision-survival check** | The evaluation that measures whether condensing loses facts: scripted conversations, and questions about facts they settled, asked once through LLM-GC and once with the whole conversation sent. | [the check](../evals/decision_survival/README.md) |
