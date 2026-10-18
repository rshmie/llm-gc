# What data LLM-GC keeps

LLM-GC sees every conversation you have through it, so you should know what it holds, where, for how long, and how to remove it.

**The short answer: LLM-GC writes your conversations nowhere.** It keeps them in the memory of the running `llm-gc` process. When the process stops, they are gone.

```
 Claude Code ──────► LLM-GC ──────────────► Anthropic, or your gateway
 (keeps its own        │  holds conversations   the conversation, with older turns
  history on disk)     │  in memory only        replaced by LLM-GC's recap
                       │
                       ├──────────────────────► LLM-GC's summary model
                       │                        quotes of older turns, to summarise them
                       │                        and read their facts; with LLM-GC's own key
                       │
                       └──► dashboard           in your browser, on this machine only;
                            (off by default)    it reads LLM-GC's memory
```

## What is held in memory

| What | Holds your words? | How long | Limit |
|---|---|---|---|
| **Each conversation's state:** the turns word for word, the summaries LLM-GC wrote, the facts it read | Yes | Until the conversation has been idle for 24 hours (checked every 5 minutes), is pushed out by newer ones, or is deleted | 64 conversations. When a 65th arrives, the least recently active closes |
| **The dashboard's copy of what was said**, only while the dashboard is on | Yes | Until its conversation closes | 50 conversations |
| **A subagent's prompt and first message**, held so LLM-GC can tell which tool call started the subagent | Yes | Normally a few seconds, until the subagent's first request arrives. One that is never matched stays until a later request from the same Claude Code session clears it, which is after 10 minutes at the earliest, or until the process stops | 64 per Claude Code session, 256 sessions |
| **The request log, running totals, and a few small ledgers** (cache state, thinking sizes, the health monitor): ids, timestamps, model names, token counts, which turn was kept | No | Until the process stops | Bounded: 10,000 requests, 1,024 conversations, 256 Claude Code sessions |
| **Token counts and importance scores**, remembered under a hash of the text so the same text is not measured twice | No. A hash cannot be turned back into text, though it can confirm a guess of it | Until the process stops | 50,000 counts, 20,000 scores |

Two limits are fixed in this version and cannot be changed in the settings: the 24 hours and the 64 conversations.

## What is on disk

**No conversation text is written to disk by LLM-GC.** What is on disk is:

- `llm-gc.toml`, the settings you write with `llm-gc config init`. It holds no secrets.
- `.env`, if you create one. It holds LLM-GC's own API key, in plain text, as you wrote it. A key exported in your shell is not on disk. Keep `.env` out of version control; the repository's `.gitignore` already does.
- The tokenizer's vocabulary, a public file with no connection to you. The first time LLM-GC counts tokens, its tokenizer library downloads the file and keeps it in your system's temporary folder, under `data-gym-cache`. Set `TIKTOKEN_CACHE_DIR` to put it elsewhere.
- Log lines, if you send LLM-GC's output to a file. They name conversations by id and carry model names and token counts. **They never contain conversation text, and never request headers**, so a log file holds neither your words nor your keys.

## What leaves this machine

- **To Anthropic, or your gateway: your requests.** This is the same conversation, to the same place, as without LLM-GC, except that once a collection is applied, older turns are replaced by a recap. Your login and any API key Claude Code sends are passed on and never kept.
- **To LLM-GC's summary model: quotes of older turns.** To write a recap and read the facts in a conversation, LLM-GC sends the older turns to the model set under `[summary]` in `llm-gc.toml`, with LLM-GC's own key. That can be a different account, or a different provider, from the one you talk to. What that provider keeps is governed by its own policy, not by LLM-GC.
- **Small checks that carry no conversation.** At startup, a one-token call to the summary model checks that its key works. LLM-GC also asks Anthropic, or your gateway, which models it lists, to learn each model's context window. And once, a request for the tokenizer's public vocabulary file.

LLM-GC sends nothing anywhere else. It has no telemetry and no analytics.

## Deleting a conversation

**One conversation.** This needs the dashboard on (`llm-gc start --dashboard`), because the delete is a dashboard action. List what LLM-GC holds, then delete one by its id:

```bash
curl -s http://127.0.0.1:9901/api/overview | python3 -c 'import json,sys; [print(s["session_id"], s["title"]) for s in json.load(sys.stdin)["sessions"]]'
curl -X DELETE -H "x-llm-gc-dashboard: 1" http://127.0.0.1:9901/api/sessions/<id>
```

A delete answers `204` when it worked, and `404` if there is no such conversation. The header is required so that a web page open in your browser cannot make this request for you.

**What a delete removes:** the conversation's turns, summaries and facts, and the dashboard's copy of what was said. It is the same close that happens when a conversation has been idle for 24 hours, asked for now.

**What it leaves:** the counts and ids in the request log and the totals, which hold none of your words. A subagent's prompt that was held for a link, if one was waiting. They go when the process stops.

**A delete forgets; it does not forbid.** Claude Code still has the conversation, and sends all of it on its next request. LLM-GC then builds the conversation again from that, under a new id, as it does after a restart. So delete conversations that are over. To end one that is still going, end it in Claude Code.

**Everything at once.** Stop LLM-GC (Ctrl-C). Everything it held in memory goes with the process. Nothing else needs cleaning up, except `.env` if you also want its key gone.

## What deleting here does not reach

- **Claude Code's own history.** Claude Code keeps its conversations on your disk, and LLM-GC never reads or changes those files.
- **What the providers keep.** Anthropic, your gateway and the summary model's provider each have their own retention. LLM-GC cannot delete from them.

See also: [SECURITY.md](../SECURITY.md) for the promises LLM-GC makes about your data, and the [visibility design](design/visibility/overview.md) for why the dashboard is opt-in and local.
