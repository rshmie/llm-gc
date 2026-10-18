"""Application's default configuration values as constants"""

from importlib.metadata import version

# Read from the installed package, so pyproject.toml is the one place the version is written.
APP_VERSION = version("llm-gc")

DEFAULT_CONTEXT_WINDOW = 128_000
# The proxy's window when none is configured: the smallest of the current models'. Too
# small costs some compression; too large lets a recap outgrow the model's window and
# fail requests. A larger model is an explicit opt-in.
DEFAULT_PROXY_CONTEXT_WINDOW = 200_000
DEFAULT_KEEP_THRESHOLD = 0.7
DEFAULT_ARCHIVE_THRESHOLD = 0.3
DEFAULT_LAST_N_MESSAGES_TO_KEEP = 5
# Smallest batch of compactable turns worth a cut. Every cut resets the provider's
# prompt cache once and strips thinking blocks from the turns it keeps, so cutting
# for a few tokens costs more than it saves. A draft value, to be tuned by watching
# real conversations.
DEFAULT_MIN_CUT_TOKENS = 1000
# When the provider's cache is warm, a staged cut is switched in only once it removes at
# least this many times what the switch makes the provider cache again. 1.0: at least as
# much as it rewrites. Zero switches as soon as there is a saving. A draft value.
DEFAULT_MIN_SAVING_SHARE = 1.0
# The most stale weight - turns a staged cut condenses that the model is still sent word for
# word - as a share of the model's window. Reaching it switches the cut in whatever the cache
# costs, for the model's sake. A draft value; it is the dial between grounding and the bill.
DEFAULT_MAX_STALE_RATIO = 0.10
DEFAULT_MAX_RECENT_TRANSITIONS = 20
DEFAULT_MAX_RECENT_DEGRADATIONS = 20
DEFAULT_MAX_RECENT_PASSES = 50  # one pass runs after every request, so most are quiet
# Sessions the health monitor keeps history for. Above this, the one heard from
# least recently is dropped; it bounds memory if closed sessions are never forgotten.
DEFAULT_MONITOR_MAX_SESSIONS = 1000

# The recap's one budget: everything the front shows for the turns before the tail,
# at whatever level each span is shown. The cap is what makes the front bounded:
# over it, the coldest spans lose detail first. A *fraction* of the window, never an
# absolute: 200 tokens is nothing to a 128k window and all of a 200-token one, and
# measured on a 200-token window an absolute memory budget of 100 made a proxy
# forward 21% more than the client sent. Generous on purpose - the score decides
# what is condensed; this only bounds the total.co
DEFAULT_MAX_RECAP_RATIO = 0.3

# Hysteresis around each detail-level band: a span moves up only once its score
# clears a band by this much, and down only once it falls this far below one, so a
# score hovering at a line does not show the model a span in full at one collection
# and only its facts at the next.
DEFAULT_LEVEL_MARGIN = 0.05

# Session lifecycle (rationale in doc/design/session/overview.md). A person's break is not the end of a
# conversation, so idle sessions are held for a day; the cap on how many are held bounds the memory.
DEFAULT_SESSION_TIMEOUT_S = 86_400  # close a session idle this many seconds (24 h)
DEFAULT_SESSION_SWEEP_INTERVAL_S = 300  # how often the idle sweeper runs (5 min)
DEFAULT_MAX_SESSIONS = 64  # held at once; past it the least recently active is closed

# Port for the context-health dashboard (spec section: Dashboard). The proxy itself will use 9900.
DEFAULT_DASHBOARD_PORT = 9901

# A regex pattern to match capital words, numbers including decimals, code fragments, and URLs for token density scoring
DENSITY_SCORER_PATTERN = r"\b[A-Z][a-zA-Z]+\b|\b\d+\.?\d*\b|`[^`]+`|https?://\S+"

# A regex pattern to identify decision-making language in messages which includes common phrases indicating explicit decisions, conclusions/resolution, commitments
DECISION_SCORER_PATTERN = (
    r"\b(?:keep|delete|summarize|uncertain|concluded|chose|settled on|we agreed|go with|the plan is|"
    r"let\'s go with|let\'s use|we\'ll go with|we\'ll use|decided to|decision is|going to use|going with|"
    r"the answer is|concluded that|settled on|final choice|in conclusion|to summarize|the solution is|"
    r"I will use|we should use|stick with|switching to)\b"
)

# LLM compaction (rationale in doc/design/generational-memory/compaction.md).
DEFAULT_COMPACTION_TIMEOUT_S = 30.0  # wall-clock budget for one summarisation call
DEFAULT_COMPACTION_TARGET_RATIO = 0.35  # output cap as a fraction of the run's own tokens
DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS = 64  # floor, so a short run still has room for one sentence
# Room for reasoning on top of the summary target. Models that think by default
# count thinking against the output cap, so a cap sized for the summary alone is
# spent before the summary starts and every summary comes back truncated.
DEFAULT_COMPACTION_THINKING_ALLOWANCE_TOKENS = 2_000

# Facts read by a model (rationale in doc/design/generational-memory/overview.md).
DEFAULT_FACTS_OUTPUT_TOKENS = 1_500  # room for the list itself; a collection settles a handful of facts
# Room to think before the list. An agent's turns carry files and command output, and
# a 5,000-token Claude Code collection spent all of a 2,000-token allowance thinking.
# Billed only as used, while an answer cut off before it starts is paid for in full.
DEFAULT_FACTS_THINKING_ALLOWANCE_TOKENS = 12_000
DEFAULT_FACTS_TIMEOUT_S = 150.0  # long enough to reach the cap; off the request path
DEFAULT_FACTS_MAX_BATCH_TOKENS = 8_000  # turns read per call, one span at most: thinking grows with what is read
DEFAULT_FACTS_ON_RECORD_MAX = 200  # facts on record shown to the reader, the newest kept
FACT_TOPIC_MAX_WORDS = 5  # "seat hold", "checkout latency at p99"; longer is a sentence
FACT_TOPIC_MAX_CHARS = 60
FACT_VALUE_MAX_CHARS = 200  # a value, not a paragraph
