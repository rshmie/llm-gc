""" Application's default configuration values as constants """

from llm_gc import __version__

APP_NAME = "llm-gc"
APP_VERSION = __version__

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_GC_THRESHOLD = 0.7
DEFAULT_SWEEP_STRATEGY = "threshold"
DEFAULT_KEEP_THRESHOLD = 0.7
DEFAULT_ARCHIVE_THRESHOLD = 0.3
DEFAULT_MIN_COMPACTABLE_TOKENS = 30
DEFAULT_LAST_N_TURNS_TO_KEEP = 5
DEFAULT_MAX_RECENT_TRANSITIONS = 20

# Permanent-generation injection. The cap is what makes the system stable, not a
# nicety: archiving produces facts, injected facts are real tokens, and more
# pressure causes more archiving. Bounded, that loop converges; unbounded, memory
# grows with the session and undoes the saving archiving produced.
DEFAULT_MAX_MEMORY_TOKENS = 200   # whole injected block, header included
DEFAULT_MAX_MEMORY_FACTS = 8      # retrieved before the token budget is applied
# How long a coroutine waits for a session's lock before giving up.
# On timeout the caller proceeds on last-persisted (possibly stale) state rather
# than hang forever — a bounded wait keeps a stuck update from freezing the
# session's live path.
DEFAULT_SESSION_LOCK_TIMEOUT_MS = 100

# Session lifecycle timings (rationale in doc/design/session/overview.md).
DEFAULT_SESSION_TIMEOUT_S = 1800          # close a session idle this many seconds (30 min)
DEFAULT_SESSION_SWEEP_INTERVAL_S = 300    # how often the idle sweeper runs (5 min)
DEFAULT_SESSION_CLOSE_DRAIN_MS = 5000     # max wait for an in-flight op before force-closing

# Port for the context-health dashboard (spec section: Dashboard). The proxy itself will use 9900.
DEFAULT_DASHBOARD_PORT = 9901

# A regex pattern to match capital words, numbers including decimals, code fragments, and URLs for token density scoring
DENSITY_SCORER_PATTERN = r'\b[A-Z][a-zA-Z]+\b|\b\d+\.?\d*\b|`[^`]+`|https?://\S+'

# A regex pattern to identify decision-making language in messages which includes common phrases indicating explicit decisions, conclusions/resolution, commitments
DECISION_SCORER_PATTERN = (r'\b(?:keep|delete|summarize|uncertain|concluded|chose|settled on|we agreed|go with|the plan is|'
                           r'let\'s go with|let\'s use|we\'ll go with|we\'ll use|decided to|decision is|going to use|going with|'
                           r'the answer is|concluded that|settled on|final choice|in conclusion|to summarize|the solution is|'
                           r'I will use|we should use|stick with|switching to)\b')

# A regex pattern to identify reference language in messages which includes common phrases indicating reference to previous points, reiteration, or emphasis on earlier statements
REFERENCE_SCORER_PATTERN = (r'\b(?:as I mentioned|as you said|like you said|as we discussed|like we discussed|as I said|like I said|as mentioned|as noted|as we said|as we mentioned|as we noted|'
                            r'earlier|previously|before|going back to|back to your point|back to what you said|referring back to|'
                            r'to reiterate|to repeat|as noted|referring to|regarding what)\b')

def get_default_config() -> dict[str, str | int | float]:
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "model": DEFAULT_MODEL,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "gc_threshold": DEFAULT_GC_THRESHOLD,
        "sweep_strategy": DEFAULT_SWEEP_STRATEGY,
        "keep_threshold": DEFAULT_KEEP_THRESHOLD,
        "min_compactable_tokens": DEFAULT_MIN_COMPACTABLE_TOKENS,
        "last_n_turns_to_keep": DEFAULT_LAST_N_TURNS_TO_KEEP,
    }
# LLM compaction (rationale in doc/design/sweeper-and-composition/compaction-mechanism.md).
DEFAULT_COMPACTION_TIMEOUT_S = 30.0       # wall-clock budget for one summarisation call
DEFAULT_COMPACTION_TARGET_RATIO = 0.35    # output cap as a fraction of the run's own tokens
DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS = 64 # floor, so a short run still has room for one sentence
