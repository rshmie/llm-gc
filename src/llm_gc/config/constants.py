""" Application's default configuration values as constants """

from llm_gc import __version__

APP_VERSION = __version__

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_KEEP_THRESHOLD = 0.7
DEFAULT_ARCHIVE_THRESHOLD = 0.3
DEFAULT_MIN_COMPACTABLE_TOKENS = 30
DEFAULT_LAST_N_TURNS_TO_KEEP = 5
# Smallest batch of compactable turns worth a cut. Every cut resets the provider's
# prompt cache once and strips thinking blocks from the turns it keeps, so cutting
# for a few tokens costs more than it saves. A draft value, to be tuned by watching
# real conversations.
DEFAULT_MIN_CUT_TOKENS = 1000
DEFAULT_MAX_RECENT_TRANSITIONS = 20
DEFAULT_MAX_RECENT_DEGRADATIONS = 20
DEFAULT_MAX_RECENT_PASSES = 50    # one pass runs after every request, so most are quiet
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

# Session lifecycle timings (rationale in doc/design/session/overview.md).
DEFAULT_SESSION_TIMEOUT_S = 1800          # close a session idle this many seconds (30 min)
DEFAULT_SESSION_SWEEP_INTERVAL_S = 300    # how often the idle sweeper runs (5 min)

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
# LLM compaction (rationale in doc/design/sweeper-and-composition/compaction-mechanism.md).
DEFAULT_COMPACTION_TIMEOUT_S = 30.0       # wall-clock budget for one summarisation call
DEFAULT_COMPACTION_TARGET_RATIO = 0.35    # output cap as a fraction of the run's own tokens
DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS = 64 # floor, so a short run still has room for one sentence
# Room for reasoning on top of the summary target. Models that think by default
# count thinking against the output cap, so a cap sized for the summary alone is
# spent before the summary starts and every summary comes back truncated.
DEFAULT_COMPACTION_THINKING_ALLOWANCE_TOKENS = 2_000
