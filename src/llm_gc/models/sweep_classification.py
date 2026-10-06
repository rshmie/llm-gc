from enum import Enum


class SweepClassification(Enum):
    """The detail level the next collection would give a verbatim turn, in the dashboard's words.

    KEEP - its span would be quoted word for word.
    COMPACT - its span would be summarised.
    ARCHIVE - its span would be archived to its opening line and facts, or folded out of view.
    """
    KEEP = "keep"
    COMPACT = "compact"
    ARCHIVE = "archive"
