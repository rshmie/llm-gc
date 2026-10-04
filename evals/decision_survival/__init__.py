"""Does a fact the conversation established early survive LLM-GC's clean-ups?

Scripted conversations with known facts are run through the proxy, then asked
about those facts, and the answers are graded with GC and without it.
"""
