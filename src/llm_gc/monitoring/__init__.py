"""Context-health monitoring — the data source of the visibility layer.

Holds ``ContextHealthMonitor`` (a continuous, passive subscriber on the event
bus) and the ``ContextHealth`` snapshot it exposes. This package has no web
dependencies; the dashboard that renders these signals lives in
``llm_gc.visualizer``.

See ``doc/visibility/overview.md`` for the design and its rationale.
"""
