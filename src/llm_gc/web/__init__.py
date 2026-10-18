"""What LLM-GC's two HTTP servers, the proxy and the dashboard, share. Application code: the library never imports it.

Its own package so the dashboard can use it without importing the proxy, which it is
built to run without.
"""
