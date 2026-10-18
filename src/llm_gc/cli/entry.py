"""The `llm-gc` command's entry point. Application code: the library never imports it.

The command needs the packages in the `proxy` extra: the web server, click and the settings loader.
Someone who installed only the library does not have them, and importing `llm_gc.cli.main` would then
fail with a bare "No module named 'click'". This module imports nothing from the extra, so it can run
first and say what to install.
"""

import sys

# The top-level packages the `proxy` extra in pyproject.toml installs. A test keeps this list equal to
# that extra, so a package added there cannot be missed here.
PROXY_EXTRA_MODULES = frozenset({"click", "fastapi", "pydantic_settings", "uvicorn"})

MISSING_EXTRA_MESSAGE = (
    "llm-gc: the command needs the 'proxy' extra, and the package '{module}' from it is not installed.\n"
    "Install it with: pip install 'llm-gc[proxy]'\n"
    "From a source checkout: uv sync --extra proxy"
)


def run() -> None:
    """Start the command line, or say which extra to install if it cannot start.

    Only a missing package from the `proxy` extra gets the message. Any other failed import is a bug
    in LLM-GC or a broken install, and it is raised as it is.
    """
    try:
        from llm_gc.cli.main import main
    except ModuleNotFoundError as err:
        missing = (err.name or "").partition(".")[0]
        if missing not in PROXY_EXTRA_MODULES:
            raise
        sys.exit(MISSING_EXTRA_MESSAGE.format(module=missing))
    main()
