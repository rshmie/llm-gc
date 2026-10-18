"""What `pip install llm-gc` gives someone who does not install the `proxy` extra.

This test process has every package installed, so each check runs in a fresh interpreter in which the
extra's packages cannot be imported.
"""

import subprocess
import sys
from pathlib import Path

import llm_gc
from llm_gc.cli.entry import PROXY_EXTRA_MODULES

# fastapi brings starlette, so the extra does not name it, but it is just as absent without the extra.
UNINSTALLED = sorted(PROXY_EXTRA_MODULES | {"starlette"})

# The packages that may use the extra. Everything else under llm_gc is the library.
APPLICATION_PACKAGES = {"cli", "proxy", "visualizer", "web"}

LIBRARY_DIR = Path(llm_gc.__file__).parent


def list_library_modules() -> list[str]:
    """The dotted name of every library module, found from the files so none has to be imported."""
    names = []
    for path in sorted(LIBRARY_DIR.rglob("*.py")):
        parts = path.relative_to(LIBRARY_DIR.parent).with_suffix("").parts
        if parts[1] in APPLICATION_PACKAGES:
            continue
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def run_without_extra(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a new interpreter where the extra's packages raise ImportError."""
    block = f"import sys\nfor name in {UNINSTALLED!r}:\n    sys.modules[name] = None\n"
    return subprocess.run([sys.executable, "-I", "-c", block + code], capture_output=True, text=True, timeout=60)


def test_every_library_module_imports_without_the_extra():
    modules = list_library_modules()
    assert "llm_gc.engine.gc_service" in modules  # the listing found the library, not nothing
    code = f"import importlib\nfor name in {modules!r}:\n    importlib.import_module(name)\n"

    result = run_without_extra(code)

    assert result.returncode == 0, result.stderr


def test_the_command_says_which_extra_to_install_instead_of_a_traceback():
    result = run_without_extra("from llm_gc.cli.entry import run\nrun()\n")

    assert result.returncode == 1
    assert "pip install 'llm-gc[proxy]'" in result.stderr
    assert "Traceback" not in result.stderr
