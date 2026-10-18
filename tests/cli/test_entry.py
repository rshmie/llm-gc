"""`llm-gc`'s entry point: what someone who installed only the library sees."""

import builtins
import re
import tomllib
from pathlib import Path

import pytest

from llm_gc.cli.entry import PROXY_EXTRA_MODULES, run

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def fail_to_import_main(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Make `from llm_gc.cli.main import main` fail the way it does when `missing` is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llm_gc.cli.main":
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize("module", sorted(PROXY_EXTRA_MODULES))
def test_a_missing_proxy_package_is_named_with_the_fix(monkeypatch, module):
    fail_to_import_main(monkeypatch, module)

    with pytest.raises(SystemExit) as stopped:
        run()

    message = str(stopped.value)
    assert f"'{module}'" in message
    assert "pip install 'llm-gc[proxy]'" in message
    assert "uv sync --extra proxy" in message


def test_a_missing_part_of_a_proxy_package_names_the_package(monkeypatch):
    fail_to_import_main(monkeypatch, "fastapi.responses")

    with pytest.raises(SystemExit) as stopped:
        run()

    assert "'fastapi'" in str(stopped.value)


def test_any_other_missing_package_is_raised_as_it_is(monkeypatch):
    fail_to_import_main(monkeypatch, "requests")

    with pytest.raises(ModuleNotFoundError, match="requests"):
        run()


def test_the_command_starts_when_the_extra_is_installed(monkeypatch):
    import llm_gc.cli.main as cli_main

    started = []
    monkeypatch.setattr(cli_main, "main", lambda: started.append(True))

    run()

    assert started == [True]


def test_the_package_list_matches_the_proxy_extra():
    """A package added to the extra and not to the list would crash with a bare ImportError."""
    requirements = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]["proxy"]

    names = {re.split(r"[<>=!~\[ ;]", requirement, maxsplit=1)[0].replace("-", "_") for requirement in requirements}

    assert names == PROXY_EXTRA_MODULES
