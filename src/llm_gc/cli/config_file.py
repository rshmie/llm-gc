"""The starter config file: reading the packaged template, and writing it out without clobbering anything."""

from importlib import resources
from pathlib import Path

from llm_gc.exceptions import ConfigError

_TEMPLATE_NAME = "config_template.toml"


def read_config_template() -> str:
    """The commented starter file, from inside the package so it exists wherever LLM-GC is installed."""
    return resources.files("llm_gc.cli").joinpath(_TEMPLATE_NAME).read_text(encoding="utf-8")


def write_config_file(path: Path, text: str, *, overwrite: bool = False) -> None:
    """Write `text` to `path`, creating its directory, and refuse to replace an existing file unless told to.

    Raises:
        ConfigError: The file exists and `overwrite` is false.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mode "x" creates the file or fails, in one step. Checking `path.exists()` first would leave
    # a gap in which another process could create it, and this would then overwrite theirs.
    try:
        with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as err:
        raise ConfigError(f"{path} already exists; pass --force to replace it") from err
