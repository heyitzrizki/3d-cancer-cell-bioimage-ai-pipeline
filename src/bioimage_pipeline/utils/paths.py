"""Cross-platform path helpers."""

from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    """Expand user markers and return an absolute path without requiring existence."""
    return Path(path).expanduser().resolve(strict=False)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and its parents if needed, then return its absolute path."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[3]
