from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = PROJECT_ROOT.parent / "_github_upload"
EXPORT_DIR = EXPORT_ROOT / PROJECT_ROOT.name

EXCLUDED_DIRS = {
    ".cache",
    ".git",
    ".mplconfig",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data/processed",
    "data/raw",
    "outputs/models",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {".DS_Store"}


def should_exclude(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if path.name in EXCLUDED_NAMES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return relative in EXCLUDED_DIRS or any(relative.startswith(f"{name}/") for name in EXCLUDED_DIRS)


def main() -> None:
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_ROOT.mkdir(exist_ok=True)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        directory_path = Path(directory)
        for name in names:
            candidate = directory_path / name
            try:
                if should_exclude(candidate):
                    ignored.add(name)
            except ValueError:
                continue
        return ignored

    shutil.copytree(PROJECT_ROOT, EXPORT_DIR, ignore=ignore)
    archive = shutil.make_archive(str(EXPORT_DIR), "zip", EXPORT_ROOT, PROJECT_ROOT.name)
    print(f"Created clean upload folder: {EXPORT_DIR}")
    print(f"Created clean upload zip: {archive}")


if __name__ == "__main__":
    main()
