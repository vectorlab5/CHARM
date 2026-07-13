"""Fail fast on common source-release packaging mistakes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REQUIRED_PATHS = (
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "configs/base.yaml",
    "docs/DATA_FORMAT.md",
    "docs/REPRODUCIBILITY.md",
    "src/charm/cli.py",
    "tests",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".pyc"}
FORBIDDEN_TEXT = (
    re.compile("response" + r"\s+to\s+(the\s+)?" + "reviewer", re.IGNORECASE),
    re.compile("review" + r"\s+" + "comment", re.IGNORECASE),
    re.compile("track" + r"(ed)?\s+" + "changes", re.IGNORECASE),
    re.compile("temporary" + r"\s+" + "synthetic", re.IGNORECASE),
)
TEXT_SUFFIXES = {".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def audit(root: Path, require_license: bool = True) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")
    if require_license and not (root / "LICENSE").is_file():
        errors.append("missing LICENSE (select the intended software license before release)")

    for path in root.rglob("*"):
        ignored_names = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
        ignored = any(part in ignored_names or part.endswith(".egg-info") for part in path.parts)
        if not path.is_file() or ignored:
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/data artifact should not be committed: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB release limit: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN_TEXT:
                if pattern.search(content):
                    errors.append(f"internal/review marker found in: {relative}")
                    break

    for name in ("base.yaml", "dunhuang.yaml", "balinese.yaml"):
        config_path = root / "configs" / name
        if config_path.exists():
            try:
                yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                errors.append(f"invalid YAML in configs/{name}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--allow-missing-license",
        action="store_true",
        help="Use only during preparation; a public release should always include LICENSE",
    )
    args = parser.parse_args()
    errors = audit(args.root.resolve(), require_license=not args.allow_missing_license)
    if errors:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(errors))
    print("Release audit passed")


if __name__ == "__main__":
    main()
