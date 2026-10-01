"""Explicit publication inventory: no downloaded corpus or temporary results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_FILES = (".gitignore", "README.md", "USAGE.md", "LICENSE", "requirements.txt",
              "environment.lock.txt", "pyproject.toml", "repository.json", "prepare.py", "study.py",
              "generate.py", "benchmark.py", "sampler_check.py", "plots.py", "verify.py", "artifacts.py")


def files(include_audit=False):
    paths = [ROOT/name for name in ROOT_FILES]
    paths += [ROOT/"data/manifest.json"]
    for folder, pattern in (("specforge", "*.py"), ("tests", "*.py"), ("figures", "*.png")):
        paths += sorted((ROOT/folder).glob(pattern))
    paths += sorted(path for path in (ROOT/"runs").rglob("*")
                    if path.is_file() and path.name != "results.partial.json")
    if include_audit:
        paths.append(ROOT/"audit.json")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(paths))
