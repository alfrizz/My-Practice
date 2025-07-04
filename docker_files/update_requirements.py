#!/usr/bin/env python3
import sys
import nbformat
import keyword
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT    = SCRIPT_DIR.parent
REQ_FILE   = SCRIPT_DIR / "requirements.txt"

# Names to skip (stdlib, __future__, meta-packages)
STD_SKIP = {
    "__future__", "sys", "os", "time", "platform", "pathlib",
    "re", "json", "logging", "math", "functools", "itertools",
    "subprocess", "threading", "argparse", "typing", "azure"
}

def is_local(pkg: str) -> bool:
    """Return True if pkg.py or pkg/__init__.py exists under PROJECT."""
    # any .py file named pkg.py
    if any(p.name == f"{pkg}.py" for p in PROJECT.rglob("*.py")):
        return True
    # any package directory named pkg
    for d in PROJECT.rglob("*"):
        if d.is_dir() and d.name == pkg and (d / "__init__.py").exists():
            return True
    return False

def scan_file(path: Path):
    """Extract top‐level module names from import/from lines."""
    text = ""
    if path.suffix == ".ipynb":
        nb = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in nb.cells)
    else:
        text = path.read_text(errors="ignore")

    found = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("import "):
            for part in line[7:].split(","):
                pkg = part.strip().split(".")[0]
                found.add(pkg)
        elif line.startswith("from "):
            pkg = line.split()[1].split(".")[0]
            found.add(pkg)
    return found

# 1) Determine which files to scan
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    changed_list = Path(sys.argv[1]).read_text().splitlines()
    paths = [PROJECT / p for p in changed_list]
    print(f"🔎 Scanning {len(paths)} changed files for imports…")
else:
    paths = list(PROJECT.rglob("*.py")) + list(PROJECT.rglob("*.ipynb"))
    print(f"🔎 No change‐list provided → full scan of {len(paths)} files…")

# 2) Gather imports
raw_pkgs = set()
for p in paths:
    if p.exists():
        raw_pkgs |= scan_file(p)

# 3) Filter out stdlib, keywords, locals, non‐alpha starts
pkgs = {
    pkg for pkg in raw_pkgs
    if pkg
       and pkg[0].isalpha()
       and pkg not in STD_SKIP
       and not keyword.iskeyword(pkg)
       and not is_local(pkg)
}

# 4) Merge with existing requirements
existing = set()
if REQ_FILE.exists():
    existing = set(REQ_FILE.read_text().splitlines())

merged = sorted(existing | pkgs)
new_txt = "\n".join(merged) + "\n"
old_txt = REQ_FILE.read_text() if REQ_FILE.exists() else ""

# 5) Write only if contents changed
if new_txt != old_txt:
    REQ_FILE.write_text(new_txt)
    print(f"✅ requirements.txt updated → {len(merged)} packages")
else:
    print("✅ requirements.txt unchanged → skipping write")
