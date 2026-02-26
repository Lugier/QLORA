#!/usr/bin/env python3
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Set


ROOT = Path(__file__).resolve().parents[1]
RUNPOD_SCRIPT = ROOT / "scripts" / "runpod_train_full.sh"


def _collect_python_commands(script_path: Path) -> List[List[str]]:
    commands = []
    pending = ""

    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if pending:
            continued = line.endswith("\\")
            pending += " " + line.rstrip("\\").strip()
            if not continued:
                commands.append(shlex.split(pending))
                pending = ""
            continue

        if line.startswith("python3 "):
            continued = line.endswith("\\")
            pending = line.rstrip("\\").strip()
            if not continued:
                commands.append(shlex.split(pending))
                pending = ""

    if pending:
        commands.append(shlex.split(pending))
    return commands


def _flags_from_command_tokens(tokens: List[str]) -> Set[str]:
    flags = set()
    for token in tokens:
        if token.startswith("--"):
            flags.add(token)
    return flags


def _extract_declared_flags(py_file: Path) -> Set[str]:
    text = py_file.read_text(encoding="utf-8")
    pattern = re.compile(r"add_argument\(\s*['\"](--[a-zA-Z0-9\-]+)")
    declared = set(pattern.findall(text))
    if declared:
        return declared

    # Wrapper support: resolve runpy module wrappers to canonical implementation file.
    run_module_match = re.search(r'run_module\(\s*["\']([a-zA-Z0-9_\.]+)["\']', text)
    if not run_module_match:
        return declared

    module_name = run_module_match.group(1)
    target = ROOT / (module_name.replace(".", "/") + ".py")
    if not target.exists():
        return declared

    target_text = target.read_text(encoding="utf-8")
    return set(pattern.findall(target_text))


def main() -> int:
    if not RUNPOD_SCRIPT.exists():
        print(f"[cli-drift] Missing script: {RUNPOD_SCRIPT}")
        return 1

    commands = _collect_python_commands(RUNPOD_SCRIPT)
    used_flags_by_script: Dict[Path, Set[str]] = {}

    for tokens in commands:
        if len(tokens) < 2:
            continue
        script_token = tokens[1]
        if not script_token.endswith(".py"):
            continue
        script_path = (ROOT / script_token).resolve()
        if not script_path.exists():
            print(f"[cli-drift] Referenced script does not exist: {script_token}")
            return 1
        used_flags_by_script.setdefault(script_path, set()).update(_flags_from_command_tokens(tokens[2:]))

    failures = []
    for script_path, used_flags in sorted(used_flags_by_script.items()):
        declared_flags = _extract_declared_flags(script_path)
        missing = sorted(flag for flag in used_flags if flag not in declared_flags)
        if missing:
            failures.append((script_path, missing))

    if failures:
        print("[cli-drift] FAIL: runpod_train_full.sh uses unknown flags.")
        for script_path, missing in failures:
            print(f"  - {script_path.name}: {missing}")
        return 1

    print("[cli-drift] OK: runpod_train_full.sh flags match argparse declarations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
