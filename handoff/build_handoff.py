"""Refresh the source listing and portable IDE handoff archive."""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff"
MODULES = ("data_agent.py", "model_agent.py", "validator_agent.py", "main_simulation.py",
           "scientific_report.py", "serve_preview.py")


def main() -> None:
    sections = [
        "# Complete WPF Python source\n",
        "Snapshot of the working tree. See README.md and HANDOFF.md for setup and data limitations.\n",
    ]
    for name in MODULES:
        source = (ROOT / name).read_text(encoding="utf-8").rstrip()
        sections.append(f"## {name}\n\n```python\n{source}\n```\n")
    (HANDOFF / "COMPLETE_CODE.md").write_text("\n".join(sections), encoding="utf-8")

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    names = {name for name in tracked if name and name != "handoff/wpf-ide-handoff.zip"}
    names.add("handoff/build_handoff.py")
    archive_path = HANDOFF / "wpf-ide-handoff.zip"
    checksums = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names):
            content = (ROOT / name).read_bytes()
            archive.writestr(name, content)
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
        archive.writestr("SHA256SUMS.txt", "\n".join(checksums) + "\n")
    print(f"Updated {archive_path.name}: {len(names)} files")


if __name__ == "__main__":
    main()
