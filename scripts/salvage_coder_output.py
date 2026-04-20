"""Extract files from a truncated coder task output.

Usage:
    python scripts/salvage_coder_output.py <input-json-file> <output-directory>

Reads a JSON file of the shape:
    {"raw_text": "...the (possibly truncated) coder output...", "parse_error": true}

Parses as many {"path": "...", "content": "..."} entries as it can from the
raw_text, stopping at the first incomplete one, and writes each file to the
output directory (creating parent folders as needed).

Exists because a max_tokens overflow turned a valid scaffold response into
unparseable JSON in the `tasks.output` column. This script recovers the
intact prefix so the user doesn't have to re-run the whole workflow.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def salvage(raw_text: str) -> list[dict]:
    """Parse as many complete file entries as possible out of a truncated JSON string."""
    # The coder output is embedded inside {"raw_text": "..."} where "..." is
    # itself an escaped JSON document. After json.loads of the outer wrapper,
    # raw_text is the original text the model produced.
    #
    # We want every object shaped like:
    #   {"path": "some/path", "content": "some content"}
    # The tricky part: "content" can contain escaped quotes, newlines, and
    # {}. Use the json library on each candidate slice rather than regex alone.

    files: list[dict] = []
    # Find each `"path":` key. Each is a file start candidate.
    starts = [m.start() for m in re.finditer(r'"path"\s*:\s*"', raw_text)]
    for i, start in enumerate(starts):
        # Find the opening `{` before this "path" key
        brace = raw_text.rfind("{", 0, start)
        if brace == -1:
            continue
        # Scan forward looking for a matching close brace at the same depth.
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(brace, len(raw_text)):
            c = raw_text[j]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            break  # truncated here
        candidate = raw_text[brace:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict) and "path" in parsed and "content" in parsed:
            files.append(parsed)

    return files


def safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/").replace("\\", "/")
    if ".." in rel.split("/"):
        return None
    out = (root / rel).resolve()
    try:
        out.relative_to(root.resolve())
    except ValueError:
        return None
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: salvage_coder_output.py <input-json> <output-dir>", file=sys.stderr)
        return 2
    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = json.loads(in_path.read_text(encoding="utf-8"))
    raw_text = wrapper.get("raw_text") or ""
    if not raw_text:
        print("no `raw_text` field — is this actually a parse-error output?", file=sys.stderr)
        return 1

    files = salvage(raw_text)
    if not files:
        print("no recoverable files in the raw text", file=sys.stderr)
        return 1

    written = 0
    for entry in files:
        target = safe_join(out_dir, entry["path"])
        if target is None:
            print(f"  [skip] unsafe path: {entry['path']!r}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry["content"], encoding="utf-8")
        written += 1
        print(f"  [ok]   {entry['path']}  ({len(entry['content'])} chars)")

    print(f"\nSalvaged {written} files to {out_dir}")
    print("Note: the final file may be incomplete if the model was cut off mid-content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
