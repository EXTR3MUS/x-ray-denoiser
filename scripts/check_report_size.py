#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


def count_markdown_words(text: str) -> int:
    in_code_block = False
    words = 0

    for raw_line in text.splitlines():
        line = raw_line

        if re.match(r"^\s*(```|~~~)", line):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"<[^>]*>", " ", line)

        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = re.sub(r"^\s*>+\s*", "", line)

        line = re.sub(r"[^\w\s]", " ", line)
        tokens = [t for t in line.split() if re.search(r"[A-Za-z0-9]", t)]
        words += len(tokens)

    return words


def main() -> int:
    report_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./report.md")

    if not report_file.is_file():
        print(f"Error: file not found: {report_file}", file=sys.stderr)
        return 1

    word_count = count_markdown_words(report_file.read_text(encoding="utf-8", errors="ignore"))
    print(f"Word count: {word_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
