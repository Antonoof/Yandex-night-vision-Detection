"""pre-commit hook: fail if any staged file contains Cyrillic text.

This repo is English-only (docs, comments, log messages - see README.md).
Written in Python rather than as a grep one-liner because grep's Unicode
support differs between BSD grep (macOS) and GNU grep (Linux/CI): a `-P`
pattern that works for one contributor silently no-ops for another.
"""

import re
import sys

# U+0400-U+04FF is the Cyrillic block, spelled as escapes rather than
# literal characters so this file does not trip its own check.
CYRILLIC = re.compile("[\u0400-\u04FF]")


def main(paths):
    found = False
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable - not this hook's job
        for lineno, line in enumerate(text.splitlines(), start=1):
            if CYRILLIC.search(line):
                print(f"{path}:{lineno}: {line}")
                found = True
    if found:
        print("\nCyrillic text found above - this repo is English-only, see README.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
