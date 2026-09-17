#!/usr/bin/env python3
"""Flag Path.read_text()/write_text() calls that do not pass an explicit encoding.

Ruff's PLW1514 covers ``open`` only, so these slip through it. They matter for the same reason:
without an encoding Python uses the locale codepage, which is cp1252 on Windows, and a UTF-8 file
then fails to decode. Parsed rather than grepped -- a regex cannot tell whether ``encoding=`` sits
on a later line of a multi-line call, and reports it as a violation.
"""

from __future__ import annotations

import ast
import sys

TARGETS = {"read_text", "write_text"}


def violations(path: str) -> list[tuple[int, str]]:
    """Return ``(lineno, method)`` for each offending call in ``path``.

    A file that will not parse is not ours to police -- ruff and the formatter already have
    an opinion about it, and a syntax error reported from here would only be noise.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return []  # not ours to police
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in TARGETS
            and not any(kw.arg == "encoding" for kw in node.keywords)
        ):
            found.append((node.lineno, node.func.attr))
    return found


def main(argv: list[str]) -> int:
    """Report every offending call across ``argv``; exit non-zero when any is found."""
    bad = [(p, ln, name) for p in argv for ln, name in violations(p)]
    for path, lineno, name in bad:
        print(
            f"{path}:{lineno}: {name}() without an explicit encoding= (locale codepage on Windows)"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
