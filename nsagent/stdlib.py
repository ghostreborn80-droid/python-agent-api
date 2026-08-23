
"""
nsagent.stdlib — Python standard-library documentation retrieval.

Answers general Python questions by running `pydoc` in the sandbox.
No project code is executed. Stdlib modules are safe to inspect.

This is a deterministic offline capability. It detects references like:
  * "itertools.groupby"
  * "collections.Counter"
  * "asyncio.gather"
and returns the first ~2,000 characters of official Python docs.
"""
from __future__ import annotations

import re
from typing import Optional

from nsagent.sandbox import RealSandbox


class PythonStdlibKnowledge:
    # Common modules and known submodules/attributes.
    STDLIB_MODULES = {
        "os", "sys", "io", "json", "csv", "re", "math", "random",
        "statistics", "datetime", "time", "collections", "itertools",
        "functools", "pathlib", "typing", "asyncio", "subprocess",
        "argparse", "logging", "unittest", "sqlite3", "hashlib", "uuid",
        "tempfile", "shutil", "glob", "textwrap", "difflib", "heapq",
        "bisect", "copy", "pprint", "secrets", "decimal", "fractions",
        "struct", "base64", "html", "urllib", "http", "socket",
        "threading", "multiprocessing", "concurrent", "contextlib",
        "enum", "dataclasses", "abc", "traceback", "warnings",
    }

    def __init__(self, sandbox: RealSandbox):
        self.sandbox = sandbox

    def _detect_target(self, question: str) -> Optional[str]:
        """Return the dotted stdlib object referenced in the question."""
        # Dotted reference: collections.Counter, itertools.groupby, etc.
        dotted = re.findall(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)", question)
        for d in dotted:
            root = d.split(".")[0]
            if root in self.STDLIB_MODULES:
                return d

        # Bare module reference: "How do I use itertools?"
        words = set(re.findall(r"\b[a-zA-Z_]\w*\b", question))
        for w in words:
            if w in self.STDLIB_MODULES:
                return w

        return None

    def answer(self, question: str) -> Optional[str]:
        target = self._detect_target(question)
        if not target:
            return None

        res = self.sandbox.run_pydoc(target, timeout=15)
        if not res.success:
            return (f"Python stdlib docs for '{target}' could not be retrieved.\n"
                    f"stderr: {res.stderr.strip()[:400]}")

        doc = (res.stdout or "").strip()
        if not doc:
            doc = (res.stderr or "").strip()

        # pydoc output can be huge; keep the most informative beginning.
        return f"Python stdlib docs for `{target}`:\n\n{doc[:2200]}"
