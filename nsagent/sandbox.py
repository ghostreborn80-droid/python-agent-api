"""
nsagent.sandbox — RealSandbox: isolated subprocess execution.

Design invariant: NEVER execute project code in the agent process.
Every run goes through `subprocess.Popen` with:
  * explicit timeout (seconds)
  * captured stdout/stderr (text)
  * controlled environment (PYTHONPATH, no bytecode side effects)
  * cwd anchored to the project root

ExecResult is the structured artifact returned by every run; it is what
later causal-memory/self-healing blocks will inspect and attach to the graph.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union


@dataclass
class ExecResult:
    """Structured result of one sandboxed run."""
    command: List[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined(self) -> str:
        return (self.stdout or "") + "\n" + (self.stderr or "")

    def summary(self) -> str:
        status = "PASS" if self.success else ("TIMEOUT" if self.timed_out else "FAIL")
        return (f"ExecResult({status}, rc={self.returncode}, "
                f"duration={self.duration_ms}ms, cmd={' '.join(self.command)})")

    def tail(self, n: int = 2000) -> str:
        text = self.combined
        return text[-n:] if len(text) > n else text


class RealSandbox:
    """Subprocess-isolated executor for the project under test."""

    def __init__(self,
                 project_root: Union[str, Path] = "/content/sample_project",
                 timeout: float = 30.0,
                 max_output_chars: int = 20_000):
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.runtime_dir = Path("/content/nsagent_runtime")
        self.script_dir = self.runtime_dir / "scripts"
        self.script_dir.mkdir(parents=True, exist_ok=True)

    def _env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.project_root) + (os.pathsep + py_path if py_path else "")
        return env

    def _truncate(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]

    def _run(self, cmd: Sequence[str], timeout: Optional[float] = None) -> ExecResult:
        timeout = self.timeout if timeout is None else timeout
        t0 = time.monotonic()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env(),
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
        return ExecResult(
            command=list(cmd),
            cwd=str(self.project_root),
            returncode=proc.returncode,
            stdout=self._truncate(out or ""),
            stderr=self._truncate(err or ""),
            duration_ms=int((time.monotonic() - t0) * 1000),
            timed_out=timed_out,
        )

    def run_pytest(self, args: Optional[Sequence[str]] = None,
                   timeout: Optional[float] = None) -> ExecResult:
        cmd = [sys.executable, "-m", "pytest"]
        cmd += list(args) if args else ["-q"]
        return self._run(cmd, timeout)

    def run_code(self, code: str, filename: Optional[str] = None,
                 timeout: Optional[float] = None) -> ExecResult:
        filename = filename or f"script_{uuid.uuid4().hex[:8]}.py"
        script_path = self.script_dir / filename
        script_path.write_text(code, encoding="utf-8")
        return self._run([sys.executable, str(script_path)], timeout)

    def run_python_file(self, path: Union[str, Path],
                        args: Optional[Sequence[str]] = None,
                        timeout: Optional[float] = None) -> ExecResult:
        cmd = [sys.executable, str(path)] + list(args or [])
        return self._run(cmd, timeout)

    def run_pydoc(self, target: str, timeout: Optional[float] = None) -> ExecResult:
        """Run `python -m pydoc <target>` in an isolated subprocess."""
        return self._run([sys.executable, "-m", "pydoc", target], timeout)
