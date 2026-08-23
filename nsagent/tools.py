"""
nsagent.tools — Typed ToolRegistry.

Tools are callable ONLY through grounded PlanSpecs. Destructive tools
(write_file, pip_install, run_command) require explicit confirmation.
Every tool invocation is subprocess-isolated via RealSandbox where code
execution is involved — never in the agent process.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from nsagent.memory import RealCodeMemory
from nsagent.sandbox import RealSandbox, ExecResult


@dataclass
class ToolSpec:
    name: str
    description: str
    risk: str  # read | write | exec
    confirm_required: bool = False
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    ok: bool
    message: str = ""
    result: Optional[Any] = None

    def __repr__(self) -> str:
        status = "OK" if self.ok else "REFUSED/ERROR"
        return f"ToolResult({self.tool}, {status}, message={self.message!r})"


class ToolRegistry:
    def __init__(self, memory: RealCodeMemory, sandbox: RealSandbox,
                 project_root: str = "/content/sample_project",
                 codegen=None, retriever=None, stdlib=None):
        self.memory = memory
        self.sandbox = sandbox
        self.project_root = Path(project_root).resolve()
        self.tools: Dict[str, ToolSpec] = {}
        self.codegen = codegen
        self.retriever = retriever
        self.stdlib = stdlib
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(ToolSpec("read_file", "Read a file from the project.", "read",
                               params={"path": "relative path within project"}))
        self.register(ToolSpec("write_file", "Write a file into the project (DESTRUCTIVE).", "write",
                               confirm_required=True,
                               params={"path": "relative path", "content": "text content"}))
        self.register(ToolSpec("run_tests", "Run pytest in an isolated subprocess.", "exec",
                               params={"args": "list of pytest args"}))
        self.register(ToolSpec("run_function",
                               "Run a specific project function in a sandboxed subprocess.",
                               "exec",
                               params={"function": "qualified function id", "args": "positional args"}))
        self.register(ToolSpec("run_command",
                               "Run an allowlisted command in a sandbox (DESTRUCTIVE/GATED).",
                               "exec", confirm_required=True,
                               params={"cmd": "allowlisted command prefix"}))
        self.register(ToolSpec("pip_install",
                               "Install a package into the sandbox environment (DESTRUCTIVE/GATED).",
                               "write", confirm_required=True,
                               params={"package": "package name"}))

        if self.retriever is not None:
            self.register(ToolSpec("python_knowledge",
                                   "Answer a general Python question using the offline MBPP knowledge base.",
                                   "read", params={"question": "natural language Python question"}))
        if self.stdlib is not None:
            self.register(ToolSpec("python_stdlib",
                                   "Answer a Python question using the sandboxed Python standard library documentation (pydoc).",
                                   "read", params={"question": "natural language Python question"}))
        if self.codegen is not None:
            self.register(ToolSpec("generate_python_script",
                                   "Generate and run a standalone Python script from a natural-language task.",
                                   "exec", params={"task": "natural language Python task"}))

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def list_tools(self) -> str:
        lines = ["ToolRegistry:"]
        for name, t in sorted(self.tools.items()):
            confirm = " (CONFIRM REQUIRED)" if t.confirm_required else ""
            lines.append(f"  {name:14s} risk={t.risk:5s}{confirm} — {t.description}")
        return "\n".join(lines)

    def invoke(self, tool_name: str, params: Optional[Dict[str, Any]] = None,
               confirmed: bool = False) -> ToolResult:
        params = params or {}
        spec = self.tools.get(tool_name)
        if not spec:
            return ToolResult(tool_name, False, f"Unknown tool '{tool_name}'. Available: {sorted(self.tools)}")
        if spec.confirm_required and not confirmed:
            return ToolResult(tool_name, False,
                              f"Tool '{tool_name}' is destructive. Confirm explicitly (confirmed=True) to proceed.")
        try:
            fn = getattr(self, f"_tool_{tool_name}")
            return fn(params)
        except Exception as exc:
            return ToolResult(tool_name, False, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------- tool impls
    def _tool_read_file(self, p: Dict[str, Any]) -> ToolResult:
        path = self._resolve_project_path(p["path"])
        if not path.is_file():
            return ToolResult("read_file", False, f"File not found: {path}")
        return ToolResult("read_file", True, f"Read {path.relative_to(self.project_root)} ({path.stat().st_size} bytes)",
                          result=path.read_text(encoding="utf-8")[:2000])

    def _tool_write_file(self, p: Dict[str, Any]) -> ToolResult:
        path = self._resolve_project_path(p["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p["content"], encoding="utf-8")
        return ToolResult("write_file", True, f"Wrote {path.relative_to(self.project_root)} ({len(p['content'])} chars)")

    def _tool_run_tests(self, p: Dict[str, Any]) -> ToolResult:
        res = self.sandbox.run_pytest(args=p.get("args", ["-q"]))
        return ToolResult("run_tests", res.success, res.summary(), result=res)

    def _tool_run_function(self, p: Dict[str, Any]) -> ToolResult:
        func_id = p["function"]
        args = p.get("args", [])
        # Grounding check: function must exist in graph and be non-external.
        if func_id not in self.memory.graph or self.memory.graph.nodes[func_id].get("kind") != "function":
            return ToolResult("run_function", False, f"Function '{func_id}' is not a grounded function node.")
        file_part, _, fname = func_id.partition("::")
        module_name = file_part[:-3].replace("/", ".") if file_part.endswith(".py") else file_part.replace("/", ".")
        arg_repr = ", ".join(repr(a) for a in args)
        code = (
            f"import json\n"
            f"from {module_name} import {fname}\n"
            f"result = {fname}({arg_repr})\n"
            f"print(json.dumps(result, default=str) if not isinstance(result, str) else result)\n"
        )
        res = self.sandbox.run_code(code, filename=f"tool_{fname}.py")
        return ToolResult("run_function", res.success, res.summary(), result=res)

    def _tool_run_command(self, p: Dict[str, Any]) -> ToolResult:
        cmd = p["cmd"]
        if not isinstance(cmd, list):
            cmd = cmd.split()
        allowlist = [sys.executable + " -m pytest", sys.executable + " -m unittest"]
        joined = " ".join(cmd[:2])
        if not any(joined.startswith(allowed) for allowed in allowlist):
            return ToolResult("run_command", False,
                              f"Command not allowlisted. Allowlist: {allowlist}")
        res = self.sandbox._run(cmd)
        return ToolResult("run_command", res.success, res.summary(), result=res)

    def _tool_pip_install(self, p: Dict[str, Any]) -> ToolResult:
        return ToolResult("pip_install", False,
                          "pip_install is gated and not yet implemented in this block.")

    def _tool_python_stdlib(self, p: Dict[str, Any]) -> ToolResult:
        if not self.stdlib:
            return ToolResult("python_stdlib", False, "No stdlib documentation engine configured.")
        answer = self.stdlib.answer(p["question"])
        if answer is None:
            return ToolResult("python_stdlib", False, "No standard-library reference detected in the question.")
        return ToolResult("python_stdlib", True, "Retrieved Python stdlib documentation", result=answer)

    def _tool_python_knowledge(self, p: Dict[str, Any]) -> ToolResult:
        if not self.retriever:
            return ToolResult("python_knowledge", False, "No Python knowledge retriever configured.")
        answer = self.retriever.describe(p["question"], top_k=3)
        return ToolResult("python_knowledge", True, "Retrieved Python knowledge examples", result=answer)

    def _tool_generate_python_script(self, p: Dict[str, Any]) -> ToolResult:
        if not self.codegen:
            return ToolResult("generate_python_script", False, "No Python script generator configured.")
        try:
            gen, res = self.codegen.run(p["task"])
        except Exception as exc:
            return ToolResult("generate_python_script", False,
                              f"{type(exc).__name__}: {exc}")
        return ToolResult("generate_python_script", res.success, res.summary(),
                          result={"generated": gen, "exec_result": res})

    def _resolve_project_path(self, p: str) -> Path:
        path = (self.project_root / p).resolve()
        if not str(path).startswith(str(self.project_root)):
            raise ValueError(f"Path escapes project root: {p}")
        return path
