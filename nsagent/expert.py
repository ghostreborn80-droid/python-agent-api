
"""
nsagent.expert — Retrieval-first Python expert using the expanded SQLite brain.

This module connects the 38,793-example knowledge base to the paid API.
It does sequential candidate execution inside the sandbox and returns the
first candidate whose execution succeeds.
"""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nsagent.db_retriever import DatabaseRetriever
from nsagent.sandbox import RealSandbox, ExecResult


class GenResult:
    def __init__(self, filename: str, code: str):
        self.filename = filename
        self.code = code


def _literal_eval_safe(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_call_args(func_name, tests, test_setup=""):
    var_map = {}
    if test_setup:
        try:
            tree = ast.parse(test_setup)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            val = _literal_eval_safe(node.value)
                            if val is not None:
                                var_map[t.id] = val
        except Exception:
            pass

    for test in tests or []:
        if not isinstance(test, str):
            continue
        try:
            tree = ast.parse(test, mode="exec")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == func_name):
                continue
            args = []
            ok = True
            for arg_node in node.args:
                val = _literal_eval_safe(arg_node)
                if val is not None:
                    args.append(val)
                elif isinstance(arg_node, ast.Name) and arg_node.id in var_map:
                    args.append(var_map[arg_node.id])
                else:
                    ok = False
                    break
            if ok:
                return args
    return None


def _extract_literals(text: str):
    literals = []
    for s in re.findall(r'"([^"]*)"|' + r"'([^']*)'", text):
        literals.append(s[0] if s[0] else s[1])
    for n in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        literals.append(float(n) if "." in n else int(n))
    return literals


class PythonExpertAgent:
    def __init__(self,
                 db_path: str = "/app/agent_data/python_knowledge.db",
                 skills_db_path: Optional[str] = "/app/agent_data/full_training_runs.db",
                 project_root: str = "/app/sample_project",
                 timeout: float = 30.0):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Knowledge DB not found: {self.db_path}")

        self.retriever = DatabaseRetriever(str(self.db_path))
        self.sandbox = RealSandbox(project_root=project_root, timeout=timeout)
        self.skill_ranks = {}

        if skills_db_path and Path(skills_db_path).exists():
            self._load_skills(skills_db_path)

    def _load_skills(self, skills_db_path: str):
        try:
            conn = sqlite3.connect(skills_db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT task_id, source_example_id, rank FROM skills"
            ).fetchall()
            conn.close()
            self.skill_ranks = {r["task_id"]: r for r in rows}
            print(f"[PythonExpert] loaded {len(self.skill_ranks)} learned skills")
        except Exception as exc:
            print(f"[PythonExpert] skills load skipped: {exc}")

    def answer(self, question: str) -> Dict[str, Any]:
        text = self.retriever.describe(question, top_k=5)
        return {"type": "answer", "message": text, "trace": ["db_retriever"]}

    def generate(self, task: str, top_k: int = 5) -> Dict[str, Any]:
        candidates = self.retriever.search(task, top_k=top_k)

        last_res = None
        last_gen = None

        for rank, cand in enumerate(candidates, 1):
            cand_id = cand.get("id", "unknown")
            cand_code = cand.get("code", "").strip()
            func_name = cand.get("func_name") or self._extract_func_name(cand_code)

            script, filename = self._make_executable_script(cand, task, func_name)
            res = self.sandbox.run_code(script, filename=filename)

            gen = GenResult(filename, script)
            gen.source_example_id = cand_id
            gen.func_name = func_name
            gen.rank = rank
            gen.similarity = cand.get("similarity", 0.0)

            if res.success:
                return {
                    "type": "python_script",
                    "generated": gen,
                    "exec_result": res,
                    "trace": [cand_id],
                }

            last_res = res
            last_gen = gen

        return {
            "type": "python_script",
            "generated": last_gen or GenResult("expert_failed.py", ""),
            "exec_result": last_res,
            "trace": [],
        }

    def _make_executable_script(self, cand, task, func_name):
        code = cand.get("code", "").strip()
        args = _extract_call_args(func_name, cand.get("tests", []), cand.get("test_setup", ""))
        if args is None:
            args = _extract_literals(task)

        declared = cand.get("args", [])
        if declared and args:
            args = args[: len(declared)]

        if func_name and args:
            arg_reprs = [repr(a) for a in args]
            invocation = f"{func_name}({', '.join(arg_reprs)})"
            code += f"\n\nif __name__ == '__main__':\n    result = {invocation}\n    print(result)\n"

        safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(cand.get("id", "x")))[:40]
        filename = f"expert_{safe_id}.py"
        return code, filename

    def _extract_func_name(self, code):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except SyntaxError:
            pass
        return ""
