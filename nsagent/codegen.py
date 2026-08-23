
"""
nsagent.codegen — Offline Python script generation from MBPP examples.

Given a natural-language Python task, this module:
  1. Retrieves the closest cleaned MBPP examples.
  2. Extracts a function definition and invocation arguments from the
     example's tests (or the request itself).
  3. Generates a complete standalone script with a __main__ block.
  4. AST-validates the generated script.
  5. Runs it ONLY through RealSandbox and returns the structured result.

No generated code ever executes in the agent process.
"""
from __future__ import annotations

import ast
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from nsagent.knowledge import PythonKnowledgeRetriever
from nsagent.sandbox import RealSandbox, ExecResult

def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "script"


def _literal_eval_safe(node):
    """Safely evaluate a Python AST expression node to a literal."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_call_args(func_name: str, tests: List[str],
                       test_setup: str = "") -> Optional[List[Any]]:
    """Extract literal arguments from the first call to func_name in tests.

    Uses AST parsing of each assert expression, so nested tuples/lists/dicts
    and quoted strings inside arguments are handled correctly.

    Also resolves simple variable assignments in test_setup code.
    """
    var_map: Dict[str, Any] = {}

    if test_setup:
        try:
            setup_tree = ast.parse(test_setup)
            for node in ast.walk(setup_tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            val = _literal_eval_safe(node.value)
                            if val is not None:
                                var_map[t.id] = val
        except Exception:
            pass

    if not isinstance(tests, list):
        return None

    for test in tests:
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

            args: List[Any] = []
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


def _extract_literals_from_text(text: str) -> List[Any]:
    """Fallback: pull strings and numbers from the NL request."""
    literals: List[Any] = []
    for s in re.findall(r'"([^"]*)"|' r"'([^']*)'", text):
        literals.append(s[0] if s[0] else s[1])
    for n in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        literals.append(float(n) if "." in n else int(n))
    return literals


@dataclass
class GeneratedScript:
    task: str
    code: str
    filename: str
    func_name: str
    args: List[Any]
    source_example_id: str
    similarity: float


class PythonScriptGenerator:
    def __init__(self, retriever: PythonKnowledgeRetriever,
                 sandbox: RealSandbox, generated_dir: str = "/content/nsagent_runtime/scripts"):
        self.retriever = retriever
        self.sandbox = sandbox
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- generate
    def generate(self, task: str, top_k: int = 3,
                 min_similarity: float = 0.16) -> GeneratedScript:
        candidates = self.retriever.search(task, top_k=top_k)
        if not candidates:
            raise RuntimeError(f"No MBPP examples found for: {task}")

        candidates = [c for c in candidates if c["similarity"] >= min_similarity]
        if not candidates:
            best = self.retriever.search(task, top_k=1)[0]
            raise RuntimeError(
                f"No sufficiently relevant MBPP example found. "
                f"Best match was {best['func_name']} with similarity "
                f"{best['similarity']:.3f}, below threshold {min_similarity:.2f}."
            )

        last_err = None
        for cand in candidates:
            func_name = cand["func_name"]
            args = _extract_call_args(func_name, cand.get("tests", []), cand.get("test_setup", ""))
            if args is None:
                args = _extract_literals_from_text(task)

            if len(args) < len(cand["args"]):
                # Not enough literals to invoke the function safely.
                last_err = (f"Example {cand['func_name']} expects "
                            f"{len(cand['args'])} argument(s), but only "
                            f"{len(args)} literal(s) could be inferred.")
                continue

            # Trim args to the function arity.
            args = args[: len(cand["args"])]

            arg_reprs = [repr(a) for a in args]
            invocation = f"{func_name}({', '.join(arg_reprs)})"
            code = (
                f"{cand['code'].strip()}\n\n\n"
                f"if __name__ == '__main__':\n"
                f"    result = {invocation}\n"
                f"    print(result)\n"
            )

            try:
                ast.parse(code)
            except SyntaxError as exc:
                last_err = f"Example {cand['func_name']} generated invalid code: {exc}"
                continue

            filename = f"agentgen_{_slug(task)}_{cand['id']}.py"
            return GeneratedScript(
                task=task,
                code=code,
                filename=filename,
                func_name=func_name,
                args=args,
                source_example_id=cand["id"],
                similarity=cand["similarity"],
            )

        raise RuntimeError(f"Could not generate a valid script. Last error: {last_err}")

    # ------------------------------------------------------------------ run
    def run(self, task: str, top_k: int = 3) -> tuple[GeneratedScript, ExecResult]:
        gen = self.generate(task, top_k=top_k)
        res = self.sandbox.run_code(gen.code, filename=gen.filename)
        return gen, res

    # ------------------------------------------------------------ describe
    def explain(self, question: str, top_k: int = 3) -> str:
        return self.retriever.describe(question, top_k=top_k)
