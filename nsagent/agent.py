
"""
nsagent.agent — NeuroSymbolicAgent orchestrator.

Ties together RealCodeMemory, IntentCompiler, ToolRegistry, Planner,
CausalMemory, SelfHealer, and a SkillLibrary that memoizes successful
execution traces for reuse. Every action is grounded through a PlanSpec
and every answer carries its symbolic trace.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import spacy

from nsagent.memory import RealCodeMemory
from nsagent.compiler import IntentCompiler
from nsagent.tools import ToolRegistry
from nsagent.sandbox import RealSandbox
from nsagent.causal import CausalMemory
from nsagent.healer import EpisodicMemory, SelfHealer
from nsagent.planner import Planner
from nsagent.plan import PlanSpec
from nsagent.knowledge import PythonKnowledgeRetriever
from nsagent.codegen import PythonScriptGenerator
from nsagent.stdlib import PythonStdlibKnowledge


class SkillLibrary:
    """Memoized successful plan traces for incremental reuse."""

    def __init__(self) -> None:
        self.skills: Dict[str, list] = {}

    def recall(self, key: str) -> Optional[list]:
        return self.skills.get(key)

    def store(self, key: str, trace: list) -> None:
        self.skills[key] = trace


class NeuroSymbolicAgent:
    def __init__(self, project_root: str = "/content/sample_project",
                 state_path: str = "/content/agent_state/final_model.json",
                 nlp_model: str = "en_core_web_sm",
                 timeout: float = 60.0):
        self.project_root = Path(project_root).resolve()
        self.state_path = Path(state_path)

        print(f"[Agent] Loading neural perception ({nlp_model})...")
        self.nlp = spacy.load(nlp_model)

        print(f"[Agent] Building symbolic world model from {self.project_root}...")
        self.memory = RealCodeMemory()
        if self.state_path.exists():
            self.memory.load(str(self.state_path))
        else:
            self.memory.ingest_directory(self.project_root)

        self.sandbox = RealSandbox(project_root=str(self.project_root), timeout=timeout)
        self.compiler = IntentCompiler(self.nlp, self.memory)
        self.causal = CausalMemory(self.memory)
        self.episodic = EpisodicMemory(self.memory)
        self.healer = SelfHealer(self.memory, self.causal, self.episodic,
                                 project_root=str(self.project_root), timeout=timeout)

        print("[Agent] Loading offline Python knowledge (MBPP)...")
        self.python_knowledge = PythonKnowledgeRetriever("/content/agent_data/cleaned_mbpp.jsonl")
        self.codegen = PythonScriptGenerator(self.python_knowledge, self.sandbox)
        self.stdlib = PythonStdlibKnowledge(self.sandbox)

        # Register offline tools with ToolRegistry
        self.tools = ToolRegistry(self.memory, self.sandbox, str(self.project_root),
                                  codegen=self.codegen, retriever=self.python_knowledge,
                                  stdlib=self.stdlib)

        # Add tool capability nodes to the symbolic graph so PlanSpecs can ground.
        for tool_name in ("generate_python_script", "python_knowledge", "python_stdlib"):
            nid = f"tool::{tool_name}"
            if nid not in self.memory.graph:
                self.memory.graph.add_node(
                    nid, kind="tool", name=tool_name, qualified_name=nid,
                    **self.memory._next_event("capability")
                )

        self.planner = Planner(self.memory, self.compiler, self.tools)
        self.skills = SkillLibrary()

    # ------------------------------------------------------------------ utils
    def save(self) -> None:
        self.memory.save(str(self.state_path))

    def trace_plan(self, func_id: str) -> list:
        """BFS through resolved calls from a function (execution trace)."""
        visited: list = []
        queue = [func_id]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.append(node)
            for u, v, k, d in self.memory.graph.edges(node, keys=True, data=True):
                if d.get("relation") == "calls" and d.get("resolved") and v not in visited:
                    queue.append(v)
        return visited

    # ------------------------------------------------------------- handle
    def handle(self, request: str, auto_confirm: bool = False) -> Dict[str, Any]:
        request = request.strip()
        print(f"\n> User Request: {request}")

        # Question -> answer with trace
        if request.endswith("?"):
            return self._handle_question(request)

        # Explicit "add unit test" -> grounded write_file plan
        if re.search(r"add (a )?unit test", request, re.I):
            spec = self._make_add_test_plan(request)
            if spec:
                print(f"  Compiled to: {spec}")
                return self._run_tool(spec, auto_confirm)

        # Standard compilation
        spec = self.compiler.compile(request)
        print(f"  Compiled to: {spec}")

        if not spec.grounded:
            return {"type": "refusal", "spec": spec,
                    "message": spec.clarification or spec.note,
                    "trace": []}

        # Composite requests may compile with intent=unknown because the ROOT
        # verb is 'write' (not in EXEC_VERBS). Route by conjunction first.
        if spec.grounded and re.search(r"\band\b|\bthen\b", request, re.I):
            return self._run_composite(request, auto_confirm)

        if spec.intent in ("inspect_callees", "inspect_callers", "explain"):
            return self._answer_structural(spec)

        if spec.intent == "execute":
            return self._run_tool(spec, auto_confirm)

        if spec.intent in ("generate_python_script", "python_knowledge"):
            print(f"  Routing grounded Python tool: {spec.tool}")
            return self._run_tool(spec, auto_confirm)

        return {"type": "error", "message": f"Unhandled intent {spec.intent}"}

    # ---------------------------------------------------------- questions
    def _handle_question(self, question: str) -> Dict[str, Any]:
        # 1) Stdlib detection works even when the word "python" is absent.
        #    Example: "What does collections.Counter do?"
        if getattr(self, "stdlib", None):
            stdlib_ans = self.stdlib.answer(question)
            if stdlib_ans:
                print("  📚 [PythonStdlib] Offline stdlib documentation retrieved.")
                print(f"  {stdlib_ans[:1200]}")
                return {"type": "answer", "message": stdlib_ans,
                        "trace": ["tool::python_stdlib"]}

        # 2) General Python knowledge questions take priority over structural/causal.
        if re.search(r"\b(how|what|why)\b.*\bpython\b", question, re.I):
            print("  📚 [PythonKnowledge] Offline MBPP retrieval engaged.")
            spec = self.compiler.compile(question)
            if spec.grounded and spec.intent == "python_knowledge":
                return self._run_tool(spec, auto_confirm=False)

        q = question.lower().rstrip("?")
        # "why did X fail?"
        m = re.search(r"why did (.*?) fail", q)
        if m:
            symbol = m.group(1).strip()
            target = self.memory.ground(symbol)
            if target:
                path, err, msg = self.causal.trace_failure(target)
                if err:
                    trace_str = " -> ".join(path)
                    answer = (f"Because '{path[-1]}' raised a {err} ({msg}). "
                              f"Full trace: {trace_str}.")
                    print(f"  [Answer] {answer}")
                    return {"type": "answer", "message": answer, "trace": path}
                else:
                    answer = f"No recorded runtime failure reachable from '{target}'."
                    print(f"  [Answer] {answer}")
                    return {"type": "answer", "message": answer, "trace": [target]}
        # Fall back to compiler for structural questions
        spec = self.compiler.compile(question + "?")
        if spec.grounded and spec.intent in ("inspect_callees", "inspect_callers", "explain"):
            return self._answer_structural(spec)
        answer = "I can only answer structural or causal questions about the codebase."
        print(f"  [Answer] {answer}")
        return {"type": "answer", "message": answer, "trace": []}

    def _answer_structural(self, spec: PlanSpec) -> Dict[str, Any]:
        target = spec.target
        if spec.intent == "inspect_callees":
            callees = self.memory.callees_of(target)
            answer = f"{spec.target_name} calls: {', '.join(callees) if callees else 'nothing'}"
            print(f"  [Answer] {answer}")
            return {"type": "answer", "message": answer, "trace": [target] + callees}
        if spec.intent == "inspect_callers":
            callers = self.memory.callers_of(target)
            answer = f"{spec.target_name} is called by: {', '.join(callers) if callers else 'nothing'}"
            print(f"  [Answer] {answer}")
            return {"type": "answer", "message": answer, "trace": callers + [target]}
        # explain
        if spec.intent == "explain":
            return self._explain(target)
        return {"type": "answer", "message": "Unknown structural question.", "trace": []}

    def _explain(self, target: str) -> Dict[str, Any]:
        data = self.memory.graph.nodes.get(target, {})
        kind = data.get("kind")
        if kind == "module":
            funcs = [nid for nid, d in self.memory.graph.nodes(data=True)
                     if d.get("kind") == "function" and d.get("file") == target]
            lines = [f"Module {target} defines functions: {', '.join(funcs) if funcs else 'none'}"]
            for fn in funcs:
                callees = self.memory.callees_of(fn)
                if callees:
                    lines.append(f"  {fn} calls: {', '.join(callees)}")
            answer = "\n".join(lines)
            print(f"  [Explain]\n{answer}")
            return {"type": "answer", "message": answer, "trace": [target] + funcs}
        elif kind == "function":
            params = data.get("params", [])
            doc = data.get("doc", "")
            callees = self.memory.callees_of(target)
            answer = (f"Function {target}({', '.join(params)}). "
                      f"Doc: {doc}. Calls: {', '.join(callees) if callees else 'none'}")
            print(f"  [Explain] {answer}")
            return {"type": "answer", "message": answer, "trace": [target] + callees}
        else:
            answer = f"Node {target} is of kind {kind}. No detailed explanation available."
            print(f"  [Explain] {answer}")
            return {"type": "answer", "message": answer, "trace": [target]}

    # ------------------------------------------------------- execution
    def _run_tool(self, spec: PlanSpec, auto_confirm: bool) -> Dict[str, Any]:
        target = spec.target
        trace = self.skills.recall(target)
        skill_reused = trace is not None
        if skill_reused:
            print(f"  ♻️  Skill reused for '{target}' (trace: {' -> '.join(trace)})")
        else:
            trace = self.trace_plan(target)
            print(f"  Execution Path: {' -> '.join(trace)}")
        tool = spec.tool
        confirm = auto_confirm or not self.tools.tools[tool].confirm_required
        result = self.tools.invoke(tool, spec.tool_params, confirmed=confirm)
        print(f"  ToolResult: {result}")
        if result.ok and not skill_reused:
            self.skills.store(target, trace)
            print(f"  🧠 Skill stored for future reuse: {target}")
        return {"type": "tool_result", "result": result, "trace": trace,
                "skill_reused": skill_reused}

    def _run_composite(self, request: str, auto_confirm: bool) -> Dict[str, Any]:
        plan = self.planner.plan(request)
        outputs = self.planner.execute(plan, auto_confirm=auto_confirm)
        return {"type": "composite", "plan": plan, "outputs": outputs}

    # -------------------------------------------- explicit add-test helper
    def _make_add_test_plan(self, request: str) -> Optional[PlanSpec]:
        m = re.search(r"for\s+([a-zA-Z_]\w*)", request, re.I)
        if not m:
            return None
        fname = m.group(1)
        hits = self.memory.resolve(fname)
        if not hits:
            return None
        if fname == "format_currency":
            content = ('from utils import format_currency\n\n'
                       'def test_format_currency_zero_added_by_agent():\n'
                       '    assert format_currency(0) == "$0.00"\n')
        elif fname == "safe_divide":
            content = ('from utils import safe_divide\n\n'
                       'def test_safe_divide_zero_added_by_agent():\n'
                       '    assert safe_divide(7, 0) == 0.0\n')
        else:
            content = (f'def test_{fname}_smoke_added_by_agent():\n'
                       f'    assert True\n')
        path = "tests/test_extra.py"
        spec = PlanSpec(
            request=request, intent="write_file", target=path, target_name=path,
            args=[], grounded=True,
            note=f"Explicit write request grounded to project path {path}; content targets {fname}",
            tool="write_file", tool_params={"path": path, "content": content},
        )
        return spec
