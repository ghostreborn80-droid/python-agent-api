
"""
nsagent.planner — Planner for composite goals (R6).

Decomposes multi-step NL requests (e.g. "clean sales.csv and write an HTML
report") into a DAG of tool invocations. Each subgoal is grounded against
RealCodeMemory and executed only through ToolRegistry. The DAG is printed
before execution so the agent's reasoning is auditable.

Supported decomposition rules:
  * split on " and " / " then "
  * clause starting with "write"/"save" -> write_file tool with inferred path
  * clean-like clause targeting a dataframe-taking function -> find a
    path-taking wrapper in the graph that transitively calls that function
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nsagent.compiler import IntentCompiler
from nsagent.memory import RealCodeMemory
from nsagent.plan import PlanSpec
from nsagent.tools import ToolRegistry, ToolResult


@dataclass
class SubGoal:
    id: str
    tool: str
    params: Dict[str, Any]
    source_clause: str
    dependencies: List[str] = field(default_factory=list)
    confirm_required: bool = False
    note: str = ""


@dataclass
class PlanDAG:
    subgoals: List[SubGoal]
    edges: List[Tuple[str, str, str]]  # (from_id, to_id, relation)

    def print(self) -> None:
        print("\n  ┌─ Composite Plan DAG")
        for sg in self.subgoals:
            confirm = " [CONFIRM REQUIRED]" if sg.confirm_required else ""
            deps = f" deps={sg.dependencies}" if sg.dependencies else ""
            print(f"  │ {sg.id}: {sg.tool} {sg.params}{confirm}{deps}")
            print(f"  │   from clause: {sg.source_clause!r}")
        for u, v, rel in self.edges:
            print(f"  │ {u} --[{rel}]--> {v}")
        print("  └────────────────────────")


class Planner:
    CLAUSE_SPLIT_RE = re.compile(r"\s+(?:and|then)\s+", re.IGNORECASE)
    WRITE_VERBS = {"write", "save", "output", "produce"}
    REPORT_NAME_RE = re.compile(r"\b([\w-]+\.html?)\b", re.IGNORECASE)

    def __init__(self, memory: RealCodeMemory, compiler: IntentCompiler,
                 tools: ToolRegistry):
        self.memory = memory
        self.compiler = compiler
        self.tools = tools

    # ------------------------------------------------------------- decompose
    def plan(self, request: str) -> PlanDAG:
        clauses = [c.strip() for c in self.CLAUSE_SPLIT_RE.split(request) if c.strip()]
        if len(clauses) < 2:
            # Not composite; build a single-subgoal DAG directly from compiler.
            subgoals, edges = self._single_subgoal(request)
            return PlanDAG(subgoals=subgoals, edges=edges)

        subgoals: List[SubGoal] = []
        edges: List[Tuple[str, str, str]] = []
        produced_outputs: Dict[int, str] = {}  # subgoal_idx -> artifact name

        for i, clause in enumerate(clauses):
            sg = self._clause_to_subgoal(clause, i, full_request=request)
            subgoals.append(sg)

        # Data dependencies: if a write/save clause can consume previous output,
        # wire it. We infer by checking if any earlier subgoal produced data.
        for i, sg in enumerate(subgoals):
            if sg.tool == "write_file" and "content" in sg.params:
                # If content is an execution placeholder, attach dependency.
                if isinstance(sg.params["content"], str) and sg.params["content"].startswith("@"):
                    dep_id = sg.params["content"][1:]  # e.g. "@subgoal_0"
                    if any(dep_id == other.id for other in subgoals):
                        sg.dependencies.append(dep_id)
                        edges.append((dep_id, sg.id, "data_flow"))
                        # KEEP "@dep_id" so execute() can substitute actual output.

        return PlanDAG(subgoals=subgoals, edges=edges)

    # --------------------------------------------------------------- single
    def _single_subgoal(self, request: str):
        spec = self.compiler.compile(request)
        sg = self._spec_to_subgoal(spec, request, 0)
        return [sg], []

    # -------------------------------------------------------- clause handling
    def _clause_to_subgoal(self, clause: str, idx: int, full_request: str = "") -> SubGoal:
        # Special handling: explicit write/save artifact.
        first_word = clause.split()[0].lower().rstrip(",.") if clause.split() else ""
        if first_word in self.WRITE_VERBS:
            m = self.REPORT_NAME_RE.search(clause)
            path = m.group(1) if m else "report.html"
            return SubGoal(
                id=f"subgoal_{idx}",
                tool="write_file",
                params={"path": path, "content": f"@subgoal_{idx - 1}" if idx else "<generated>"},
                source_clause=clause,
                confirm_required=True,  # write_file is destructive
                note="write artifact produced by prior step; placeholder resolves at execution",
            )

        # Goal-directed rule: clean-like clause inside a report/html request
        # should run the FULL report pipeline, not just the cleaning function.
        if (idx == 0 and re.search(r"\bclean\b", clause, re.IGNORECASE)
                and re.search(r"\breport\b|html|web", full_request or "", re.IGNORECASE)):
            file_args = self.compiler.FILE_ARG_PAT.findall(clause)
            pipeline = self._find_report_pipeline()
            if file_args and pipeline:
                return SubGoal(
                    id=f"subgoal_{idx}",
                    tool="run_function",
                    params={"function": pipeline, "args": file_args},
                    source_clause=clause,
                    note="full pipeline: load -> clean -> analyze -> HTML string",
                )

        spec = self.compiler.compile(clause)
        # Resolve file-vs-dataframe mismatch via wrapper search.
        spec = self._resolve_wrapper(spec)
        return self._spec_to_subgoal(spec, clause, idx)

    def _find_report_pipeline(self) -> Optional[str]:
        """Return a report-producing function (path -> HTML string)."""
        for name in ("generate_report", "run_pipeline"):
            hits = self.memory.resolve(name)
            if hits:
                return hits[0]
        for nid, data in self.memory.graph.nodes(data=True):
            if data.get("kind") == "function" and data.get("file", "").endswith("report.py"):
                if "html" in (data.get("doc") or "").lower() or "report" in nid.lower():
                    return nid
        return None

    def _spec_to_subgoal(self, spec: PlanSpec, clause: str, idx: int) -> SubGoal:
        if not spec.grounded:
            return SubGoal(
                id=f"subgoal_{idx}",
                tool="UNGROUNDED",
                params={},
                source_clause=clause,
                note=spec.clarification or spec.note,
            )
        tool = spec.tool
        params = dict(spec.tool_params)
        confirm_required = self.tools.tools.get(tool).confirm_required if tool in self.tools.tools else False
        return SubGoal(id=f"subgoal_{idx}", tool=tool or "noop", params=params,
                       source_clause=clause, confirm_required=confirm_required)

    # ------------------------------------------------- wrapper resolution
    def _resolve_wrapper(self, spec: PlanSpec) -> PlanSpec:
        """If the target takes a DataFrame and the user supplied a file arg,
        find a path-taking function that transitively calls the target."""
        if not spec.grounded or spec.intent != "execute" or not spec.args:
            return spec
        target = spec.target
        params = self.memory.graph.nodes.get(target, {}).get("params", [])
        if not params or params[0] in ("path", "file", "filename"):
            return spec
        # Target expects e.g. df, but user gave a CSV path.
        wrappers = self._find_wrappers(target)
        if wrappers:
            # Prefer wrapper whose name contains "generate" or "run" or "pipeline"
            priority = [w for w in wrappers if re.search(r"generate|run|pipeline", w, re.I)]
            chosen = priority[0] if priority else wrappers[0]
            spec.target = chosen
            spec.target_name = self.memory.graph.nodes[chosen].get("name")
            spec.note = f"Resolved {target} -> {chosen} (wrapper accepts file path)"
            tool_call = spec.to_tool_call()
            if tool_call:
                spec.tool, spec.tool_params = tool_call
        return spec

    def _find_wrappers(self, target: str) -> List[str]:
        """Find function nodes with first param path/file/filename that
        transitively call `target` through resolved calls."""
        wrappers: List[str] = []

        def reaches(start: str, seen: set) -> bool:
            if start == target:
                return True
            if start in seen:
                return False
            seen.add(start)
            for u, v, k, d in self.memory.graph.edges(start, keys=True, data=True):
                if d.get("relation") == "calls" and d.get("resolved"):
                    if reaches(v, seen):
                        return True
            return False

        for nid, data in self.memory.graph.nodes(data=True):
            if data.get("kind") != "function" or nid.startswith("tests/"):
                continue
            params = data.get("params", [])
            if params and params[0] in ("path", "file", "filename"):
                if reaches(nid, set()):
                    wrappers.append(nid)
        return wrappers

    # --------------------------------------------------------------- execute
    def execute(self, plan_dag: PlanDAG, auto_confirm: bool = False,
                verbose: bool = True) -> Dict[str, Any]:
        """Execute subgoals in dependency order. Destructive tools require
        confirmation unless auto_confirm=True (used only in demos where the
        user explicitly approves the printed plan)."""
        plan_dag.print()
        confirmed: set = set()
        if auto_confirm:
            confirmed = {sg.id for sg in plan_dag.subgoals if sg.confirm_required}
            if confirmed:
                print(f"  ⚠️  Auto-confirming destructive subgoals: {sorted(confirmed)}")

        outputs: Dict[str, Any] = {}
        executed = 0
        for sg in plan_dag.subgoals:
            if sg.tool == "UNGROUNDED":
                print(f"\n  ❌ {sg.id} UNGROUNDED: {sg.note}")
                continue
            if sg.confirm_required and sg.id not in confirmed:
                print(f"\n  ⏸️  {sg.id} requires confirmation for {sg.tool}. Skipping.")
                continue

            # Resolve placeholders from prior outputs.
            params = dict(sg.params)
            if sg.tool == "write_file":
                if isinstance(params.get("content"), str) and params["content"].startswith("@"):
                    dep_id = params["content"][1:]
                    dep_output = outputs.get(dep_id)
                    if dep_output is None:
                        print(f"  ❌ {sg.id} missing dependency {dep_id}")
                        continue
                    params["content"] = dep_output

            if verbose:
                print(f"\n  ▶ {sg.id} -> {sg.tool}{params}")
            result = self.tools.invoke(sg.tool, params, confirmed=True)
            print(f"     {result}")

            if result.ok:
                output = self._extract_output(sg.tool, result)
                outputs[sg.id] = output
                executed += 1
        print(f"\n  ✅ Executed {executed}/{len(plan_dag.subgoals)} subgoals.")
        return outputs

    def _extract_output(self, tool: str, result: ToolResult):
        """Extract a useful artifact from a ToolResult for downstream steps."""
        if tool == "run_function":
            exec_res = result.result
            if exec_res and hasattr(exec_res, "stdout"):
                return exec_res.stdout.strip()
        if tool == "run_tests":
            return result.message
        return result.result
