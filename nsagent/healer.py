
"""
nsagent.healer — SelfHealingLoop: shadow-copy patch -> test -> commit-or-rollback.

Patches are generated as AST transforms + ast.unparse into a SHADOW copy.
The project's test suite runs inside RealSandbox against the shadow. The
patch is committed to the real project only on green tests. Failed attempts
are stored as episodic "attempted" edges and are never repeated.

Failure localization:
  1. Parse project "File \"...\", line N, in fname" frames when available.
  2. STATIC FALLBACK (no traceback frames): parse FAILED test ids from the
     pytest summary, follow resolved calls from those tests in the graph to
     candidate project functions, and search each candidate's source for the
     error token (e.g. "qty" in the KeyError message).
     This keeps the loop fully grounded in the symbolic world model.
"""
from __future__ import annotations

import ast
import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from nsagent.causal import CausalMemory
from nsagent.memory import RealCodeMemory
from nsagent.sandbox import ExecResult, RealSandbox


@dataclass
class PatchProposal:
    file: str
    function: str
    new_code: str
    description: str
    patch_hash: str


class EpisodicMemory:
    def __init__(self, memory: RealCodeMemory):
        self.memory = memory
        self.graph = memory.graph

    @staticmethod
    def _hash(code: str) -> str:
        return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]

    def _patch_node(self, patch_code: str, patch_hash: str) -> str:
        pid = f"patch::{patch_hash}"
        if pid not in self.graph:
            self.graph.add_node(pid, kind="patch", patch_hash=patch_hash,
                                code_preview=patch_code[:200],
                                **self.memory._next_event("episodic"))
        return pid

    def already_attempted(self, function_id: str, patch_code: str) -> bool:
        h = self._hash(patch_code)
        return any(d.get("relation") in ("attempted", "patched")
                   and u == function_id and d.get("patch_hash") == h
                   for u, v, k, d in self.graph.edges(keys=True, data=True))

    def record_attempt(self, function_id: str, patch_code: str, success: bool,
                       error: str = "", shadow_id: str = "") -> None:
        h = self._hash(patch_code)
        pid = self._patch_node(patch_code, h)
        self.graph.add_edge(function_id, pid, key="attempted", relation="attempted",
                            success=success, error=error, shadow_id=shadow_id,
                            patch_hash=h, **self.memory._next_event("episodic"))

    def record_patch_success(self, function_id: str, patch_code: str) -> None:
        h = self._hash(patch_code)
        pid = self._patch_node(patch_code, h)
        self.graph.add_edge(function_id, pid, key="patched", relation="patched",
                            patch_hash=h, **self.memory._next_event("episodic"))

    def attempts(self, function_id: str) -> list:
        return [(v, d.get("success"), d.get("error"), d.get("patch_hash"))
                for u, v, k, d in self.graph.edges(keys=True, data=True)
                if u == function_id and d.get("relation") == "attempted"]


class ProjectShadow:
    def __init__(self, project_root: str, base_dir: str = "/content/nsagent_shadow"):
        self.project_root = Path(project_root).resolve()
        self.base_dir = Path(base_dir)
        self.path: Optional[Path] = None

    def create(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        sid = f"shadow_{uuid.uuid4().hex[:8]}"
        self.path = self.base_dir / sid
        shutil.copytree(self.project_root, self.path,
                        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"))
        return self.path

    def remove(self) -> None:
        if self.path and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


class FixQtyColumn(ast.NodeTransformer):
    """Replace df['qty'] with df['quantity'] inside clean_data only."""
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.in_function = False
        self.changed = False

    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            old = self.in_function
            self.in_function = True
            self.generic_visit(node)
            self.in_function = old
            return node
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if (self.in_function and isinstance(node.slice, ast.Constant)
                and node.slice.value == "qty"):
            node.slice = ast.Constant(value="quantity")
            self.changed = True
        return node


class SelfHealer:
    def __init__(self, memory: RealCodeMemory, causal: CausalMemory,
                 episodic: EpisodicMemory,
                 project_root: str = "/content/sample_project",
                 timeout: float = 60.0):
        self.memory = memory
        self.graph = memory.graph   # bind graph for direct traversal
        self.causal = causal
        self.episodic = episodic
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout

    # ------------------------------------------------------- diagnosis
    def diagnose(self, res: ExecResult) -> dict:
        text = res.combined
        m = re.search(r"E\s+([A-Za-z_]\w*(?:Error|Exception))\s*:\s*(.*?)(?:\n|$)", text)
        if not m:
            m = re.search(r"([A-Za-z_]\w*(?:Error|Exception))\s*:\s*(.*)", text)
        error_type = m.group(1) if m else "Exception"
        message = (m.group(2).strip().strip("'\"") if m else "")

        # Preferred: project frames in a verbose traceback.
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\w+)', text)
        project_frames = []
        for f, lineno, fn in frames:
            try:
                rel = Path(f).resolve().relative_to(self.project_root).as_posix()
            except ValueError:
                continue
            if rel.endswith(".py"):
                project_frames.append((rel, int(lineno), fn))

        if project_frames:
            for rel, lineno, fn in reversed(project_frames):
                candidate = f"{rel}::{fn}"
                if candidate in self.memory.graph:
                    return {"error_type": error_type, "message": message,
                            "frames": project_frames,
                            "function_id": candidate, "localized_by": "traceback"}

        # Static fallback: graph call paths from failed tests + source token search.
        function_id = self._localize_by_graph_and_source(error_type, message, text)
        return {"error_type": error_type, "message": message,
                "frames": project_frames, "function_id": function_id,
                "localized_by": "static-graph-source"}

    def _parse_failed_test_ids(self, text: str) -> List[str]:
        ids: List[str] = []
        for m in re.finditer(r"FAILED\s+(\S+)", text):
            test_id = m.group(1).strip()
            # Normalize "tests/test_pipeline.py::test_x -" -> remove trailing '-'
            test_id = test_id.rstrip("-").strip()
            if "::" in test_id and test_id not in ids:
                ids.append(test_id)
        return ids

    def _reachable_project_functions(self, test_id: str) -> List[str]:
        """BFS through resolved calls from a test function."""
        out: List[str] = []
        start = test_id
        if start not in self.graph:
            return out
        queue = [start]
        seen = set()
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if self.graph.nodes[node].get("kind") == "function":
                if node not in out:
                    out.append(node)
            for u, v, k, d in self.graph.edges(node, keys=True, data=True):
                if d.get("relation") == "calls" and d.get("resolved") and v not in seen:
                    queue.append(v)
        return out

    def _localize_by_graph_and_source(self, error_type: str, message: str,
                                      text: str) -> Optional[str]:
        test_ids = self._parse_failed_test_ids(text)
        if not test_ids:
            # If summary doesn't show FAILED ids, use any test node as a seed.
            test_ids = [nid for nid, d in self.graph.nodes(data=True)
                        if d.get("kind") == "function" and nid.startswith("tests/")]

        candidates: List[str] = []
        for test_id in test_ids:
            for fn in self._reachable_project_functions(test_id):
                # Exclude test functions themselves; we want the project code
                # that the test invokes, not the test node.
                if fn.startswith("tests/"):
                    continue
                if fn not in candidates:
                    candidates.append(fn)

        if not candidates:
            # Fallback: scan all project function nodes for the error token.
            token = message.strip().strip("'\"")
            for nid, data in self.graph.nodes(data=True):
                if data.get("kind") != "function" or nid.startswith("tests/"):
                    continue
                file_part = data.get("file", "")
                if file_part.endswith(".py"):
                    src_path = self.project_root / file_part
                    if src_path.is_file():
                        source = src_path.read_text(encoding="utf-8")
                        if token and token in source:
                            candidates.append(nid)
            if not candidates:
                return None

        token = message.strip().strip("'\"")
        best_fn, best_score = None, -1
        for fn in candidates:
            file_part, _, _ = fn.partition("::")
            src_path = self.project_root / file_part
            if not src_path.is_file():
                continue
            source = src_path.read_text(encoding="utf-8")
            score = 0
            if token and token in source:
                score += 10
            if "quantity" in source or "qty" in source:
                score += 4
            if error_type and error_type.lower() in source.lower():
                score += 1
            if score > best_score:
                best_fn, best_score = fn, score

        return best_fn if best_score >= 4 else None

    # ------------------------------------------------------- patch proposal
    def propose_patch(self, function_id: str, error_type: str,
                      message: str) -> Optional[PatchProposal]:
        if error_type != "KeyError" or "qty" not in message:
            return None
        file_part, _, fname = function_id.partition("::")
        src_path = self.project_root / file_part
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        tx = FixQtyColumn(fname)
        new_tree = tx.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_code = ast.unparse(new_tree)
        if not tx.changed:
            return None
        h = hashlib.sha1(new_code.encode("utf-8")).hexdigest()[:12]
        return PatchProposal(file=file_part, function=fname, new_code=new_code,
                             description="replace column 'qty' with 'quantity' in clean_data",
                             patch_hash=h)

    # ------------------------------------------------------- healing loop
    def heal_from_failure(self, res: ExecResult) -> bool:
        print("  🩺 [SelfHealer] Diagnosing failure...")
        diag = self.diagnose(res)
        print(f"     diag = {{'error_type': {diag['error_type']!r}, "
              f"'message': {diag['message']!r}, 'function_id': {diag['function_id']!r}, "
              f"'localized_by': {diag['localized_by']!r}}}")
        function_id = diag["function_id"]
        if not function_id:
            print("  ❌ [SelfHealer] Could not localize failure to a project function.")
            return False

        self.causal.record_failure(function_id, diag["error_type"], diag["message"],
                                   traceback_text=res.combined)
        print(f"  🧠 [CausalMemory] recorded failure: {function_id} -[raised]-> "
              f"{diag['error_type']} ('{diag['message']}')")

        proposal = self.propose_patch(function_id, diag["error_type"], diag["message"])
        if not proposal:
            print("  ❌ [SelfHealer] No valid patch proposal found for this failure.")
            return False

        if self.episodic.already_attempted(function_id, proposal.new_code):
            print(f"  🚫 [EpisodicMemory] Patch {proposal.patch_hash} already attempted; "
                  f"refusing to repeat a failed patch.")
            return False

        print(f"  🛠️  [SelfHealer] Proposed patch: {proposal.description}")
        print(f"       patch_hash={proposal.patch_hash}")

        shadow = ProjectShadow(str(self.project_root))
        shadow_path = shadow.create()
        print(f"  📁 [Shadow] Created shadow project: {shadow_path}")

        (shadow_path / proposal.file).write_text(proposal.new_code, encoding="utf-8")

        shadow_sandbox = RealSandbox(project_root=str(shadow_path), timeout=self.timeout)
        test_res = shadow_sandbox.run_pytest(args=["tests", "-q"], timeout=self.timeout)
        print(f"  🧪 [ShadowTests] {test_res.summary()}")

        if not test_res.success:
            self.episodic.record_attempt(function_id, proposal.new_code, success=False,
                                         error=test_res.tail(500), shadow_id=str(shadow_path))
            print("  ↩️  [SelfHealer] Shadow tests RED. Rolling back and storing failed attempt.")
            shadow.remove()
            return False

        (self.project_root / proposal.file).write_text(proposal.new_code, encoding="utf-8")
        print(f"  ✅ [SelfHealer] Shadow tests GREEN. Committing patch to "
              f"{self.project_root / proposal.file}")

        # Re-ingest the real project so the graph reflects the patched source.
        self.memory.ingest_directory(self.project_root)

        self.episodic.record_attempt(function_id, proposal.new_code, success=True,
                                     shadow_id=str(shadow_path))
        self.episodic.record_patch_success(function_id, proposal.new_code)
        shadow.remove()
        print(f"  🎉 [SelfHealer] Patch committed and graph updated.")
        return True
