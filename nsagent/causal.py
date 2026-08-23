
"""
nsagent.causal — CausalMemory: raised / thrown_by edges + failure traces.

Runtime failures are stored as typed edges with event IDs, timestamps, and
captured tracebacks. "Why did X fail?" is answered by backward/forward
traversal over the graph.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

from nsagent.memory import RealCodeMemory


class CausalMemory:
    def __init__(self, memory: RealCodeMemory):
        self.memory = memory
        self.graph = memory.graph

    def _error_node(self, error_type: str, message: str) -> str:
        h = hashlib.sha1(f"{error_type}:{message}".encode("utf-8")).hexdigest()[:10]
        eid = f"error::{error_type}::{h}"
        if eid not in self.graph:
            self.graph.add_node(eid, kind="error", error_type=error_type, msg=message,
                                **self.memory._next_event("runtime"))
        return eid

    def record_failure(self, function_id: str, error_type: str, message: str,
                       traceback_text: str = "") -> str:
        err = self._error_node(error_type, message)
        now = self.memory._next_event("runtime")
        self.graph.add_edge(function_id, err, key="raised", relation="raised",
                            error_type=error_type, msg=message,
                            traceback=traceback_text[-800:], **now)
        self.graph.add_edge(err, function_id, key="thrown_by", relation="thrown_by",
                            **self.memory._next_event("runtime"))
        return err

    def failure_of(self, function_id: str) -> List[Tuple[str, str]]:
        return [(v, self.graph.nodes[v].get("msg", ""))
                for u, v, k, d in self.graph.edges(keys=True, data=True)
                if u == function_id and d.get("relation") == "raised"]

    def trace_failure(self, start: str, max_depth: int = 20) -> Tuple[List[str], Optional[str], Optional[str]]:
        """BFS through resolved calls from `start` looking for a raised edge."""
        queue = [(start, [start])]
        seen = set()
        while queue:
            node, path = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            for u, v, k, d in self.graph.edges(node, keys=True, data=True):
                if d.get("relation") == "raised" and not d.get("external"):
                    msg = self.graph.nodes[v].get("msg", "")
                    return path, v, msg
            for u, v, k, d in self.graph.edges(node, keys=True, data=True):
                if d.get("relation") == "calls" and d.get("resolved") and v not in seen:
                    if len(path) < max_depth:
                        queue.append((v, path + [v]))
        return path, None, None

    def print_trace(self, start: str) -> None:
        path, err, msg = self.trace_failure(start)
        if err is None:
            print(f"  [Causal] No runtime failure recorded reachable from '{start}'")
            return
        print(f"  [Causal] Why did '{start}' fail?")
        print(f"    path : {' -> '.join(path)}")
        print(f"    error: {err}")
        print(f"    msg  : {msg}")
        throwers = [v for u, v, k, d in self.graph.edges(err, keys=True, data=True)
                    if d.get("relation") == "thrown_by"]
        print(f"    thrown_by: {throwers}")
