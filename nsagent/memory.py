"""
nsagent.memory — Symbolic World Model (RealCodeMemory).

Ingests a real project directory of .py files via `ast` into a typed
NetworkX MultiDiGraph; persists to JSON and reloads with verification.

Invariants
----------
* The graph is the SOURCE OF TRUTH. This module NEVER executes project code.
* Node ids are file-anchored and fully qualified ("cleaner.py::clean_data");
  a `name` attribute carries the bare symbol for NL grounding.
* Every node/edge carries provenance (file, lineno), an event_id and a UTC
  timestamp — the substrate for episodic memory and causal traces.
* Edge key == relation, so (u, v, relation) is unique: re-ingesting a file
  MERGES provenance instead of duplicating edges.
* Cross-module symbol resolution runs in a second pass, so
  "from cleaner import clean_data" grounds call edges to the real node.

Accepted limitations: calls in decorators of nested defs, imports inside
`if` blocks, and assigns inside nested statement bodies are not recorded
(none occur in the sample project).
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

GRAPH_FORMAT = "nsagent-graph-v1"
CORE_RELATIONS = ("defines", "calls", "imports", "assigns")
# Reserved relations for later blocks (causal memory / episodes / skills):
FUTURE_RELATIONS = ("raised", "thrown_by", "attempted", "patched", "skill", "episode")

_KIND_PRIORITY = {"function": 0, "class": 1, "module": 2, "variable": 3}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short(s: str, w: int = 54) -> str:
    return s if len(s) <= w else s[: w - 1] + "…"


def _node_link_data(g):
    try:
        return nx.node_link_data(g, edges="links")
    except TypeError:  # networkx < 3.4
        return nx.node_link_data(g)


def _node_link_graph(d):
    try:
        return nx.node_link_graph(d, edges="links")
    except TypeError:  # networkx < 3.4
        return nx.node_link_graph(d)


class RealCodeMemory:
    """The codebase as an inspectable, editable, persistable graph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.event_counter = 0
        self.meta: Dict[str, Any] = {"format": GRAPH_FORMAT, "files": {}, "project_root": None}
        self._defs_by_module: Dict[str, Dict[str, str]] = {}  # ingestion-time index

    # ------------------------------------------------------------- event log
    def _next_event(self, origin: str) -> Dict[str, str]:
        self.event_counter += 1
        return {"event_id": f"evt_{self.event_counter:06d}", "ts": _utcnow(), "origin": origin}

    # ----------------------------------------------------------------- nodes
    def _node(self, node_id: str, kind: str, name: str, file=None, lineno=None,
              origin: str = "ingest", **attrs) -> bool:
        if node_id in self.graph:
            return False
        data = {"kind": kind, "name": name, "file": file, "lineno": lineno,
                **self._next_event(origin)}
        data.update(attrs)
        self.graph.add_node(node_id, **data)
        return True

    # ----------------------------------------------------------------- edges
    def _edge(self, u: str, v: str, relation: str, file=None, lineno=None,
              origin: str = "ingest", **attrs) -> bool:
        for n in (u, v):
            if n not in self.graph:
                self.graph.add_node(n, kind="external",
                                    name=str(n).split("::")[-1].split(".")[-1],
                                    **self._next_event(origin))
        loc = f"{file}:{lineno}" if (file is not None and lineno is not None) else str(file or "?")
        if self.graph.has_edge(u, v, key=relation):
            d = self.graph[u][v][relation]
            if loc not in d.get("locs", []):
                d.setdefault("locs", []).append(loc)
            d["count"] = d.get("count", 1) + 1
            d["last_ts"] = _utcnow()
            return False
        data = {"relation": relation, "file": file, "lineno": lineno, "locs": [loc],
                "count": 1, **self._next_event(origin)}
        data.update(attrs)
        self.graph.add_edge(u, v, key=relation, **data)
        return True

    # ------------------------------------------------------------- ingestion
    def ingest_directory(self, root, exclude=("venv", ".venv", "__pycache__",
                                              ".git", "node_modules", "agent_state")) -> "RealCodeMemory":
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"project root not found: {root}")
        self.meta["project_root"] = str(root)
        py_files = sorted(p for p in root.rglob("*.py")
                          if not any(part in exclude for part in p.relative_to(root).parts))
        if not py_files:
            raise FileNotFoundError(f"no .py files under {root}")
        print(f"[Memory] ingesting {len(py_files)} python file(s) from {root}")
        pending: List[Tuple] = []
        for path in py_files:
            rel = path.relative_to(root).as_posix()
            src = path.read_text(encoding="utf-8")
            self.meta["files"][rel] = {
                "sha256": hashlib.sha256(src.encode("utf-8")).hexdigest()[:12],
                "lines": len(src.splitlines()),
                "ingested_at": _utcnow(),
            }
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                self._node(rel, "module", rel, file=rel, parse_error=str(exc))
                print(f"  [!] {rel}: SYNTAX ERROR recorded: {exc}")
                continue
            self._node(rel, "module", rel, file=rel)
            symtab = self._module_symbols(tree)
            self._walk_scope(tree, module_id=rel, scope=rel, symtab=symtab, pending=pending)
            print(f"  [+] {rel:<26} lines={self.meta[chr(39)+chr(39)] if False else self.meta['files'][rel]['lines']:<3} "
                  f"defs={len(self._defs_by_module.get(rel, {}))}")
        stats = self._resolve_pending(pending)
        print(f"[Memory] symbol resolution: calls {stats['resolved_calls']} resolved / "
              f"{stats['external_calls']} external | imports {stats['resolved_imports']} resolved / "
              f"{stats['external_imports']} external")
        return self

    def _module_symbols(self, tree) -> Dict[str, tuple]:
        syms: Dict[str, tuple] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    syms[a.asname or a.name.split(".")[0]] = ("module", a.name)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    syms[a.asname or a.name] = ("from", node.module or "", a.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                syms[node.name] = ("local", node.name)
        return syms

    def _walk_scope(self, node, module_id: str, scope: str, symtab: dict, pending: list) -> None:
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"{scope}::{stmt.name}"
                params = [a.arg for a in stmt.args.args]
                self._node(fid, "function", stmt.name, file=module_id, lineno=stmt.lineno,
                           doc=ast.get_docstring(stmt), params=params)
                self._edge(scope, fid, "defines", file=module_id, lineno=stmt.lineno)
                self._defs_by_module.setdefault(module_id, {})[stmt.name] = fid
                self._walk_scope(stmt, module_id, fid, symtab, pending)
            elif isinstance(stmt, ast.ClassDef):
                cid = f"{scope}::{stmt.name}"
                self._node(cid, "class", stmt.name, file=module_id, lineno=stmt.lineno,
                           doc=ast.get_docstring(stmt))
                self._edge(scope, cid, "defines", file=module_id, lineno=stmt.lineno)
                self._defs_by_module.setdefault(module_id, {})[stmt.name] = cid
                self._walk_scope(stmt, module_id, cid, symtab, pending)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                pending.append(("import", scope, stmt, module_id, symtab))
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                self._handle_assign(stmt, scope, module_id)
                for call in self._calls_in_stmt(stmt):
                    pending.append(("call", scope, call, module_id, symtab))
            else:
                for call in self._calls_in_stmt(stmt):
                    pending.append(("call", scope, call, module_id, symtab))

    def _handle_assign(self, stmt, scope: str, module_id: str) -> None:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        else:  # AnnAssign / AugAssign
            targets, value = [stmt.target], stmt.value
        literal: Any = None
        if isinstance(value, ast.Constant):
            literal = value.value
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            literal = "<composite-literal>"
        for t in targets:
            for nm in ast.walk(t):
                if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Store):
                    vid = f"{scope}::{nm.id}"
                    self._node(vid, "variable", nm.id, file=module_id, lineno=stmt.lineno,
                               literal=literal,
                               scope_kind="module" if scope == module_id else "local")
                    self._edge(scope, vid, "assigns", file=module_id, lineno=stmt.lineno)

    def _calls_in_stmt(self, stmt) -> List[ast.Call]:
        calls: List[ast.Call] = []

        def rec(n):
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef, ast.Lambda)):
                    continue  # attributed to their own scope
                if isinstance(child, ast.Call):
                    calls.append(child)
                rec(child)

        rec(stmt)
        return calls

    # ---------------------------------------------------- pass B: resolution
    def _resolve_pending(self, pending: list) -> Dict[str, int]:
        stats = {"resolved_calls": 0, "external_calls": 0,
                 "resolved_imports": 0, "external_imports": 0}
        for kind, scope, node, module_id, symtab in pending:
            if kind == "import":
                out = self._resolve_import(scope, node, module_id)
                stats["resolved_imports"] += out["resolved_imports"]
                stats["external_imports"] += out["external_imports"]
            else:
                ok = self._resolve_call(scope, node, module_id, symtab)
                stats["resolved_calls" if ok else "external_calls"] += 1
        return stats

    def _is_module_node(self, nid: str) -> bool:
        return nid in self.graph and self.graph.nodes[nid].get("kind") == "module"

    def _file_for_module(self, mod: str) -> str:
        rel = mod.replace(".", "/")
        for cand in (rel + ".py", rel + "/__init__.py"):
            if self._is_module_node(cand):
                return cand
        return rel + ".py"

    def _resolve_import(self, scope: str, node, module_id: str) -> Dict[str, int]:
        out = {"resolved_imports": 0, "external_imports": 0}
        if isinstance(node, ast.Import):
            for a in node.names:
                target = self._file_for_module(a.name)
                if self._is_module_node(target):
                    self._edge(scope, target, "imports", file=module_id, lineno=node.lineno,
                               resolved=True, imported=a.name, alias=a.asname)
                    out["resolved_imports"] += 1
                else:
                    self._edge(scope, a.name, "imports", file=module_id, lineno=node.lineno,
                               resolved=False, external=True, imported=a.name, alias=a.asname)
                    out["external_imports"] += 1
        else:  # ImportFrom
            mod = node.module or "?"
            target_file = self._file_for_module(mod)
            if self._is_module_node(target_file):
                self._edge(scope, target_file, "imports", file=module_id, lineno=node.lineno,
                           resolved=True, imported=mod, kind="module")
                out["resolved_imports"] += 1
            for a in node.names:
                q = self._defs_by_module.get(target_file, {}).get(a.name)
                if q:
                    self._edge(scope, q, "imports", file=module_id, lineno=node.lineno,
                               resolved=True, imported=f"{mod}.{a.name}", alias=a.asname)
                    out["resolved_imports"] += 1
                else:
                    self._edge(scope, f"{mod}.{a.name}", "imports", file=module_id,
                               lineno=node.lineno, resolved=False, external=True,
                               imported=f"{mod}.{a.name}")
                    out["external_imports"] += 1
        return out

    def _resolve_call(self, scope: str, call: ast.Call, module_id: str, symtab: dict) -> bool:
        target, resolved = self._callee_target(call, module_id, symtab)
        if resolved:
            self._edge(scope, target, "calls", file=module_id, lineno=call.lineno, resolved=True)
            return True
        self._edge(scope, target, "calls", file=module_id, lineno=call.lineno,
                   resolved=False, external=True)
        return False

    def _callee_target(self, call: ast.Call, module_id: str, symtab: dict) -> Tuple[str, bool]:
        f = call.func
        if isinstance(f, ast.Name):
            info = symtab.get(f.id)
            if info is None:
                return f.id, False  # builtin / undefined in module scope
            if info[0] == "local":
                q = self._defs_by_module.get(module_id, {}).get(f.id)
                return (q, True) if q else (f.id, False)
            if info[0] == "from":
                _, mod, orig = info
                q = self._defs_by_module.get(self._file_for_module(mod), {}).get(orig)
                return (q, True) if q else (f"{mod}.{orig}", False)
            return f.id, False
        if isinstance(f, ast.Attribute):
            return self._dotted(f, symtab), False
        return "<callable-expr>", False

    def _dotted(self, attr_node: ast.Attribute, symtab: dict) -> str:
        parts: List[str] = []
        node = attr_node
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            dotted = ".".join(reversed(parts))
            info = symtab.get(node.id)
            base = info[1] if (info and info[0] == "module") else node.id
            return f"{base}.{dotted}"
        return parts[0] if parts else "<expr>"  # chained call: keep outermost attr

    # ------------------------------------------------------- grounding query API
    def resolve(self, symbol: Optional[str]) -> List[str]:
        """Bare symbol -> qualified node ids (grounded, non-external entities)."""
        if not symbol:
            return []
        symbol = symbol.strip()
        if symbol in self.graph and self.graph.nodes[symbol].get("kind") != "external":
            return [symbol]
        hits = [nid for nid, d in self.graph.nodes(data=True)
                if d.get("name") == symbol and d.get("kind") != "external"]
        hits.sort(key=lambda nid: _KIND_PRIORITY.get(self.graph.nodes[nid].get("kind"), 9))
        return hits

    def ground(self, symbol: Optional[str]) -> Optional[str]:
        hits = self.resolve(symbol)
        return hits[0] if hits else None

    def query(self, subject=None, relation=None, obj=None):
        def match(node_id, want):
            if want is None:
                return True
            d = self.graph.nodes.get(node_id, {})
            return node_id == want or d.get("name") == want

        return [(u, d.get("relation"), v) for u, v, k, d in self.graph.edges(keys=True, data=True)
                if (relation is None or d.get("relation") == relation)
                and match(u, subject) and match(v, obj)]

    def callees_of(self, name: str, resolved_only: bool = True) -> List[str]:
        return [v for u, v, k, d in self.graph.edges(keys=True, data=True)
                if d.get("relation") == "calls"
                and (u == name or self.graph.nodes[u].get("name") == name)
                and (not resolved_only or d.get("resolved"))]

    def callers_of(self, name: str, resolved_only: bool = True) -> List[str]:
        return [u for u, v, k, d in self.graph.edges(keys=True, data=True)
                if d.get("relation") == "calls"
                and (v == name or self.graph.nodes[v].get("name") == name)
                and (not resolved_only or d.get("resolved"))]

    def find_data_files(self, suffixes=(".csv", ".json", ".xlsx", ".parquet", ".tsv", ".txt")):
        out = []
        root = self.meta.get("project_root")
        if root:
            for p in sorted(Path(root).rglob("*")):
                if p.is_file() and p.suffix.lower() in suffixes:
                    out.append(("disk", p.relative_to(root).as_posix(), "<file on disk>"))
        for nid, d in self.graph.nodes(data=True):
            if d.get("kind") == "variable" and isinstance(d.get("literal"), str)                     and Path(d["literal"]).suffix.lower() in suffixes:
                out.append(("literal", d["literal"], nid))
        return out

    # --------------------------------------------------------- stats + persist
    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        rels: Dict[str, int] = {}
        for _, _, _, d in self.graph.edges(keys=True, data=True):
            r = d.get("relation", "?")
            rels[r] = rels.get(r, 0) + 1
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "node_kinds": kinds, "edge_relations": rels}

    def fingerprint(self) -> str:
        items = sorted((u, v, k, d.get("relation"), d.get("file"), d.get("lineno"),
                        tuple(sorted(d.get("locs", []))))
                       for u, v, k, d in self.graph.edges(keys=True, data=True))
        nodes = tuple(sorted((nid, d.get("kind"), d.get("name"))
                             for nid, d in self.graph.nodes(data=True)))
        h = hashlib.sha256()
        h.update(repr(nodes).encode("utf-8"))
        h.update(repr(items).encode("utf-8"))
        return h.hexdigest()[:16]

    def save(self, path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": {**self.meta, "event_counter": self.event_counter,
                            "saved_at": _utcnow(), "fingerprint": self.fingerprint()},
                   "graph": _node_link_data(self.graph)}
        path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"[Memory] persisted -> {path}  ({path.stat().st_size:,} bytes, "
              f"fingerprint {payload['meta']['fingerprint']})")
        return str(path)

    def load(self, path) -> "RealCodeMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = payload.get("meta", {})
        self.event_counter = self.meta.get("event_counter", 0)
        self.graph = _node_link_graph(payload["graph"])
        print(f"[Memory] loaded <- {path}")
        return self

    # ----------------------------------------------------------------- dump
    def dump(self, include_external: bool = True) -> None:
        s = self.stats()
        bar = "=" * 78
        print(bar)
        print("SYMBOLIC WORLD MODEL — RealCodeMemory")
        print(f"  project_root : {self.meta.get('project_root')}")
        print(f"  nodes        : {s['nodes']}  {s['node_kinds']}")
        print(f"  edges        : {s['edges']}  {s['edge_relations']}")
        print(f"  events       : evt_{self.event_counter:06d} (every node/edge stamped)")
        print(bar)
        for rel in CORE_RELATIONS:
            edges = [(u, v, d) for u, v, k, d in self.graph.edges(keys=True, data=True)
                     if d.get("relation") == rel and not d.get("external")]
            print(f"\n--- {rel} (project-grounded): {len(edges)} edge(s) ---")
            for u, v, d in sorted(edges, key=lambda e: (e[0], e[1])):
                locs = ",".join(d.get("locs", []))
                cnt = f"  x{d['count']}" if d.get("count", 1) > 1 else ""
                print(f"  {_short(u):<54} -[{rel}]-> {_short(v):<54} @{locs}{cnt}")
        if include_external:
            ext: Dict[Tuple[str, str], List[str]] = {}
            for u, v, k, d in self.graph.edges(keys=True, data=True):
                if d.get("external"):
                    ext.setdefault((d.get("relation", "?"), v), []).append(u)
            print(f"\n--- external surface (edges leaving the project): {len(ext)} target(s) ---")
            for (rel, v), us in sorted(ext.items()):
                callers = sorted({u.split("::")[-1] for u in us})
                tail = f" (+{len(callers) - 5} more)" if len(callers) > 5 else ""
                print(f"  [{rel}] {_short(v, 26):<26} <- {', '.join(callers[:5])}{tail}")
        print(bar)
