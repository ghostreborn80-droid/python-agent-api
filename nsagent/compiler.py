
"""
nsagent.compiler — IntentCompiler: NL -> typed, graph-grounded PlanSpec.

Neural layer (spaCy) proposes intent and entities; the graph is the
disposing authority. Any plan whose entities are not grounded in the
world model is refused with a clarification question.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from nsagent.plan import PlanSpec
from nsagent.memory import RealCodeMemory


class IntentCompiler:
    EXEC_VERBS = {"run", "execute", "generate", "produce", "make", "build",
                  "create", "clean", "fix", "patch", "repair", "test"}
    EXPLAIN_VERBS = {"explain", "describe", "show", "trace", "summarize",
                     "inspect", "list", "display"}
    CALL_VERBS = {"call", "use", "invoke", "depend"}
    FILE_ARG_PAT = re.compile(r"\b[\w-]+\.(?:csv|json|txt|xlsx|parquet|tsv|html)\b")

    def __init__(self, nlp, memory: RealCodeMemory):
        self.nlp = nlp
        self.memory = memory

    def _norm(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _tokens(self, text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def _split_parts(self, s: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9_]+", s.lower()))

    def compile(self, request: str) -> PlanSpec:
        doc = self.nlp(self._norm(request))
        text = self._norm(request)
        tokens = self._tokens(text)

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        root_lemma = root.lemma_.lower() if root is not None else ""
        first = doc[0].text.lower() if len(doc) else ""

        if first == "who" and any(t.lemma_ in self.CALL_VERBS for t in doc):
            intent = "inspect_callers"
        elif first == "what" and any(t.lemma_ in self.CALL_VERBS for t in doc):
            intent = "inspect_callees"
        elif root_lemma in self.EXEC_VERBS:
            intent = "execute"
        elif root_lemma in self.EXPLAIN_VERBS or root_lemma in self.CALL_VERBS:
            intent = "explain" if root_lemma in self.EXPLAIN_VERBS else "unknown"
        else:
            intent = "unknown"

        args = self.FILE_ARG_PAT.findall(text)
        args = list(dict.fromkeys(args))

        # ----------------------------------------------------------------
        # General Python task override: these are grounded by the knowledge
        # engine, not by project graph entities.
        # ----------------------------------------------------------------
        if self._is_python_generation_request(text):
            target = "tool::generate_python_script"
            if target in self.memory.graph:
                return PlanSpec(
                    request=request, intent="generate_python_script",
                    target=target, target_name="generate_python_script",
                    args=args, grounded=True,
                    note="grounded to offline Python script generation tool",
                    tool="generate_python_script",
                    tool_params={"task": request},
                )

        if self._is_python_knowledge_request(text):
            target = "tool::python_knowledge"
            if target in self.memory.graph:
                return PlanSpec(
                    request=request, intent="python_knowledge",
                    target=target, target_name="python_knowledge",
                    args=args, grounded=True,
                    note="grounded to offline Python knowledge retrieval tool",
                    tool="python_knowledge",
                    tool_params={"question": request},
                )

        # ---------------------------------------------------------------- 
        # Graph-disposes guard: if the user explicitly names an entity that
        # is not in the graph, REFUSE before any scoring can co-ground it.
        # ----------------------------------------------------------------
        if intent == "execute":
            unknown = self._unknown_entity_candidates(doc, text, args)
            if unknown:
                return PlanSpec(
                    request=request, intent=intent, target=None, target_name=None,
                    args=args, grounded=False,
                    note=f"Unknown entity candidate(s): {', '.join(unknown)}",
                    clarification=(
                        f"I don't recognize {', '.join(unknown)} in the world model. "
                        f"Did you mean one of the known functions or files?"
                    ),
                )

        target, target_name, score = self._resolve_target(tokens, intent)

        grounded = target is not None and target in self.memory.graph

        note = ""
        clarification = None
        if not target:
            note = "No project entity matching the request could be found."
            clarification = (f"I could not ground '{text}' in the world model. "
                             f"Which symbol did you mean, or should I inspect the graph first?")
        elif not grounded:
            note = f"Entity '{target}' not in world model"
            clarification = f"I resolved '{target_name}' but it is not present in the graph."

        spec = PlanSpec(request=request, intent=intent, target=target,
                        target_name=target_name, args=args, grounded=grounded,
                        note=note, clarification=clarification)

        if grounded:
            tool_call = spec.to_tool_call()
            if tool_call:
                spec.tool, spec.tool_params = tool_call
        return spec

    # -------------------------------------------------------------- unknown
    def _unknown_entity_candidates(self, doc, text: str, file_args: List[str]) -> List[str]:
        """Candidate nouns/proper-nouns that are not graph-grounded and are
        not data-file arguments. These are likely explicit unknown entities."""
        unknown: List[str] = []
        for tok in doc:
            word = tok.text.lower()
            if tok.is_stop or tok.pos_ not in ("NOUN", "PROPN"):
                continue
            if any(word in a.lower() or a.lower() in word for a in file_args):
                continue  # e.g. "sales.csv"
            if not self._known_entity_word(word):
                unknown.append(word)
        return list(dict.fromkeys(unknown))

    def _known_entity_word(self, word: str) -> bool:
        if self.memory.resolve(word):
            return True
        # Also recognize file-base names ("report" -> "report.py")
        for nid, data in self.memory.graph.nodes(data=True):
            if data.get("external"):
                continue
            parts = self._split_parts(nid)
            if word in parts:
                return True
        return False

    # ---------------------------------------------------- python detection
    def _is_python_generation_request(self, text: str) -> bool:
        return bool(re.search(r"\b(write|generate|create)\b.*\bpython\b.*\b(script|program|function|code)\b", text, re.I))

    def _is_python_knowledge_request(self, text: str) -> bool:
        return bool(re.search(r"\b(how|what|why)\b.*\bpython\b", text, re.I))

    # ------------------------------------------------------------- resolve
    def _resolve_target(self, tokens: Set[str], intent: str) -> Tuple[Optional[str], Optional[str], int]:
        best_id, best_name, best_score = None, None, 0
        for nid, data in self.memory.graph.nodes(data=True):
            if data.get("kind") not in ("function", "class", "module"):
                continue
            if data.get("external"):
                continue
            s = self._score_node(nid, data, tokens, intent)
            if s > best_score:
                best_id, best_name, best_score = nid, data.get("name"), s

        if best_id and self.memory.graph.nodes[best_id].get("kind") == "module":
            module_file = best_id
            for nid, data in self.memory.graph.nodes(data=True):
                if data.get("kind") == "function" and data.get("file") == module_file:
                    s = self._score_node(nid, data, tokens, intent) + 4
                    if s > best_score:
                        best_id, best_name, best_score = nid, data.get("name"), s

        if best_score < 5:
            return None, None, best_score
        return best_id, best_name, best_score

    def _score_node(self, nid: str, data: dict, tokens: Set[str], intent: str) -> int:
        name = data.get("name", "")
        qualified = data.get("qualified_name", "") or nid
        doc = (data.get("doc") or "").lower()
        score = 0
        if name.lower() in tokens:
            score += 12
        score += 8 * len(self._split_parts(qualified) & tokens)
        score += 3 * len(self._tokens(doc) & tokens)
        if intent == "execute":
            if data.get("kind") == "function":
                score += 4
            elif data.get("kind") == "class":
                score += 1
            elif data.get("kind") == "module":
                score -= 6
        return score
