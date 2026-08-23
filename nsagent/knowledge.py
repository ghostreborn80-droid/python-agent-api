
"""
nsagent.knowledge — Offline Python knowledge retrieval from MBPP.

Indexes cleaned MBPP descriptions AND code tokens so queries like
"join two strings with delimiter" match examples whose code uses join/join
even when the description uses "concatenate".

This is an offline neural-retrieval layer. It never executes code.
"""
from __future__ import annotations

import ast
import json
import warnings
import re
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore", category=SyntaxWarning)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _code_keywords(code: str) -> str:
    """Extract identifier-ish tokens from Python code for retrieval."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is None:
        return re.sub(r"[^a-zA-Z0-9_]+", " ", code).lower()

    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
    return " ".join(names).lower()


class PythonKnowledgeRetriever:
    """Retrieve Python examples from a cleaned MBPP JSONL corpus."""

    def __init__(self, data_path: str = "/content/agent_data/cleaned_mbpp.jsonl"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Cleaned MBPP data not found: {self.data_path}")

        self.examples: List[Dict[str, Any]] = []
        self.search_texts: List[str] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words="english",
            sublinear_tf=True,
        )
        self._load()
        self._fit_index()

    def _load(self) -> None:
        self.examples = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text_norm = entry.get("text_norm", "")
                code_kw = _code_keywords(entry.get("code", ""))
                test_kw = self._tests_keywords(entry)
                # Description appears twice to bias ranking toward task intent.
                self.search_texts.append(f"{text_norm} {text_norm} {code_kw} {test_kw}")
                self.examples.append(entry)

        if not self.examples:
            raise RuntimeError(f"No valid examples found in {self.data_path}")
        print(f"[Knowledge] loaded {len(self.examples)} MBPP examples from {self.data_path}")

    @staticmethod
    def _tests_keywords(entry: Dict[str, Any]) -> str:
        parts = []
        for key in ("tests", "challenge_tests"):
            val = entry.get(key) or []
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, str):
                        parts.append(t)
        setup = entry.get("test_setup") or ""
        if setup:
            parts.append(setup)
        return " ".join(parts).lower()

    def _fit_index(self) -> None:
        self.matrix = self.vectorizer.fit_transform(self.search_texts)
        print(f"[Knowledge] TF-IDF index built: vocabulary_size={len(self.vectorizer.vocabulary_)}, "
              f"matrix_shape={self.matrix.shape}")

    @staticmethod
    def _normalize_query(query: str) -> str:
        query = query.lower().strip()

        # Remove boilerplate so meaningful terms dominate.
        query = re.sub(
            r"^(write|generate|create)\s+a\s+python\s+(script|program|function|code)\s+(to\s+)?",
            "",
            query,
        )
        query = re.sub(r"\bhow\s+do\s+i\s+", "", query)
        query = re.sub(r"\bin\s+python\b", "", query)
        query = re.sub(r"[^a-z0-9\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()

        # Query expansion for common synonyms and intent patterns.
        expansions = []
        if "join" in query:
            expansions.append("concatenate")
            if "string" in query or "delimiter" in query:
                expansions.append("concatenate_tuple concatenate each element tuple delimiter")
        if "concatenate" in query:
            expansions.append("join")
        if "check if" in query and "prime" in query:
            expansions.append("prime_num")
        if "delimiter" in query:
            expansions.append("concatenate_tuple tuple delimiter concatenate each element")
        if "sort" in query and "dictionar" in query:
            expansions.append("sorted_models")
        if expansions:
            query = query + " " + " ".join(expansions)
        return query

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self._normalize_query(query)
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        # Lexical overlap bonus: direct word overlap between query and
        # description + code + tests. This fixes cases where TF-IDF prefers
        # a related but wrong example.
        q_tokens = set(q.split())
        overlap_scores = np.zeros_like(sims)
        for i, doc_text in enumerate(self.search_texts):
            doc_tokens = set(doc_text.split())
            if not q_tokens:
                continue
            overlap = len(q_tokens & doc_tokens) / len(q_tokens)
            overlap_scores[i] = overlap

        # Hybrid score: TF-IDF cosine (0.65) + lexical overlap (0.35)
        hybrid = 0.65 * sims + 0.35 * overlap_scores

        k = min(top_k, len(hybrid))
        if k <= 0:
            return []

        top_indices = np.argsort(hybrid)[::-1][:k]

        results = []
        for idx in top_indices:
            ex = self.examples[idx]
            results.append({
                "id": ex.get("id"),
                "task_id": ex.get("task_id"),
                "text": ex.get("text"),
                "func_name": ex.get("func_name"),
                "args": ex.get("args", []),
                "code": ex.get("code"),
                "similarity": float(hybrid[idx]),
                "tests": ex.get("tests", []),
                "test_setup": ex.get("test_setup", ""),
                "challenge_tests": ex.get("challenge_tests", []),
            })
        return results

    def describe(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return f"No relevant Python examples found for: {query}"

        lines = [f"PythonKnowledgeRetriever results for: {query}"]
        for i, r in enumerate(results, 1):
            args = ", ".join(r["args"])
            sim = f"{r['similarity']:.3f}"
            lines.append(f"\n{i}. [{sim}] {r['func_name']}({args}) — {r['text'][:120]}")
            code_preview = r["code"][:200].replace("\n", " | ")
            lines.append(f"   code: {code_preview}")
        return "\n".join(lines)
