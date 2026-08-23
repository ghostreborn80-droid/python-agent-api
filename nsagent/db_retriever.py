
"""
nsagent.db_retriever — TF-IDF retrieval over SQLite Python examples.

Loads all examples into memory and ranks by cosine similarity over:
  - normalized task description
  - function name
  - code identifiers

Query expansion is applied for common Python task phrases.
"""
import sqlite3
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _code_keywords(code: str) -> str:
    """Extract identifier-ish tokens from Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
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


class DatabaseRetriever:
    def __init__(self, db_path: str = "/content/agent_data/python_knowledge.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.examples = self._load_examples()
        self.texts = self._build_search_texts()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)
        print(f"[DBRetriever] loaded {len(self.examples)} examples, "
              f"vocab={len(self.vectorizer.vocabulary_)}")

    def _load_examples(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM code_examples").fetchall()
        conn.close()

        examples = []
        for row in rows:
            d = dict(row)
            for key in ("args", "tests"):
                try:
                    d[key] = json.loads(d.get(key, "[]") or "[]")
                except Exception:
                    d[key] = []
            examples.append(d)
        return examples

    def _build_search_texts(self) -> List[str]:
        texts = []
        for ex in self.examples:
            text_norm = ex.get("text_norm", "") or ex.get("text", "").lower()
            func_name = ex.get("func_name", "")
            code_kw = _code_keywords(ex.get("code", ""))
            texts.append(f"{text_norm} {text_norm} {func_name} {code_kw}")
        return texts

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()))

    @staticmethod
    def _expand_query(query: str) -> str:
        q = query.lower()
        expansions = []
        if "join" in q:
            expansions += ["concatenate", "concatenate tuple delimiter"]
        if "concatenate" in q:
            expansions += ["join"]
        if "prime" in q:
            expansions += ["prime_num", "is_prime", "is_not_prime"]
        if "sort" in q and "dictionary" in q:
            expansions += ["sorted_models", "sort list of dictionaries lambda"]
        if "reverse" in q and "string" in q:
            expansions += ["reverse_string", "reverse words"]
        if expansions:
            q += " " + " ".join(expansions)
        return q

    @staticmethod
    def _normalize_query(query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r"^(write|generate|create)\s+a\s+python\s+(script|program|function|code)\s+(to\s+)?", "", query)
        query = re.sub(r"\bhow\s+do\s+i\s+", "", query)
        query = re.sub(r"\bin\s+python\b", "", query)
        query = re.sub(r"[^a-z0-9\s]", " ", query)
        return re.sub(r"\s+", " ", query).strip()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self._expand_query(self._normalize_query(query))
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        q_tokens = self._tokens(q)
        overlap_scores = np.zeros_like(sims)
        for i, ex in enumerate(self.examples):
            fn_tokens = self._tokens(ex.get("func_name", ""))
            text_tokens = self._tokens(ex.get("text", ""))
            code_tokens = self._tokens(ex.get("code", ""))

            overlap = (3 * len(q_tokens & fn_tokens) +
                       2 * len(q_tokens & text_tokens) +
                       1 * len(q_tokens & code_tokens))
            overlap_scores[i] = overlap

        # Hybrid: 0.7 cosine + 0.3 lexical overlap
        hybrid = 0.7 * sims + 0.3 * overlap_scores

        k = min(top_k, len(self.examples))
        top_indices = np.argsort(hybrid)[::-1][:k]

        results = []
        for idx in top_indices:
            ex = dict(self.examples[idx])
            ex["similarity"] = float(hybrid[idx])
            results.append(ex)
        return results

    def describe(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return f"No Python examples found for: {query}"

        lines = [f"DatabaseRetriever results for: {query}"]
        for i, r in enumerate(results, 1):
            args = ", ".join(r.get("args", []))
            sim = f"{r.get('similarity', 0):.3f}"
            lines.append(f"\n{i}. [{sim}] [{r.get('source')}] {r.get('func_name')}({args}) — {r.get('text', '')[:120]}")
            code_preview = r.get("code", "")[:200].replace("\n", " | ")
            lines.append(f"   code: {code_preview}")
        return "\n".join(lines)

    def close(self):
        pass
