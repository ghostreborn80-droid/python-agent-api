
"""
nsagent.db_retriever — Lightweight SQLite FTS retrieval with lexical re-ranking.

No TF-IDF, no sklearn, no large in-memory matrix. Works within Render's
free-tier memory while still returning useful Python examples.
"""
import sqlite3
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class DatabaseRetriever:
    def __init__(self, db_path: str = "/content/agent_data/python_knowledge.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print(f"[DBRetriever] connected to {self.db_path}")

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

    def _to_fts_query(self, text: str) -> Optional[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        tokens = [t for t in tokens if len(t) > 1]
        if not tokens:
            return None
        return " OR ".join(f'"{t}"' for t in tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self._expand_query(self._normalize_query(query))
        candidates = self._search_fts(q, top_k=200)

        if not candidates:
            candidates = self._search_like(q, top_k=200)

        if not candidates:
            return []

        q_tokens = self._tokens(q)

        scored = []
        for ex in candidates:
            fn_tokens = self._tokens(ex.get("func_name", ""))
            text_tokens = self._tokens(ex.get("text", ""))
            code_tokens = self._tokens(ex.get("code", ""))

            overlap = (3 * len(q_tokens & fn_tokens) +
                       2 * len(q_tokens & text_tokens) +
                       1 * len(q_tokens & code_tokens))

            scored.append((overlap, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def _search_fts(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        fts_query = self._to_fts_query(query)
        if not fts_query:
            return []

        sql = """
        SELECT
            c.id, c.source, c.text, c.code, c.func_name, c.args, c.tests, c.test_setup
        FROM code_examples_fts f
        JOIN code_examples c ON c.rowid = f.rowid
        WHERE code_examples_fts MATCH ?
        ORDER BY bm25(code_examples_fts, 1.0, 1.0, 0.5, 0.0)
        LIMIT ?
        """
        try:
            rows = self.conn.execute(sql, (fts_query, top_k)).fetchall()
        except Exception as exc:
            return []
        return [self._row_to_dict(r) for r in rows]

    def _search_like(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        if not tokens:
            return []

        clauses = " OR ".join([
            "(LOWER(text) LIKE ? OR LOWER(code) LIKE ? OR LOWER(func_name) LIKE ?)"
            for _ in tokens
        ])
        params = []
        for t in tokens:
            pattern = f"%{t}%"
            params.extend([pattern, pattern, pattern])

        sql = f"""
        SELECT id, source, text, code, func_name, args, tests, test_setup
        FROM code_examples
        WHERE {clauses}
        LIMIT ?
        """
        rows = self.conn.execute(sql, (*params, top_k)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("args", "tests"):
            try:
                d[key] = json.loads(d.get(key, "[]") or "[]")
            except Exception:
                d[key] = []
        return d

    def describe(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return f"No Python examples found for: {query}"

        lines = [f"DatabaseRetriever results for: {query}"]
        for i, r in enumerate(results, 1):
            args = ", ".join(r.get("args", []))
            lines.append(f"\n{i}. [{r.get('source')}] {r.get('func_name')}({args}) — {r.get('text', '')[:120]}")
            code_preview = r.get("code", "")[:200].replace("\n", " | ")
            lines.append(f"   code: {code_preview}")
        return "\n".join(lines)

    def close(self):
        self.conn.close()
