"""Local web UI for trying BM25 queries against the Lucene index.

Shows, for each query:
  - the analyzed query terms (after tokenization, stopword removal, Porter
    stemming) with their document frequency and IDF;
  - the top-k documents with BM25 score, per-term score contributions, and
    the document text with matched terms highlighted.

k1, b and k can be changed per query. The index is opened read-only and is
never modified.

Usage:
    python -m src.bm25.search_app                          # http://127.0.0.1:8000
    python -m src.bm25.search_app --index-dir path/to/index --port 8080

JSON API:
    GET /api/search?q=vitamin+d&k=10&k1=0.9&b=0.4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from markupsafe import Markup, escape

from src.bm25.build_index import DEFAULT_INDEX_DIR, PARAMS_FILENAME

MAX_K = 100
SNIPPET_CHARS = 400
WORD_RE = re.compile(r"\w+|\W+")


class BM25Engine:
    """Thread-safe wrapper around a Pyserini LuceneSearcher and index reader."""

    def __init__(self, index_dir: Path):
        from pyserini.index.lucene import LuceneIndexReader
        from pyserini.search.lucene import LuceneSearcher

        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}. Run `python -m src.bm25.build_index` first.")
        self.index_dir = index_dir
        self.searcher = LuceneSearcher(str(index_dir))
        self.reader = LuceneIndexReader(str(index_dir))
        stats = self.reader.stats()
        self.num_docs = stats["documents"]
        self.avgdl = stats["total_terms"] / max(self.num_docs, 1)
        params_path = index_dir / PARAMS_FILENAME
        self.default_params = json.loads(params_path.read_text()) if params_path.exists() else {"k1": 0.9, "b": 0.4}
        self._stem_cache: dict[str, list[str]] = {}
        # pyjnius objects are not safe to share across Flask worker threads.
        self._lock = threading.Lock()

    def _analyze_word(self, word: str) -> list[str]:
        key = word.lower()
        if key not in self._stem_cache:
            self._stem_cache[key] = self.reader.analyze(word)
        return self._stem_cache[key]

    def highlight(self, text: str, terms: set[str]) -> Markup:
        """HTML-escape ``text`` and wrap words whose analyzed form is a query term."""
        out = []
        for tok in WORD_RE.findall(text):
            if tok[0].isalnum() or tok[0] == "_":
                if terms.intersection(self._analyze_word(tok)):
                    out.append(Markup("<mark>{}</mark>").format(tok))
                    continue
            out.append(escape(tok))
        return Markup("").join(out)

    def search(self, query: str, k: int, k1: float, b: float) -> dict:
        with self._lock:
            start = time.perf_counter()
            self.searcher.set_bm25(k1, b)
            hits = self.searcher.search(query, k=k)
            latency_ms = (time.perf_counter() - start) * 1000

            # Pyserini's bag-of-words query boosts each term by its count in the query.
            qtf = Counter(self.reader.analyze(query))
            terms = []
            for term, count in qtf.items():
                df, _ = self.reader.get_term_counts(term, analyzer=None)
                idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
                terms.append({"term": term, "query_tf": count, "df": df, "idf": round(idf, 4)})

            results = []
            for rank, hit in enumerate(hits, start=1):
                doc = json.loads(self.searcher.doc(hit.docid).raw())
                contrib = {
                    t["term"]: round(
                        t["query_tf"]
                        * self.reader.compute_bm25_term_weight(hit.docid, t["term"], analyzer=None, k1=k1, b=b),
                        4,
                    )
                    for t in terms
                }
                results.append(
                    {
                        "rank": rank,
                        "docid": hit.docid,
                        "score": round(float(hit.score), 4),
                        "doc_length": sum(self.reader.get_document_vector(hit.docid).values()),
                        "term_contributions": contrib,
                        "contents": doc.get("contents", ""),
                    }
                )
        return {
            "query": query,
            "k": k,
            "k1": k1,
            "b": b,
            "latency_ms": round(latency_ms, 2),
            "query_terms": terms,
            "results": results,
        }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BM25 Query Tester</title>
<style>
  :root { --bg:#f7f7f8; --card:#fff; --fg:#1d1d1f; --muted:#6b6b75; --line:#e2e2e6; --accent:#2f5bd3; --mark:#ffe58a; --err:#b42318; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141417; --card:#1d1d22; --fg:#ececf1; --muted:#9a9aa6; --line:#2e2e36; --accent:#7c9cff; --mark:#6b5a12; --err:#ff8a80; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:960px; margin:0 auto; padding:24px 16px 48px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
  form { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; display:grid; gap:10px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:end; }
  label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:3px; }
  input { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:7px 9px; }
  input[name=q] { width:100%; font-size:16px; }
  input[type=number] { width:90px; }
  button { font:inherit; background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 18px; cursor:pointer; }
  .err { color:var(--err); margin-top:12px; }
  .meta { color:var(--muted); font-size:13px; margin:16px 0 8px; }
  table { border-collapse:collapse; font-size:13px; margin-bottom:16px; }
  th, td { text-align:left; padding:4px 12px 4px 0; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; }
  .hit { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:10px; }
  .hit-head { display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; }
  .docid { font-weight:600; font-family:ui-monospace, monospace; }
  .score { font-variant-numeric:tabular-nums; }
  .contrib { font-size:12px; color:var(--muted); margin:4px 0 6px; font-family:ui-monospace, monospace; }
  .text { overflow-wrap:anywhere; }
  mark { background:var(--mark); color:inherit; padding:0 1px; border-radius:2px; }
  details summary { cursor:pointer; color:var(--accent); font-size:13px; }
</style>
</head>
<body>
<main>
  <h1>BM25 Query Tester</h1>
  <div class="sub">Index <code>{{ index_dir }}</code> · {{ num_docs }} docs · avgdl {{ "%.1f"|format(avgdl) }}</div>
  <form method="get" action="/">
    <label>Query<input name="q" value="{{ q }}" placeholder="e.g. vitamin D and cancer risk" autofocus></label>
    <div class="row">
      <label>top-k<input type="number" name="k" min="1" max="{{ max_k }}" value="{{ k }}"></label>
      <label>k1<input type="number" name="k1" step="0.05" min="0" value="{{ k1 }}"></label>
      <label>b<input type="number" name="b" step="0.05" min="0" max="1" value="{{ b }}"></label>
      <button type="submit">Search</button>
    </div>
  </form>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  {% if res %}
    <div class="meta">{{ res.results|length }} hits · {{ res.latency_ms }} ms search latency · k1={{ res.k1 }}, b={{ res.b }}</div>
    {% if res.query_terms %}
    <table>
      <tr><th>Analyzed term</th><th>query tf</th><th>df</th><th>IDF</th></tr>
      {% for t in res.query_terms %}
      <tr><td><code>{{ t.term }}</code></td><td>{{ t.query_tf }}</td><td>{{ t.df }}</td><td>{{ t.idf }}</td></tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="meta">No indexable terms in this query (all stopwords or punctuation).</div>
    {% endif %}
    {% for h in res.results %}
    <div class="hit">
      <div class="hit-head">
        <span>#{{ h.rank }} <span class="docid">{{ h.docid }}</span></span>
        <span class="score">score {{ h.score }} · len {{ h.doc_length }}</span>
      </div>
      <div class="contrib">{% for t, v in h.term_contributions.items() %}{{ t }}={{ v }}{% if not loop.last %} + {% endif %}{% endfor %}</div>
      {% if h.contents|length > snippet_chars %}
        <details><summary><span class="text">{{ h.snippet }}…</span></summary><div class="text">{{ h.full }}</div></details>
      {% else %}
        <div class="text">{{ h.full }}</div>
      {% endif %}
    </div>
    {% endfor %}
  {% endif %}
</main>
</body>
</html>
"""


def parse_params(args, defaults: dict) -> tuple[dict | None, str | None]:
    try:
        k = int(args.get("k", 10))
        k1 = float(args.get("k1", defaults["k1"]))
        b = float(args.get("b", defaults["b"]))
    except ValueError:
        return None, "k, k1 and b must be numbers."
    if not 1 <= k <= MAX_K:
        return None, f"k must be between 1 and {MAX_K}."
    if k1 < 0:
        return None, "k1 must be >= 0."
    if not 0 <= b <= 1:
        return None, "b must be in [0, 1]."
    return {"k": k, "k1": k1, "b": b}, None


def create_app(index_dir: Path) -> Flask:
    engine = BM25Engine(index_dir)
    app = Flask(__name__)

    @app.get("/")
    def index():
        q = request.args.get("q", "").strip()
        params, error = parse_params(request.args, engine.default_params)
        res = None
        if q and params:
            res = engine.search(q, **params)
            terms = {t["term"] for t in res["query_terms"]}
            for h in res["results"]:
                h["snippet"] = engine.highlight(h["contents"][:SNIPPET_CHARS], terms)
                h["full"] = engine.highlight(h["contents"], terms)
        shown = params or {"k": 10, **engine.default_params}
        return render_template_string(
            PAGE,
            q=q,
            res=res,
            error=error,
            max_k=MAX_K,
            snippet_chars=SNIPPET_CHARS,
            index_dir=engine.index_dir,
            num_docs=engine.num_docs,
            avgdl=engine.avgdl,
            **shown,
        )

    @app.get("/api/search")
    def api_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "Missing query parameter q."}), 400
        params, error = parse_params(request.args, engine.default_params)
        if error:
            return jsonify({"error": error}), 400
        return jsonify(engine.search(q, **params))

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    create_app(a.index_dir).run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
