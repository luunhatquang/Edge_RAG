"""Tests for the BM25 query UI, against a real index built from the fixture corpus."""

import json

import pytest

pytest.importorskip("pyserini")

from src.bm25 import build_index, prepare_data, search_app  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in FIXTURE_DOCS))
    prepare_data.prepare(raw, tmp / "processed")
    build_index.build(
        tmp / "processed", tmp / "index", tmp / "rec.json",
        k1=0.9, b=0.4, threads=1, smoke_query="tea", smoke_k=1,
    )
    return search_app.create_app(tmp / "index").test_client()


def test_api_search_ranks_and_explains(client):
    r = client.get("/api/search?q=green+tea&k=3")
    assert r.status_code == 200
    data = r.get_json()
    top = data["results"][0]
    assert top["docid"] == "MED-301"
    assert {t["term"] for t in data["query_terms"]} == {"green", "tea"}
    # Per-term contributions add up to the Lucene score.
    assert sum(top["term_contributions"].values()) == pytest.approx(top["score"], abs=1e-3)


def test_repeated_query_terms_are_weighted(client):
    top = client.get("/api/search?q=green+green+tea").get_json()["results"][0]
    assert top["term_contributions"]["green"] == pytest.approx(2 * top["term_contributions"]["tea"], abs=1e-3)
    assert sum(top["term_contributions"].values()) == pytest.approx(top["score"], abs=1e-3)


def test_k1_b_change_scores(client):
    a = client.get("/api/search?q=cancer&k1=0.9&b=0.4").get_json()
    b = client.get("/api/search?q=cancer&k1=1.5&b=0.9").get_json()
    assert a["results"][0]["score"] != b["results"][0]["score"]


@pytest.mark.parametrize("qs", ["q=", "q=tea&k=0", "q=tea&b=2", "q=tea&k1=-1", "q=tea&k=abc"])
def test_api_rejects_bad_params(client, qs):
    assert client.get(f"/api/search?{qs}").status_code == 400


def test_html_page_highlights_and_escapes(client):
    html = client.get("/?q=cancers+%3Cscript%3E").get_data(as_text=True)
    assert "<mark>cancer</mark>" in html
    assert "<script>" not in html
    assert "MED-10" in html


def test_stopword_only_query(client):
    html = client.get("/?q=the+and").get_data(as_text=True)
    assert "No indexable terms" in html
