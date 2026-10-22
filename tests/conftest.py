import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tiny synthetic corpus in BEIR format. Used only to exercise the pipeline;
# it is not NFCorpus and produces no benchmark numbers.
FIXTURE_DOCS = [
    {"_id": "MED-10", "title": "Statin use and breast cancer", "text": "Statins and breast cancer survival.", "metadata": {}},
    {"_id": "MED-14", "title": "Vitamin D", "text": "Vitamin D status and cancer risk in adults.", "metadata": {}},
    {"_id": "MED-118", "title": "", "text": "Dietary fiber intake and colorectal cancer.", "metadata": {}},
    {"_id": "MED-301", "title": "Green tea", "text": "Green tea catechins and cardiovascular health.", "metadata": {}},
]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw" / "nfcorpus"
    d.mkdir(parents=True)
    with open(d / "corpus.jsonl", "w") as f:
        for doc in FIXTURE_DOCS:
            f.write(json.dumps(doc) + "\n")
    return d
