"""Check that every citation in eval/answer_key.json names a real place in the parsed corpus.

  python -m eval.check_key
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
key = json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))
chunks = [json.loads(l) for l in (ROOT / "data" / "corpus" / "chunks_256.jsonl").read_text(encoding="utf-8").splitlines()]
known = {c["cite"] for c in chunks}
# a section cite like "MOSHI §C" also matches when only its subsections have text ("MOSHI §C.1")
prefixes = {c["cite"].split(".")[0] for c in chunks} | {c["cite"].rsplit(".", 1)[0] for c in chunks}

bad = []
for q in key["questions"]:
    for r in q.get("required", []):
        for c in r["cites"]:
            if c not in known and c not in prefixes:
                bad.append((q["id"], c))
print(f"{sum(len(r['cites']) for q in key['questions'] for r in q.get('required', []))} cites checked; "
      f"{len(bad)} unknown: {bad}")
print(f"verified: {sum(q['verified'] for q in key['questions'])}/{len(key['questions'])}")
