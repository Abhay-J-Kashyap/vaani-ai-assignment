"""List the models this Groq key can use (GET /models). This is not a chat request.
The response is appended to results/models_raw.json for the record."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
r = httpx.get("https://api.groq.com/openai/v1/models",
              headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]}, timeout=30)
print("status", r.status_code, "| rate-limit headers:",
      {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})
data = r.json()
(ROOT / "results").mkdir(exist_ok=True)
with (ROOT / "results" / "models_raw.json").open("a", encoding="utf-8") as f:
    f.write(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "body": data}) + "\n")
for m in sorted(data.get("data", []), key=lambda m: m["id"]):
    print(f"{m['id']:<45} active={m.get('active')} ctx={m.get('context_window')} "
          f"max_out={m.get('max_completion_tokens')} owner={m.get('owned_by')}")
