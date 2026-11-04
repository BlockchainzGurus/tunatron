#!/usr/bin/env bash
# Fase 3.2 — verifica con curl che il guided decoding (response_format = JSON Schema) restituisca SOLO il JSON del verdetto.
#
#   scripts/test_guided_decoding.sh [URL]       # default http://127.0.0.1:8000, modello "sast-nemotron"
# Esce con codice != 0 se la risposta non è un oggetto JSON valido secondo shared/vex_schema.json.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:-http://127.0.0.1:8000}"; MODEL="${MODEL:-sast-nemotron}"

BODY="$(python3 - "$MODEL" <<'PY'
import json, sys
sys.path.insert(0, "ai-pipeline")
import common
kb = common.load_kb()
entry = next(e for e in kb["entries"] if e["id"] == "CWE-89")
block = {"file": "src/db/orders.py", "name": "get_orders", "start_line": 40, "end_line": 46, "depth": 0, "lang": "python",
         "code": "def get_orders(request):\n    uid = request.args['user_id']\n    cur = db.cursor()\n    cur.execute(f\"SELECT * FROM orders WHERE user_id = {uid}\")\n    return cur.fetchall()"}
sink = {"function": "cur.execute", "file": "src/db/orders.py", "line": 43}
print(json.dumps({
    "model": sys.argv[1],
    "messages": common.build_messages("CWE-89", common.render_card(entry), sink, [block]),
    "temperature": 0.0,
    "max_tokens": 768,
    "response_format": {"type": "json_schema", "json_schema": {"name": "vex_finding", "strict": True, "schema": common.load_schema()}},
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
)"

echo "+ curl $URL/v1/chat/completions" >&2
RESPONSE="$(curl -sS --fail-with-body -m 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")"

RESPONSE="$RESPONSE" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "ai-pipeline")
import common
content = json.loads(os.environ["RESPONSE"])["choices"][0]["message"]["content"]
print(content)
try:
    finding = json.loads(content)
except ValueError as exc:
    sys.exit(f"FALLITO: la risposta non è solo JSON ({exc})")
errors = common.validate_finding(finding)
if errors:
    sys.exit("FALLITO: " + "; ".join(errors))
print(f"\nOK: JSON valido — vulnerable={finding['vulnerable']} {finding['cwe_id']} {finding['severity']}")
PY
