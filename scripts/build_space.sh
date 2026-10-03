#!/usr/bin/env bash
# Assemble the Hugging Face Space (space/dist/) from the gateway package and the demo app.
#   make space-build        # then: make space-push (needs `hf auth login`)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=space/dist
rm -rf "$OUT"; mkdir -p "$OUT/configs"
cp space/app.py space/README.md "$OUT/"
cp -R src/pii_gateway "$OUT/pii_gateway"
cp -R configs/policy "$OUT/configs/policy"
find "$OUT" -name '__pycache__' -prune -exec rm -rf {} +
rm -f "$OUT/pii_gateway/api.py"   # the Space runs the pipeline in-process; no FastAPI needed
cat > "$OUT/requirements.txt" <<REQ
httpx>=0.27
pyyaml>=6.0
numpy>=1.26
phonenumbers>=8.13
cryptography>=43
REQ
# smoke: the app imports and redacts with the validators (no model, no network)
(cd "$OUT" && env -u PII_LLM_URL python3 -c "
import app
gw_model, gw_fb = app.build_gateways()
lim = app.Limiter(10, 300)
red, spans, status = app.run('card 4111 1111 1111 1111', 'support', 'local', gw_model, gw_fb, lim)
assert '[CREDITCARDNUMBER]' in red and 'validators only' in status, (red, status)
print('space build OK:', red)
")
