#!/usr/bin/env bash
# Smoke-test the CPU gateway image: /health, /redact, /restore (key scope), /proxy (404 when
# disabled; full round trip against scripts/mock_llm.py), and no values in the container log.
#   make docker-smoke        (builds pii-gateway:cpu first)
set -euo pipefail
IMG=${IMG:-pii-gateway:cpu}; PORT=${PORT:-8766}; MPORT=${MPORT:-8799}
work=$(mktemp -d); trap 'docker rm -f pii-smoke pii-smoke-proxy >/dev/null 2>&1; kill $mock 2>/dev/null; rm -rf "$work"' EXIT
VK=$(python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())")
RK=$(python3 -c "import secrets;print(secrets.token_hex(16))")
run() { docker run -d --name "$1" -p "127.0.0.1:$2:8000" --read-only --tmpfs /tmp --cap-drop ALL \
        --add-host=host.docker.internal:host-gateway -e PII_VAULT_KEY="$VK" -e PII_RESTORE_KEY="$RK" "${@:3}" "$IMG" >/dev/null; }
wait_up() { for _ in $(seq 1 60); do curl -sf "localhost:$1/health" >/dev/null && return; sleep 0.5; done; echo "no health"; exit 1; }
SECRET='card 4111 1111 1111 1111, mail ann.lee@example.com, call +44 20 7946 0958'

run pii-smoke "$PORT"; wait_up "$PORT"
echo "health:   $(curl -s localhost:$PORT/health)"
R=$(curl -sf -X POST localhost:$PORT/redact -H 'Content-Type: application/json' \
    -d "{\"text\":\"$SECRET\",\"policy\":\"analytics\",\"conversation_id\":\"s1\"}")
echo "redact:   $(echo "$R" | python3 -c 'import sys,json;print(json.load(sys.stdin)["redacted"])')"
RED=$(echo "$R" | python3 -c "import sys,json;print(json.dumps({'text':json.load(sys.stdin)['redacted'],'conversation_id':'s1'}))")
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$PORT/restore -H 'Content-Type: application/json' -d "$RED")" = 403 ]
BACK=$(curl -sf -X POST localhost:$PORT/restore -H 'Content-Type: application/json' -H "X-Restore-Key: $RK" -d "$RED" \
       | python3 -c 'import sys,json;print(json.load(sys.stdin)["restored"])')
[ "$BACK" = "$SECRET" ] && echo "restore:  exact (403 without key)"
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$PORT/proxy -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}]}')" = 404 ] && echo "proxy:    404 when PII_UPSTREAM_URL is unset"

(cd "$work" && exec python3 "$OLDPWD/scripts/mock_llm.py" "$MPORT") & mock=$!
run pii-smoke-proxy $((PORT + 1)) -e PII_UPSTREAM_URL="http://host.docker.internal:$MPORT/v1"; wait_up $((PORT + 1))
REPLY=$(curl -sf -X POST localhost:$((PORT + 1))/proxy -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$SECRET\"}]}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["reply"])')
[ "$REPLY" = "$SECRET" ] && echo "proxy:    round trip restored the reply exactly"
! grep -qE '4111|ann\.lee|7946' "$work/mock_llm_requests.jsonl" && echo "proxy:    upstream saw only redacted text"
LEAKS=$(docker logs pii-smoke 2>&1; docker logs pii-smoke-proxy 2>&1) 
! echo "$LEAKS" | grep -qE '4111|ann\.lee|7946' && echo "logs:     no values in either container log"
echo "docker smoke: OK"
