"""A tiny OpenAI-compatible echo server for smoke tests of /proxy (stdlib only, no model).

    python scripts/mock_llm.py 8799

POST /v1/chat/completions answers with the last message's content, so a /proxy round trip
can be checked end to end. Every request body is appended to mock_llm_requests.jsonl in the
working directory, so the smoke test can assert the upstream never saw a raw value.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Echo(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open("mock_llm_requests.jsonl", "ab") as f:
            f.write(body + b"\n")
        last = json.loads(body)["messages"][-1]["content"]
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": last}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *args):  # keep test output quiet
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8799), Echo).serve_forever()
