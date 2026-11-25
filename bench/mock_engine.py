#!/usr/bin/env python3
"""Deterministic OpenAI-compatible mock engine for the campaign guard tests.

Stdlib only, no GPU, localhost only.  It answers the routes `bench/bb-campaign.sh`
and BetterBench 0.6.0 touch:

  GET  /v1/models                model list (also the reachability probe)
  GET  /get_server_info          the server args the provenance gate archives
  GET  /health                   liveness
  POST /v1/chat/completions      streamed chunks, so decode, prefill and
  POST /v1/completions           concurrency produce real rows

The framing is deliberately the plain SSE form (`data: {...}` plus
`data: [DONE]`) on a closing connection, the one BetterBench 0.6.0 was observed
to parse on this host; `server-info.json` values mirror the live
`qwen38-rc2` answer so the guard tests exercise the real shape.  Port is
argv[1] (default 39124).  This is a test double, never a bench target: a number
produced against it is worthless, only the gates are being tested.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMPLETION_CHUNKS = 16
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 39124

# mirrors the live production answer, measured 2026-11-22 (not invented):
# chunked_prefill_size 8192, max_prefill_tokens 16384, context_length 524288
SERVER_INFO = {
    "model_path": "/models/qwen38",
    "served_model_name": ["flash-next"],
    "chunked_prefill_size": 8192,
    "max_prefill_tokens": 16384,
    "max_running_requests": 8,
    "kv_cache_dtype": "nvfp4",
    "enable_hierarchical_cache": True,
    "enable_mixed_chunk": False,
    "page_size": 64,
    "context_length": 524288,
    "mem_fraction_static": 0.94,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                 # keep the harness output clean
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{
                "id": "flash-next", "object": "model",
                "max_model_len": 524288, "max_tokens": 524288}]})
        elif self.path.startswith("/get_server_info"):
            self._send(200, SERVER_INFO)
        elif self.path.startswith("/health"):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "no such route"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        text = req.get("prompt") or req.get("messages") or ""
        prompt_tokens = max(1, len(str(text)) // 4)

        if not req.get("stream", True):
            self._send(200, {
                "id": "cmpl-mock", "object": "chat.completion",
                "model": req.get("model"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": "ok " * COMPLETION_CHUNKS},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": COMPLETION_CHUNKS,
                          "total_tokens": prompt_tokens + COMPLETION_CHUNKS}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(COMPLETION_CHUNKS):
            chunk = {"id": "cmpl-mock", "object": "chat.completion.chunk",
                     "model": req.get("model"),
                     "choices": [{"index": 0, "delta": {"content": "tok "},
                                  "finish_reason": None}]}
            if i == COMPLETION_CHUNKS - 1:
                chunk["choices"][0]["finish_reason"] = "stop"
                chunk["usage"] = {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": COMPLETION_CHUNKS,
                                  "total_tokens": prompt_tokens + COMPLETION_CHUNKS}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.004)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
