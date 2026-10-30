"""OpenAI-compatible stub LLM server for TATG smoke tests (no real model).

Usage: python stub_llm.py PORT LOG_DIR CANNED_JAVA_FILE

The stub answers every TATG prompt type deterministically:
  * connectivity probe          -> "OK"
  * teardown synthesis          -> "NO_TEARDOWN"
  * json_schema "new_tests"     -> @Test methods taken round-robin from the
                                   canned JUnit class, in TATG's patch format
                                   (bound to an allowed overload_id when the
                                   prompt lists overload contracts)
  * any other json_schema       -> the minimal instance valid for the schema
  * anything else (e.g. repair) -> the canned class in a ```java fence
Streaming requests get HTTP 400 so the client falls back to plain requests.
Every request/response pair is logged to LOG_DIR/NNNN.json.
"""
import itertools
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_canned(text):
    imports = [l.strip() for l in text.splitlines() if l.strip().startswith("import ")]
    lines = text.splitlines()
    tests, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "@Test":
            start, depth, seen = i, 0, False
            j = i
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                seen = seen or "{" in lines[j]
                if seen and depth == 0:
                    break
                j += 1
            body = [l[4:] if l.startswith("    ") else l for l in lines[start:j + 1]]
            name = re.search(r"void\s+(\w+)\s*\(", "\n".join(body)).group(1)
            tests.append({"test_name": name, "test_code_lines": body})
            i = j + 1
        else:
            i += 1
    return imports, tests


def pick_overload_id(prompt, test_lines):
    match = re.search(r"Allowed overload_id values: ([^\n]+?)\. Every", prompt)
    if not match:
        return ""
    allowed = [item.strip() for item in match.group(1).split(",") if item.strip()]
    code = "\n".join(test_lines)

    def score(overload_id):
        params = overload_id.split("#", 1)[-1].split("-")
        return sum(1 for p in params if p and p in code)

    return max(allowed, key=score)


def minimal_instance(schema):
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), kind[0])
    if "enum" in schema:
        return schema["enum"][-1]
    if kind == "object":
        props = schema.get("properties", {})
        return {k: minimal_instance(props.get(k, {})) for k in schema.get("required", [])}
    if kind == "array":
        return [minimal_instance(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    return {"string": "", "integer": 0, "number": 0, "boolean": False}.get(kind)


PORT, LOG, CANNED = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LOG.mkdir(parents=True, exist_ok=True)
CANNED_TEXT = CANNED.read_text(encoding="utf-8")
IMPORTS, TESTS = parse_canned(CANNED_TEXT)
_rr = itertools.cycle(range(len(TESTS)))
_lock, _n = threading.Lock(), [0]


def respond(req):
    messages = req.get("messages", [])
    system = str(messages[0].get("content", "")) if messages else ""
    schema = ((req.get("response_format") or {}).get("json_schema") or {}).get("schema")
    if "connectivity probe" in system:
        return "OK", "probe"
    if "teardown" in system.lower() and not schema:
        return "NO_TEARDOWN", "teardown"
    if schema and "new_tests" in schema.get("properties", {}):
        count = max(1, schema["properties"]["new_tests"].get("minItems", 1))
        with _lock:
            picked = [TESTS[next(_rr)] for _ in range(count)]
        prompt = str(messages[-1].get("content", ""))
        items = []
        for test in picked:
            item = dict(test, new_imports_lines=IMPORTS)
            overload_id = pick_overload_id(prompt, test["test_code_lines"])
            if overload_id:
                item["overload_id"] = overload_id
            items.append(item)
        return json.dumps({"new_tests": items}), "new_tests"
    if schema:
        return json.dumps(minimal_instance(schema)), "schema:" + req["response_format"]["json_schema"].get("name", "")
    return "```java\n" + CANNED_TEXT + "\n```", "java"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if req.get("stream"):
            return self._send(400, {"error": {"message": "streaming not supported by stub"}})
        content, kind = respond(req)
        with _lock:
            _n[0] += 1
            n = _n[0]
        (LOG / f"{n:04d}.json").write_text(json.dumps({"kind": kind, "request": req, "response": content}, indent=1))
        self._send(200, {
            "id": f"stub-{n}", "object": "chat.completion", "model": req.get("model", "stub-model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


if __name__ == "__main__":
    print(f"stub LLM on http://127.0.0.1:{PORT}/v1 serving {len(TESTS)} canned tests from {CANNED}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
