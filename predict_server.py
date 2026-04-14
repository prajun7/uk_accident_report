#!/usr/bin/env python3
"""
Local HTTP server for the current UK accident risk-zone model bundle.

Run from the project root:

  ./.venv/bin/python predict_server.py

Then open http://127.0.0.1:8765/ in your browser. The server exposes:

  GET  /         -> index.html with /predict injected
  POST /predict  -> risk-zone prediction JSON
  GET  /schema   -> accepted feature schema and defaults
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import Any
from urllib.parse import unquote, urlparse

from risk_zone_model import RiskZonePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

_predictor: RiskZonePredictor | None = None


def load_predictor() -> RiskZonePredictor:
    global _predictor
    if _predictor is None:
        print("Loading current risk-zone model bundle...")
        _predictor = RiskZonePredictor(BASE_DIR)
        print(f"Loaded {_predictor.model_name}.")
    return _predictor


def inject_page_updates(html: str) -> str:
    replacements = [
        ('var PREDICT_URL = "";', 'var PREDICT_URL = "/predict";'),
        ("Accident severity predictor", "Accident risk zone predictor"),
        (
            '<strong style="color: #e8edf7">Fatal</strong>, <strong style="color: #e8edf7">Serious</strong>, or <strong style="color: #e8edf7">Slight</strong>',
            '<strong style="color: #e8edf7">Low Risk</strong>, <strong style="color: #e8edf7">Medium Risk</strong>, or <strong style="color: #e8edf7">High Risk</strong>',
        ),
        (
            'severity. Values use UK DfT numeric codes expected by <code style="color: var(--accent)">10_inference.py</code>.',
            'risk zone. This form sends a compact payload; the server rebuilds the engineered features expected by <code style="color: var(--accent)">10_inference.py</code>.',
        ),
        ("Predict severity", "Predict risk zone"),
        ("Predicted severity:", "Predicted risk zone:"),
        ("trained <code>7_rf_model.joblib</code>", "trained <code>7_model_bundle.joblib</code>"),
        ('model = joblib.load("output/7_rf_model.joblib")', 'from risk_zone_model import RiskZonePredictor\npredictor = RiskZonePredictor()'),
        ('pred = model.predict(row[cols])[0]', 'result = predictor.predict(row.iloc[0].to_dict())'),
        ('probs = model.predict_proba(row[cols])[0]', 'probs = result["probabilities"]'),
        ('names = {1: "Fatal", 2: "Serious", 3: "Slight"}', ""),
        ('print(names.get(pred, pred), "—", "Fatal=%.2f Serious=%.2f Slight=%.2f" % tuple(probs))', 'print(result["label"], "-", result.get("detail", ""))'),
        (
            "Browser cannot load the trained Random Forest (~685MB joblib). Run the snippet below from the project root to predict with the same features.",
            "Browser cannot run the trained model bundle directly. This page works when served through predict_server.py, or you can use the Python snippet below.",
        ),
        (
            "Tip: set const PREDICT_URL in index.html if you expose a small POST /predict JSON service that loads 7_rf_model.joblib.",
            "Tip: GET /schema to see the accepted fields and defaults used by the current model bundle.",
        ),
    ]

    for old, new in replacements:
        html = html.replace(old, new)

    html = html.replace(
        """function severityClass(label) {
        if (/fatal/i.test(label)) return "sev-fatal";
        if (/serious/i.test(label)) return "sev-serious";
        if (/slight/i.test(label)) return "sev-slight";
        return "";
      }""",
        """function severityClass(label) {
        if (/high/i.test(label)) return "sev-fatal";
        if (/medium/i.test(label)) return "sev-serious";
        if (/low/i.test(label)) return "sev-slight";
        return "";
      }""",
    )

    return html


def safe_output_path(url_path: str) -> str | None:
    if not url_path.startswith("/output/"):
        return None

    rel = unquote(url_path[len("/output/") :].lstrip("/"))
    if not rel or ".." in rel:
        return None

    full = os.path.normpath(os.path.join(OUTPUT_DIR, rel))
    output_abs = os.path.abspath(OUTPUT_DIR)
    if not full.startswith(output_abs + os.sep) and full != output_abs:
        return None

    return full if os.path.isfile(full) else None


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "PredictServer/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path in ("/", "/index.html"):
            if not os.path.isfile(INDEX_PATH):
                self.send_error(404, "index.html not found")
                return

            with open(INDEX_PATH, encoding="utf-8") as handle:
                body = inject_page_updates(handle.read()).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/schema":
            write_json(self, 200, load_predictor().schema())
            return

        if path == "/healthz":
            write_json(self, 200, {"ok": True, "model_name": load_predictor().model_name})
            return

        static = safe_output_path(path)
        if static:
            content_type = "application/octet-stream"
            if static.lower().endswith(".png"):
                content_type = "image/png"
            elif static.lower().endswith(".csv"):
                content_type = "text/csv"

            with open(static, "rb") as handle:
                body = handle.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"

        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        if not isinstance(data, dict):
            self.send_error(400, "JSON object required")
            return

        try:
            result = load_predictor().predict(data)
        except Exception as exc:
            write_json(self, 500, {"error": str(exc)})
            return

        write_json(self, 200, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve index.html and POST /predict for the current risk-zone model bundle."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    args = parser.parse_args()

    predictor = load_predictor()
    print(f"Serving {predictor.model_name} at http://{args.host}:{args.port}/")

    httpd: BaseServer = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
