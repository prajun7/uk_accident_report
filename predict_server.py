#!/usr/bin/env python3
"""
Local HTTP server: serves index.html (with PREDICT_URL injected) and POST /predict
using output/7_rf_model.joblib. Run from the project root:

  python3 predict_server.py

Then open http://127.0.0.1:8765/ in your browser. The UI will call /predict on the
same origin — no edits to index.html are required on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import Any
from urllib.parse import unquote, urlparse

import joblib
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "7_rf_model.joblib")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

FEATURE_COLS = [
    "Speed_limit",
    "Road_Type",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Day_of_Week",
    "Urban_or_Rural_Area",
    "Hour",
    "IsNight",
    "Month",
]

SEVERITY_NAMES = {1: "Fatal", 2: "Serious", 3: "Slight"}

_model = None


def load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not os.path.isfile(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading model from {MODEL_PATH} (this may take a minute for large files)…")
    _model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return _model


def row_from_payload(data: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in FEATURE_COLS:
        if col not in data:
            row[col] = 0
            continue
        v = data[col]
        if col in ("Weather_Conditions", "Road_Surface_Conditions"):
            row[col] = float(v)
        else:
            row[col] = int(v)
    return pd.DataFrame([row])[FEATURE_COLS]


def inject_predict_url(html: str) -> str:
    return html.replace('var PREDICT_URL = "";', 'var PREDICT_URL = "/predict";', 1)


def safe_output_path(url_path: str) -> str | None:
    """Map /output/... to a path under OUTPUT_DIR; return None if invalid."""
    if not url_path.startswith("/output/"):
        return None
    rel = unquote(url_path[len("/output/") :].lstrip("/"))
    if not rel or ".." in rel:
        return None
    full = os.path.normpath(os.path.join(OUTPUT_DIR, rel))
    out_abs = os.path.abspath(OUTPUT_DIR)
    if not full.startswith(out_abs + os.sep) and full != out_abs:
        return None
    return full if os.path.isfile(full) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "PredictServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path in ("/", "/index.html"):
            if not os.path.isfile(INDEX_PATH):
                self.send_error(404, "index.html not found")
                return
            with open(INDEX_PATH, encoding="utf-8") as f:
                body = inject_predict_url(f.read()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        static = safe_output_path(path)
        if static:
            ctype = "application/octet-stream"
            if static.lower().endswith(".png"):
                ctype = "image/png"
            elif static.lower().endswith(".csv"):
                ctype = "text/csv"
            with open(static, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
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
            model = load_model()
            row = row_from_payload(data)
            pred = int(model.predict(row)[0])
            probs = model.predict_proba(row)[0]
            label = SEVERITY_NAMES.get(pred, str(pred))
            detail = "Probabilities: Fatal=%.2f, Serious=%.2f, Slight=%.2f" % tuple(probs)
            out = {
                "label": label,
                "prediction": pred,
                "probabilities": {
                    "Fatal": float(probs[0]),
                    "Serious": float(probs[1]),
                    "Slight": float(probs[2]),
                },
                "detail": detail,
            }
            payload = json.dumps(out).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve index.html and POST /predict for the RF model.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    args = parser.parse_args()

    load_model()

    httpd: BaseServer = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving http://{args.host}:{args.port}/ — open in your browser, then use Predict severity.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
