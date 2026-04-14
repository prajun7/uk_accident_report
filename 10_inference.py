#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from risk_zone_model import RiskZonePredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local inference against the current UK accident risk-zone model bundle."
    )
    parser.add_argument(
        "--json",
        help="Inline JSON payload. Partial payloads are allowed; missing fields use trained defaults.",
    )
    parser.add_argument(
        "--input",
        help="Path to a JSON file containing one payload object.",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print the accepted raw feature schema and exit.",
    )
    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Print the default payload used for omitted fields and exit.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def load_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.json and args.input:
        raise ValueError("Use either --json or --input, not both.")

    if args.json:
        return json.loads(args.json)

    if args.input:
        payload_path = Path(args.input)
        return json.loads(payload_path.read_text(encoding="utf-8"))

    return None


def example_payloads() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "Friday night rural corridor",
            {
                "Speed_limit": 60,
                "Road_Type": 6,
                "1st_Road_Class": 3,
                "1st_Road_Number": 507.0,
                "Junction_Detail": 2.0,
                "Junction_Control": 4.0,
                "Light_Conditions": 4,
                "Weather_Conditions": 2.0,
                "Road_Surface_Conditions": 2.0,
                "Day_of_Week": 6,
                "Urban_or_Rural_Area": 2,
                "Hour": 23,
            },
        ),
        (
            "Weekday daylight urban trip",
            {
                "Speed_limit": 30,
                "Road_Type": 3,
                "1st_Road_Class": 4,
                "1st_Road_Number": 0.0,
                "Junction_Detail": 1.0,
                "Junction_Control": 4.0,
                "Light_Conditions": 1,
                "Weather_Conditions": 1.0,
                "Road_Surface_Conditions": 1.0,
                "Day_of_Week": 2,
                "Urban_or_Rural_Area": 1,
                "Hour": 14,
            },
        ),
    ]


def print_json(data: Any, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(json.dumps(data))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    predictor = RiskZonePredictor()

    if args.show_schema:
        print_json(predictor.schema(), pretty=True)
        return

    if args.show_defaults:
        print_json(predictor.default_payload, pretty=True)
        return

    payload = load_payload(args)
    if payload is not None:
        print_json(predictor.predict(payload), pretty=args.pretty)
        return

    for title, example in example_payloads():
        print(f"--- {title} ---")
        print_json(predictor.predict(example), pretty=True)


if __name__ == "__main__":
    main()
