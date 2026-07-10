"""Live HTTP edge-case checks for the running Task 13 scoring container.

Start the container first, then run:
    python src/edge_case_tests_api.py

Set API_BASE_URL to target a non-default host or port.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    from .score_extractor import FEATURE_NAMES
except ImportError:  # pragma: no cover - direct-script compatibility
    from score_extractor import FEATURE_NAMES

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "clean_modelling_table.csv"
results: list[tuple[str, str]] = []


def request(method: str, path: str, payload=None, raw_body: bytes | None = None):
    body = raw_body if raw_body is not None else (
        json.dumps(payload).encode("utf-8") if payload is not None else None
    )
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def check(name, assertion):
    try:
        assertion()
        results.append((name, "PASS"))
        print(f"PASS  {name}")
    except Exception as exc:
        results.append((name, f"FAIL: {exc}"))
        print(f"FAIL  {name}: {exc}")


def sample_payloads():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_NAMES)
    if len(data) < 2:
        raise RuntimeError("Need at least two complete records for live API tests")
    def payload(row, record_id):
        return {
            "record_id": record_id,
            "features": {
                name: (row[name].item() if hasattr(row[name], "item") else row[name])
                for name in FEATURE_NAMES
            },
        }
    return payload(data.iloc[0], "test_001"), payload(data.iloc[1], "test_002")


def main():
    try:
        single, second = sample_payloads()
        status, _ = request("GET", "/health")
    except URLError as exc:
        raise SystemExit(f"API is not reachable at {BASE_URL}: {exc}") from exc

    check("Container health endpoint returns 200", lambda: _assert(status == 200, status))
    check("Missing feature returns 422", lambda: _assert_status("POST", "/score", _without_feature(single), 422))
    check("Non-numeric feature returns 422", lambda: _assert_status("POST", "/score", _non_numeric(single), 422))
    check("Malformed JSON returns 422", lambda: _assert_raw_status("POST", "/score", b"{not valid json", 422))
    check("Wrong HTTP method returns 405", lambda: _assert_status("GET", "/score", None, 405))
    check("Empty batch returns 422", lambda: _assert_status("POST", "/score-batch", [], 422))
    check("Unknown route returns 404", lambda: _assert_status("GET", "/unknown-endpoint", None, 404))
    check("Clean single record returns a valid score", lambda: _assert_single(single))
    check("Clean batch returns two valid scores", lambda: _assert_batch([single, second]))
    check("Repeated scoring is deterministic", lambda: _assert_reproducible(single))

    passed = sum(result == "PASS" for _, result in results)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        raise SystemExit(1)


def _assert(condition, actual):
    if not condition:
        raise AssertionError(f"unexpected result: {actual}")


def _assert_status(method, path, payload, expected):
    status, body = request(method, path, payload)
    _assert(status == expected, {"status": status, "body": body})


def _assert_raw_status(method, path, raw_body, expected):
    status, body = request(method, path, raw_body=raw_body)
    _assert(status == expected, {"status": status, "body": body})


def _without_feature(payload):
    invalid = {**payload, "features": dict(payload["features"])}
    invalid["features"].pop("python_required")
    return invalid


def _non_numeric(payload):
    invalid = {**payload, "features": dict(payload["features"])}
    invalid["features"]["python_required"] = "not-a-number"
    return invalid


def _assert_score(result):
    required = {
        "match_score", "decision", "score_meaning", "model_version",
        "record_id", "container_image_digest",
    }
    _assert(required.issubset(result), result)
    _assert(0.0 <= result["match_score"] <= 1.0, result)
    _assert(result["decision"] in (0, 1), result)


def _assert_single(payload):
    status, body = request("POST", "/score", payload)
    _assert(status == 200, {"status": status, "body": body})
    _assert_score(body)


def _assert_batch(payloads):
    status, body = request("POST", "/score-batch", payloads)
    _assert(status == 200, {"status": status, "body": body})
    _assert(len(body) == len(payloads), body)
    for result in body:
        _assert_score(result)


def _assert_reproducible(payload):
    first_status, first = request("POST", "/score", payload)
    second_status, second = request("POST", "/score", payload)
    _assert(first_status == second_status == 200, [first_status, second_status])
    _assert(first["match_score"] == second["match_score"], [first, second])


if __name__ == "__main__":
    main()
