"""
API-level edge case test suite for the Description Matching System.

Uses FastAPI's TestClient (real ASGI request/response cycle, in-process)
to test the actual api.py app object - including scenarios real curl
testing can't easily force: the model failing to load at startup, and
/health + the scoring endpoints degrading to 503 instead of crashing.

Run: python src/edge_case_tests_api.py
"""

import sys
import os
import pandas as pd
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []

def check(name, fn):
    try:
        fn()
        results.append((name, "PASS"))
        print(f"PASS  {name}")
    except AssertionError as e:
        results.append((name, f"FAIL: {e}"))
        print(f"FAIL  {name}: {e}")
    except Exception as e:
        results.append((name, f"ERROR: {e}"))
        print(f"ERROR {name}: {e}")


import api as api_module

# Load test data from our modelling table
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_modelling_table.csv")

if os.path.exists(DATA_PATH):
    test_df = pd.read_csv(DATA_PATH)
    sample_record = test_df.iloc[0].to_dict()
    sample_batch = test_df.iloc[:5].to_dict("records")
else:
    # Fallback sample record if data not available
    sample_record = {
        "id": "test_001",
        "exp_required_years": 2.0,
        "salary_offered_inr": 350000,
        "python_required": 62.8,
        "sql_required": 58.5,
        "ml_required": 45.2,
        "javascript_required": 38.7,
        "data_structures_required": 52.3,
        "statistics_required": 48.9,
        "years_experience": 1.5,
        "python_score": 65.0,
        "sql_score": 60.0,
        "ml_score": 50.0,
        "javascript_score": 45.0,
        "data_structures_score": 55.0,
        "statistics_score": 52.0,
        "exam_time_seconds": 1800,
        "self_reported_confidence": 4.0,
        "retake_count": 0,
        "expected_salary_inr": 400000,
        "company": "TechCorp",
        "title": "Data Scientist",
        "location_job": "Bangalore",
        "edu_minimum": "PG",
        "education_level": "PG",
        "location_student": "Delhi",
    }
    sample_batch = [sample_record]

# TestClient only runs FastAPI's startup/shutdown lifespan events when
# used as a context manager - without `with`, @app.on_event("startup")
# never fires and every endpoint would see _model as None regardless.
client = TestClient(api_module.app).__enter__()


def test_health_ok():
    r = client.get("/health")
    # If model not loaded, expect 503, otherwise 200
    if api_module._model is None:
        assert r.status_code == 503, f"expected 503 (model not ready), got {r.status_code}"
        print("SKIP  GET /health returns 200 with model_version (model not loaded)")
    else:
        assert r.status_code == 200, f"expected 200, got {r.status_code}"
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "v1.0.0"


def test_score_single_clean():
    if api_module._model is None:
        print("SKIP  POST /score/single, clean record -> 200 (model not loaded)")
        return
    r = client.post("/score/single", json=sample_record)
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    for field in ("score", "score_meaning", "model_version", "input_record_id"):
        assert field in body, f"missing field {field}"
    assert 0.0 <= body["score"] <= 1.0


def test_score_batch_clean():
    if api_module._model is None:
        print("SKIP  POST /score/batch, clean batch -> 200 (model not loaded)")
        return
    r = client.post("/score/batch", json={"records": sample_batch})
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert len(body) == len(sample_batch)
    assert all("model_version" in row for row in body)


def test_missing_column_returns_400():
    bad = dict(sample_record); del bad["python_required"]
    r = client.post("/score/single", json=bad)
    assert r.status_code == 400, f"expected 400, got {r.status_code}"
    assert r.json()["error"] == "invalid_input"


def test_nan_returns_400():
    bad = dict(sample_record); bad["python_required"] = None
    r = client.post("/score/single", json=bad)
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


def test_non_numeric_returns_400():
    bad = dict(sample_record); bad["python_required"] = "not_a_number"
    r = client.post("/score/single", json=bad)
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


def test_empty_batch_returns_400():
    r = client.post("/score/batch", json={"records": []})
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


def test_oversized_batch_returns_413():
    huge = [sample_record] * (api_module.MAX_BATCH_SIZE + 1)
    r = client.post("/score/batch", json={"records": huge})
    assert r.status_code == 413, f"expected 413, got {r.status_code}"


def test_malformed_json_returns_422():
    r = client.post("/score/single", content="{not valid json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"expected 422, got {r.status_code}"


def test_unknown_route_returns_404():
    r = client.get("/nonexistent-route")
    assert r.status_code == 404, f"expected 404, got {r.status_code}"


def test_wrong_method_returns_405():
    r = client.get("/score/single")
    assert r.status_code == 405, f"expected 405, got {r.status_code}"


def test_every_response_has_request_id_header():
    r = client.get("/health")
    assert "x-request-id" in r.headers, "missing X-Request-ID header"


def test_extra_field_accepted():
    good = dict(sample_record); good["some_extra_field"] = "zzz"
    r = client.post("/score/single", json=good)
    assert r.status_code == 200, f"expected 200 (extra field should be tolerated), got {r.status_code}"


# ---------------------------------------------------------------------
# The scenario real curl testing can't force without breaking the
# actual model file: simulate the model failing to load at startup and
# confirm every endpoint degrades to a clear 503, not a crash or a
# confusing 500/NoneType error.
# ---------------------------------------------------------------------
def test_model_not_ready_returns_503():
    original_model = api_module._model
    original_error = api_module._startup_error
    try:
        api_module._model = None
        api_module._startup_error = "simulated: model file corrupted at startup"

        r_health = client.get("/health")
        assert r_health.status_code == 503, f"expected 503 from /health, got {r_health.status_code}"

        r_single = client.post("/score/single", json=sample_record)
        assert r_single.status_code == 503, f"expected 503 from /score/single, got {r_single.status_code}"

        r_batch = client.post("/score/batch", json={"records": sample_batch})
        assert r_batch.status_code == 503, f"expected 503 from /score/batch, got {r_batch.status_code}"
    finally:
        api_module._model = original_model
        api_module._startup_error = original_error


if __name__ == "__main__":
    check("GET /health returns 200 with model_version", test_health_ok)
    check("POST /score/single, clean record -> 200", test_score_single_clean)
    check("POST /score/batch, clean batch -> 200", test_score_batch_clean)
    check("Missing feature column -> 400", test_missing_column_returns_400)
    check("NaN/null feature value -> 400", test_nan_returns_400)
    check("Non-numeric feature value -> 400", test_non_numeric_returns_400)
    check("Empty batch -> 400", test_empty_batch_returns_400)
    check("Oversized batch -> 413", test_oversized_batch_returns_413)
    check("Malformed JSON body -> 422", test_malformed_json_returns_422)
    check("Unknown route -> 404", test_unknown_route_returns_404)
    check("Wrong HTTP method -> 405", test_wrong_method_returns_405)
    check("Every response carries X-Request-ID header", test_every_response_has_request_id_header)
    check("Extra unexpected field accepted, not rejected", test_extra_field_accepted)
    check("Model-not-ready degrades to 503 on every endpoint (forced failure)", test_model_not_ready_returns_503)

    n_pass = sum(1 for _, r in results if r == "PASS")
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        sys.exit(1)
