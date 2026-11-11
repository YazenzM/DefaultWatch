import pytest
from fastapi.testclient import TestClient

from defaultwatch import api
from defaultwatch.pipeline import bundle_path


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(bundle_path(trained["cfg"])))
    monkeypatch.delenv("MODEL_URI", raising=False)
    api.get_bundle.cache_clear()
    yield TestClient(api.app)
    api.get_bundle.cache_clear()


RISKY = {"credit_score": 640, "dti": 47, "ltv": 95, "loan_purpose": "cash_out_refinance",
         "occupancy": "investment", "property_type": "condo"}


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["synthetic"] is True and "credit_score" in info["features"]


def test_score_declines_with_reasons(client):
    r = client.post("/score", json=RISKY)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "decline" and body["pd"] > body["cutoff_pd"]
    assert body["reasons"] and body["reasons"][0]["rank"] == 1
    assert body["synthetic_model"] is True


def test_validation_and_batch(client):
    assert client.post("/score", json={**RISKY, "credit_score": 900}).status_code == 422
    assert client.post("/score", json={**RISKY, "occupancy": "boat"}).status_code == 422
    safe = {"credit_score": 800, "dti": 20, "ltv": 60, "loan_term_months": 180}
    r = client.post("/score/batch", json={"applications": [RISKY, safe, {"ltv": 80}]})
    assert r.status_code == 200
    out = r.json()
    assert [o["decision"] for o in out[:2]] == ["decline", "approve"]
    assert out[2]["pd"] > 0  # missing credit score and DTI are scored, not rejected


def test_no_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    api.get_bundle.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "no_model"
    assert c.post("/score", json=RISKY).status_code == 503
    api.get_bundle.cache_clear()
