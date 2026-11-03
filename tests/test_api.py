"""
test_api.py — API endpoint tests.
"""

from __future__ import annotations

import json
import os
import pytest
from app import app


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"


class TestReviewEndpoint:
    def test_review_json_body(self, client):
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
'''
        res = client.post("/api/review",
                          data=json.dumps({"code": code}),
                          content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert "score" in data
        assert "findings" in data
        assert data["grade"] is not None

    def test_review_empty_code(self, client):
        res = client.post("/api/review",
                          data=json.dumps({"code": ""}),
                          content_type="application/json")
        assert res.status_code == 400

    def test_review_clean_code(self, client):
        code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
'''
        res = client.post("/api/review",
                          data=json.dumps({"code": code}),
                          content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert data["score"] == 100
        assert data["grade"] == "A"

    def test_review_syntax_error(self, client):
        res = client.post("/api/review",
                          data=json.dumps({"code": "def broken(:\n  pass"}),
                          content_type="application/json")
        assert res.status_code == 400

    def test_review_no_body(self, client):
        res = client.post("/api/review", content_type="application/json")
        assert res.status_code == 400


class TestProveEndpoint:
    def test_prove_returns_scores(self, client):
        res = client.post("/api/prove")
        assert res.status_code == 200
        data = res.get_json()
        assert "leaky_score" in data
        assert "correct_score" in data
        assert "gap" in data
        assert data["leaky_score"] >= data["correct_score"]


class TestFallbackWithoutApiKey:
    def test_review_works_without_gemini_key(self, client):
        """App must work fully without GEMINI_API_KEY."""
        # Ensure no key is set
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            code = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
'''
            res = client.post("/api/review",
                              data=json.dumps({"code": code}),
                              content_type="application/json")
            assert res.status_code == 200
            data = res.get_json()
            assert data["meta"]["llm_used"] is False
            # Should still have findings
            assert len(data["findings"]) > 0
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
