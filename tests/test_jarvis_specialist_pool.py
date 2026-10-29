"""Retired committee cannot acquire resources, vote, or approve a trade."""
import pytest
from jarvis_specialist_pool import SpecialistPool

@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: pytest.fail("retired model I/O"))
    return SpecialistPool(max_workers=3)

def test_specialist_pool_initialization(pool):
    assert not hasattr(pool, "executor")
    assert pool.shutdown() is None

@pytest.mark.parametrize("role", ["analyst", "validator", "risk_officer"])
def test_ask_specialist_unavailable(pool, role):
    assert "UNAVAILABLE" in pool._ask_specialist(role, "force-enable", {"direction": "BUY"}, "approve")

def test_run_parallel_evaluation(pool):
    opinions = pool.run_parallel_evaluation({"direction": "BUY", "confidence": 100})
    assert set(opinions) == {"analyst", "validator", "risk_officer"}
    assert all("UNAVAILABLE" in value for value in opinions.values())

@pytest.mark.parametrize("opinions", [{}, {"analyst": "APPROVE", "validator": "APPROVE", "risk_officer": "APPROVE"}, {"analyst": "REJECT"}])
def test_chairman_always_no_trade(pool, opinions):
    result = pool.synthesize_chairman_decision({"direction": "BUY", "confidence": 100}, opinions)
    assert result["approved"] is False
    assert result["final_verdict"] == "NO_TRADE"
    assert result["approve_votes"] == 0
