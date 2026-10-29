import pytest
from jarvis_lot_limits import enforce_entry_lots

def test_bounds_and_cap(monkeypatch):
 monkeypatch.setenv('JARVIS_MIN_ENTRY_LOTS','4'); monkeypatch.setenv('JARVIS_MAX_ENTRY_LOTS','8')
 assert enforce_entry_lots(4)==4; assert enforce_entry_lots(8)==8; assert enforce_entry_lots(99)==8

def test_below_rejected(monkeypatch):
 monkeypatch.setenv('JARVIS_MIN_ENTRY_LOTS','4'); monkeypatch.setenv('JARVIS_MAX_ENTRY_LOTS','8')
 with pytest.raises(ValueError): enforce_entry_lots(3)

def test_invalid_config(monkeypatch):
 monkeypatch.setenv('JARVIS_MIN_ENTRY_LOTS','9'); monkeypatch.setenv('JARVIS_MAX_ENTRY_LOTS','8')
 with pytest.raises(ValueError): enforce_entry_lots(8)

def test_exit_not_constrained(monkeypatch):
 monkeypatch.setenv('JARVIS_MIN_ENTRY_LOTS','4'); monkeypatch.setenv('JARVIS_MAX_ENTRY_LOTS','8')
 assert enforce_entry_lots(1, reduce_only=True)==1

def test_contract_step_and_balance(monkeypatch):
 monkeypatch.setenv('JARVIS_MIN_ENTRY_LOTS','4'); monkeypatch.setenv('JARVIS_MAX_ENTRY_LOTS','8')
 assert enforce_entry_lots(7, metadata={'contract_step':2}, available_balance=1)==6
 with pytest.raises(ValueError): enforce_entry_lots(4, metadata={'contract_step':5}, available_balance=1)
 with pytest.raises(ValueError): enforce_entry_lots(4, available_balance=0)
