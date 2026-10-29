import os, unittest
from unittest.mock import patch
from jarvis_laya_advisor import advise, request_runtime_advisory

class RuntimeTests(unittest.TestCase):
 def test_advice_isolation_and_no_ollama_call(self):
  snapshot={'symbol':'BTCUSDT','direction':'SELL','confidence':71}; before=dict(snapshot); calls=[]
  def fake(s): calls.append(s); return {'suggestion':'BUY','confidence':.7}
  result=advise(snapshot,symbol='BTCUSDT',deterministic_decision='SELL',predictor=fake,enabled=True)
  self.assertEqual(snapshot,before); self.assertEqual(result.deterministic_decision,'SELL'); self.assertEqual(result.suggestion,'BUY'); self.assertEqual(len(calls),1)
  # Adapter/runtime path is isolated: it never imports or invokes Ollama.
  self.assertNotIn('ollama', calls[0])
 def test_disabled_never_calls_predictor(self):
  r=advise({'symbol':'BTCUSDT'},symbol='BTCUSDT',deterministic_decision='NO_TRADE',predictor=lambda _:self.fail('called'),enabled=False)
  self.assertEqual(r.status,'disabled')
 def test_request_returns_immediately_and_does_not_mutate_snapshot(self):
  # Explicitly exercise the actual runtime wrapper while faking slow predictor.
  import jarvis_laya_advisor as m
  gate=__import__('threading').Event(); original=m.advise
  def slow(s,**kw): gate.wait(.2); return original(s,**kw)
  snapshot={'symbol':'BTCUSDT','direction':'SELL'}; before=dict(snapshot)
  with patch.dict(os.environ, {'JARVIS_LAYA_ADVISORY':'true'}), patch.object(m,'advise',side_effect=slow):
   r=request_runtime_advisory(snapshot,symbol='BTCUSDT',deterministic_decision='SELL')
  self.assertEqual(r.status,'pending'); self.assertEqual(snapshot,before); gate.set()

 def test_upstream_router_shape_with_fake_only(self):
  import sys, types, jarvis_laya_advisor as m
  seen=[]
  class Router:
   def predict(self,state,questions):
    seen.append((state,questions))
    return {'answers':{'suggestion':{'choice':'NO_TRADE','confidence':.4}}}
  with patch.dict(sys.modules,{'laya':types.SimpleNamespace(Router=Router)}):
   response=m._predict_with_laya({'symbol':'ETHUSDT'})
  self.assertEqual(response['suggestion'],'NO_TRADE')
  self.assertEqual(response['confidence'],.4)
  self.assertIn('BUY',seen[0][1]['suggestion']['criteria'])

 def test_timeout_never_turns_into_confirmation(self):
  import time, jarvis_laya_advisor as m
  prior=dict(m._state)
  try:
   with m._lock:
    m._state.update(busy=True,started=time.monotonic()-m.TIMEOUT_SECONDS-1,
                    fingerprint=None,result=None)
   with patch.dict(os.environ,{'JARVIS_LAYA_ADVISORY':'true'}):
    r=m.request_runtime_advisory({'symbol':'BTCUSDT'},symbol='BTCUSDT',deterministic_decision='BUY')
   self.assertEqual(r.status,'timeout')
   self.assertIsNone(r.suggestion)
  finally:
   with m._lock:m._state.update(prior)

 def test_cached_result_requires_exact_snapshot_and_is_consumed_once(self):
  import time, threading, jarvis_laya_advisor as m
  before=dict(m._state)
  snapshot={'symbol':'BTCUSDT','direction':'BUY','price':100}
  result=m.Advisory('ok','BTCUSDT','BUY','SELL',.6)
  try:
   with m._lock:
    m._state.update(busy=False, fingerprint=m._fingerprint(snapshot,'BTCUSDT','BUY'),
                    result=result, completed=time.monotonic())
   # Prevent any new inference worker from running; inspect cached response only.
   with patch.dict(os.environ,{'JARVIS_LAYA_ADVISORY':'true'}), patch.object(threading.Thread,'start',return_value=None):
    self.assertEqual(m.request_runtime_advisory({'symbol':'BTCUSDT','direction':'BUY','price':101},symbol='BTCUSDT',deterministic_decision='BUY').status,'pending')
    self.assertEqual(m.request_runtime_advisory(snapshot,symbol='ETHUSDT',deterministic_decision='BUY').status,'invalid')
   # The changed snapshot invalidated the cached result; stale advice cannot be returned.
   with m._lock:m._state.update(busy=False, fingerprint=m._fingerprint(snapshot,'BTCUSDT','BUY'),
                                result=result, completed=time.monotonic())
   with patch.dict(os.environ,{'JARVIS_LAYA_ADVISORY':'true'}), patch.object(threading.Thread,'start',return_value=None):
    self.assertEqual(m.request_runtime_advisory(snapshot,symbol='BTCUSDT',deterministic_decision='BUY').suggestion,'SELL')
    self.assertEqual(m.request_runtime_advisory(snapshot,symbol='BTCUSDT',deterministic_decision='BUY').status,'pending')
  finally:
   with m._lock:m._state.update(before)

 def test_thread_start_failure_does_not_leave_busy_advisory(self):
  import threading, jarvis_laya_advisor as m
  prior=dict(m._state)
  try:
   with m._lock:m._state.update(busy=False,result=None)
   with patch.dict(os.environ,{'JARVIS_LAYA_ADVISORY':'true'}), patch.object(threading.Thread,'start',side_effect=RuntimeError('no thread')):
    r=m.request_runtime_advisory({'symbol':'BTCUSDT'},symbol='BTCUSDT',deterministic_decision='NO_TRADE')
   self.assertEqual(r.status,'unavailable')
   self.assertFalse(m._state['busy'])
  finally:
   with m._lock:m._state.update(prior)

if __name__=='__main__': unittest.main()
