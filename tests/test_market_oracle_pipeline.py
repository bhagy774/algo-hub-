#!/usr/bin/env python3
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
"""
Comprehensive Integration Test for JARVIS Market Oracle System
"""
import time
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_pipeline():
    print("=" * 70)
    print("🚀 RUNNING FULL END-TO-END JARVIS MARKET ORACLE PIPELINE TEST")
    print("=" * 70)

    # 1. Test Binance Microstructure
    print("\n[TEST 1] Testing Binance Microstructure...")
    from binance_data import get_binance_data
    bd = get_binance_data()
    spot = bd.get_live_price("BTCUSDT")
    funding = bd.get_funding_rate("BTCUSDT")
    depth = bd.get_order_book_depth("BTCUSDT", limit=20)
    oi = bd.get_open_interest("BTCUSDT")
    stats = bd.get_24h_stats("BTCUSDT")
    print(f"  ✓ Spot: ${spot:,.2f}")
    print(f"  ✓ Funding: {funding.get('description')} (Next: {funding.get('countdown_str')})")
    print(f"  ✓ Depth Imbalance: {depth.get('imbalance_pct'):+.2f}% ({depth.get('bias')})")
    print(f"  ✓ Binance OI: {oi.get('open_interest'):,.2f} contracts")
    print(f"  ✓ 24h Vol: ${stats.get('volume_24h_usdt', 0)/1e9:.2f}B")

    # 2. Test Deribit Options Client
    print("\n[TEST 2] Testing Deribit Options Client...")
    from deribit_options_client import DeribitOptionsClient
    deribit = DeribitOptionsClient(currency="BTC")
    chain = deribit.get_option_chain()
    greeks = deribit.get_greeks_and_iv()
    if chain:
        print(f"  ✓ Deribit Max Pain: ${chain.get('max_pain'):,}")
        print(f"  ✓ Deribit PCR: {chain.get('pcr'):.3f}")
        print(f"  ✓ Gamma Walls: Call ${chain.get('resistance_wall'):,} | Put ${chain.get('support_wall'):,}")
    if greeks:
        print(f"  ✓ Deribit ATM IV: {greeks.get('call_iv'):.1f}% | Gamma: {greeks.get('call_gamma')}")

    # 3. Test Delta Exchange
    print("\n[TEST 3] Testing Delta Exchange Options...")
    from delta_api_wrapper import DeltaExchangeData
    delta = DeltaExchangeData()
    d_chain = delta.get_options_chain("BTC")
    print(f"  ✓ Delta Total OI: {d_chain.get('total_oi', 0):,.2f} BTC | PCR: {d_chain.get('pcr', 0):.3f}")

    # 4. Test CognitiveBus Pub/Sub with ORACLE_FORECAST
    print("\n[TEST 4] Testing CognitiveBus ORACLE_FORECAST channel...")
    from jarvis_cognitive_bus import CognitiveBus
    bus = CognitiveBus()
    received_msgs = []
    bus.subscribe("ORACLE_FORECAST", lambda m: received_msgs.append(m))
    test_payload = {"status": "ok", "test": True, "ts": datetime.now().isoformat()}
    bus.publish("ORACLE_FORECAST", "OracleTest", test_payload)
    latest_fc = bus.get_latest_oracle_forecast()
    assert len(received_msgs) == 1, "Failed to receive message on ORACLE_FORECAST"
    assert latest_fc.get("test") is True, "Failed to retrieve latest oracle forecast from bus"
    print("  ✓ CognitiveBus ORACLE_FORECAST pub/sub verified")

    # 5. Test Oracle Trade Gate (Gate 0.5)
    print("\n[TEST 5] Testing OracleTradeGate...")
    from oracle_trade_gate import OracleTradeGate
    class DummyOracle:
        def get_latest_forecast(self):
            return {
                "model_used": "gemini-3.6-flash",
                "5min": {"direction": "BULLISH", "confidence": 80},
                "30min": {"direction": "BULLISH", "confidence": 75},
                "trade_suggestion": "CALL",
                "entry_zone": {"price_from": spot * 0.999, "price_to": spot * 1.002},
                "exit_target": round(spot * 1.008, 1),
                "stop_loss": round(spot * 0.995, 1),
                "hold_minutes": 20,
                "gemini_summary": "Bullish momentum aligned"
            }

    gate = OracleTradeGate(oracle_ref=DummyOracle())
    call_ok, call_r = gate.is_trade_aligned("CALL")
    put_ok, put_r = gate.is_trade_aligned("PUT")
    in_zone, z_r = gate.is_price_in_entry_zone(spot)
    tp_sl = gate.get_oracle_tp_sl(spot, "CALL")

    assert call_ok is True, "CALL should be allowed"
    assert put_ok is False, "PUT should be blocked when Oracle is BULLISH"
    assert in_zone is True, "Spot price should be in entry zone"
    assert tp_sl.get("use_oracle") is True, "Oracle TP/SL should be valid"
    print(f"  ✓ Gate 0.5 CALL Allowed: {call_ok} ({call_r})")
    print(f"  ✓ Gate 0.5 PUT Blocked: {not put_ok} ({put_r})")
    print(f"  ✓ Gate 0.5 Entry Zone: {in_zone} ({z_r})")
    print(f"  ✓ Gate 0.5 Dynamic TP/SL: TP=${tp_sl['tp_price']:,} / SL=${tp_sl['sl_price']:,}")

    # 7. Test Market Oracle Forecast & Console Rendering
    print("\n[TEST 7] Testing JARVIS Market Oracle Instance...")
    from jarvis_market_oracle import JarvisMarketOracle
    oracle = JarvisMarketOracle(bus=bus)
    cached_fc = oracle.get_latest_forecast()
    print(f"  ✓ Initial forecast state ready: {cached_fc.get('trade_suggestion')}")

    print("\n" + "=" * 70)
    print("✅ ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
