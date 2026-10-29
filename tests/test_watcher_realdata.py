#!/usr/bin/env python3
"""
WATCHER AI - Real Parts Data Test
Tests EXACT messages that Part2, Part3, Part5, Part8, Part11 publish to bus.
"""
import sys, time
from datetime import datetime

print('='*55)
print('  WATCHER AI - REAL PARTS DATA TEST')
print('='*55)

# Mock CognitiveBus
class MockBus:
    def __init__(self): self.subscribers = {}
    def subscribe(self, topic, cb):
        self.subscribers.setdefault(topic, []).append(cb)
    def publish(self, topic, sender, payload):
        for cb in self.subscribers.get(topic, []):
            cb({'topic': topic, 'sender': sender, 'payload': payload})

bus = MockBus()
from jarvis_watcher_ai import JarvisWatcherAI
watcher = JarvisWatcherAI(bus=bus)

# ── Simulate EXACT messages that real Parts publish ──
real_messages = [
    ('Part2_Neural',        'Neural Network Predictions: BULLISH (Avg score: 0.142)'),
    ('Part3_Institutional', 'Institutional MTF Analysis: BULLISH (Consensus: 0.55). Dominant Regime: trending'),
    ('Part5_Fusion',        'Fusion Engine (MTF): Consensus = 0.32, Direction = CALL, Confidence = 74.2%'),
    ('Part8_Pattern',       'Pattern Recognition: Strong BEARISH setup detected. Score: 0.68'),
    ('Part11_Confidence',   'Confidence Engine: Final Score = 82.5%. VALID | LegacyLLM Adj: +3'),
]

print()
print('[Step 1] Simulating THOUGHTS messages from all 5 Parts...')
for sender, msg in real_messages:
    bus.publish('THOUGHTS', sender, msg)
    print('  -> ' + sender + ': ' + msg[:60])

# Also simulate HEALTH error from Part7
bus.publish('HEALTH', 'Part7_Live', {'severity': 'WARNING', 'error_msg': 'Timeout'})
print('  -> Part7_Live: [HEALTH] WARNING - Timeout')

print()
print('[Step 2] Checking parsed snapshots...')
for part, snap in watcher._part_snapshots.items():
    direction   = snap.get('direction', 'N/A')
    confidence  = snap.get('confidence', 'N/A')
    consensus   = snap.get('consensus', 'N/A')
    valid_flag  = snap.get('valid', 'N/A')
    print('  ' + part + ':')
    print('     direction=' + str(direction) + ', confidence=' + str(confidence) +
          ', consensus=' + str(consensus) + ', valid=' + str(valid_flag))

print()
print('[Step 3] Generating Smart Packet...')
packet = watcher.generate_smart_packet(trigger_event='LIVE_TRADE_SIGNAL')
agg = packet['aggregate']

print()
print('  +==============================================+')
print('  |         SMART CONTEXT PACKET               |')
print('  +==============================================+')
print('  |  Active Parts   : ' + str(packet['active_parts']))
print('  |  Dominant Trend : ' + str(agg['dominant_trend']))
print('  |  BULLISH votes  : ' + str(agg['bullish_votes']))
print('  |  BEARISH votes  : ' + str(agg['bearish_votes']))
print('  |  NEUTRAL votes  : ' + str(agg['neutral_votes']))
print('  |  Avg Confidence : ' + str(agg['avg_confidence']) + '%')
print('  |  System Health  : ' + str(packet['system_health']))
print('  |  Error Count    : ' + str(packet['recent_error_count']))
print('  |  Trigger        : ' + str(packet['trigger_event']))
print('  +==============================================+')

print()
print('[Step 4] Assertions...')

assert packet['active_parts'] == 5, 'Expected 5 parts, got ' + str(packet['active_parts'])
print('  PASS: All 5 Parts data captured')

assert agg['dominant_trend'] == 'BULLISH', 'Expected BULLISH, got ' + str(agg['dominant_trend'])
print('  PASS: Dominant trend = BULLISH (4 bullish vs 1 bearish vote)')

assert agg['bullish_votes'] == 4, 'Expected 4 bullish (3 direction + 1 VALID confirm), got ' + str(agg['bullish_votes'])
print('  PASS: Bullish votes = 4 (Part2+Part3+Part5 direction votes + Part11 VALID confirmation)')

assert agg['bearish_votes'] == 1
print('  PASS: Bearish votes = 1 (Part8_Pattern BEARISH)')

assert agg['avg_confidence'] > 0
print('  PASS: Avg confidence = ' + str(agg['avg_confidence']) + '%')

assert packet['system_health'] == 'DEGRADED'
print('  PASS: Health = DEGRADED (Part7 warning captured)')

print()
print('[Step 5] Testing regex parser on edge cases...')
edge_cases = [
    ('PartX', 'NO TRADE signal. Confidence = 45.0%'),
    ('PartY', 'PUT direction confirmed. Score: 0.89'),
    ('PartZ', 'Market is NEUTRAL, consensus: 0.02'),
]
for sender, msg in edge_cases:
    bus.publish('THOUGHTS', sender, msg)

parsed_x = watcher._part_snapshots.get('PartX', {})
assert parsed_x.get('direction') == 'NEUTRAL', 'NO TRADE should map to NEUTRAL, got ' + str(parsed_x)
print('  PASS: "NO TRADE" -> NEUTRAL mapping')

parsed_y = watcher._part_snapshots.get('PartY', {})
assert parsed_y.get('direction') == 'BEARISH', 'PUT should map to BEARISH, got ' + str(parsed_y)
print('  PASS: "PUT" -> BEARISH mapping')

print()
print('='*55)
print('  ALL TESTS PASSED!')
print('  Watcher correctly reads ALL 5 Parts THOUGHTS data')
print('  and compresses into Smart Packet for Specialist AIs')
print('='*55)
