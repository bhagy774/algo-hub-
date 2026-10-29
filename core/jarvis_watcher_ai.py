#!/usr/bin/env python3
"""
JARVIS Pipeline Architecture - Layer 1: The Watcher AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Continuously monitors the CognitiveBus 'THOUGHTS' topic where ALL parts
(Part2_Neural, Part3_Institutional, Part5_Fusion, Part8_Pattern, Part11_Confidence)
publish their analysis results as text strings.

Watcher parses these text messages → extracts key signals → builds 
a compressed 'Smart Context Packet' for the Specialist AI pool.

Real message format from parts:
  Part2:  "Neural Network Predictions: BULLISH (Avg score: 0.142)"
  Part3:  "Institutional MTF Analysis: BULLISH (Consensus: 0.55). Dominant Regime: trending"
  Part5:  "Fusion Engine (MTF): Consensus = 0.32, Direction = CALL, Confidence = 74.2%"
  Part8:  "Pattern Recognition: ... BEARISH ..."
  Part11: "Confidence Engine: Final Score = 82.5%. VALID | LegacyLLM Adj: +3"
"""

import re
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("WatcherAI")


class JarvisWatcherAI:
    def __init__(self, bus):
        self.bus = bus
        self.is_running = False
        self.thread = None

        # Rolling Context — stores parsed data from Parts
        self._lock = threading.Lock()
        self._part_snapshots: Dict[str, Dict[str, Any]] = {}
        # e.g. {
        #   "Part2_Neural":   {"direction": "BULLISH", "score": 0.142, "ts": datetime},
        #   "Part3_Institutional": {"direction": "BULLISH", "consensus": 0.55, "ts": ...},
        #   "Part5_Fusion":   {"direction": "CALL", "confidence": 74.2, "ts": ...},
        #   "Part8_Pattern":  {"direction": "BEARISH", "ts": ...},
        #   "Part11_Confidence": {"score": 82.5, "valid": True, "ts": ...},
        # }

        # Health events from HEALTH topic
        self._recent_errors: List[Dict] = []

        # Subscribe to ALL topics Parts use
        self.bus.subscribe('THOUGHTS', self._on_thoughts)
        self.bus.subscribe('HEALTH',   self._on_health)
        self.bus.subscribe('SIGNALS',  self._on_signals)   # future-proof

        logger.info("WatcherAI subscribed to THOUGHTS + HEALTH + SIGNALS topics")

    # ─────────────────────────────────────────
    # Callbacks from CognitiveBus
    # ─────────────────────────────────────────

    def _on_thoughts(self, message: dict):
        """
        Called every time any Part publishes to 'THOUGHTS'.
        Parses the text string and stores structured snapshot.
        """
        sender  = message.get('sender', 'UNKNOWN')
        payload = message.get('payload', '')
        if isinstance(payload, dict):
            payload = str(payload)

        parsed = self._parse_thought(sender, payload)
        if parsed:
            with self._lock:
                self._part_snapshots[sender] = {
                    **parsed,
                    'raw': payload[:120],
                    'ts': datetime.now()
                }

    def _on_health(self, message: dict):
        """Capture error/health events to reflect in Smart Packet risk level."""
        payload = message.get('payload', {})
        with self._lock:
            self._recent_errors.append({
                'sender': message.get('sender', '?'),
                'severity': payload.get('severity', 'WARNING') if isinstance(payload, dict) else 'WARNING',
                'ts': datetime.now()
            })
            # Keep only last 20 errors
            if len(self._recent_errors) > 20:
                self._recent_errors = self._recent_errors[-20:]

    def _on_signals(self, message: dict):
        """Future-proof: handle structured SIGNALS topic."""
        payload = message.get('payload', {})
        if isinstance(payload, dict) and 'direction' in payload:
            with self._lock:
                self._part_snapshots['SIGNALS_DIRECT'] = {
                    'direction': payload.get('direction', 'NEUTRAL'),
                    'confidence': payload.get('confidence', 0),
                    'ts': datetime.now()
                }

    # ─────────────────────────────────────────
    # Text Parser — extracts signal data from Part messages
    # ─────────────────────────────────────────

    def _parse_thought(self, sender: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse thought text from Parts into structured data.
        Works for all known Part message formats.
        """
        result = {}
        text_upper = text.upper()

        # ── Direction ─────────────────────────────
        # Matches: BULLISH, BEARISH, NEUTRAL, CALL, PUT, NO TRADE, NO-TRADE
        dir_match = re.search(
            r'\b(BULLISH|BEARISH|NEUTRAL|CALL|PUT|NO[- ]TRADE)\b',
            text_upper
        )
        if dir_match:
            raw_dir = dir_match.group(1).replace(' ', '_').replace('-', '_')
            # Normalize: CALL→BULLISH, PUT→BEARISH for trend aggregation
            direction_map = {
                'CALL': 'BULLISH', 'PUT': 'BEARISH',
                'NO_TRADE': 'NEUTRAL', 'BULLISH': 'BULLISH',
                'BEARISH': 'BEARISH', 'NEUTRAL': 'NEUTRAL'
            }
            result['direction'] = direction_map.get(raw_dir, 'NEUTRAL')
            result['raw_direction'] = raw_dir

        # ── Confidence / Score ────────────────────
        # Matches: "Confidence = 74.2%" or "Final Score = 82.5%"
        conf_match = re.search(r'(?:confidence|score|avg score)[^\d]*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))

        # ── Consensus ────────────────────────────
        # Matches: "Consensus = 0.55" or "Consensus: 0.32"
        cons_match = re.search(r'consensus[^\d]*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        if cons_match:
            result['consensus'] = float(cons_match.group(1))

        # ── Regime ────────────────────────────────
        regime_match = re.search(r'regime[:\s]+(\w+)', text, re.IGNORECASE)
        if regime_match:
            result['regime'] = regime_match.group(1)

        # ── Valid/Invalid ─────────────────────────
        if 'VALID' in text_upper and 'INVALID' not in text_upper:
            result['valid'] = True
        elif 'INVALID' in text_upper:
            result['valid'] = False

        return result if result else None

    # ─────────────────────────────────────────
    # Smart Packet Generator
    # ─────────────────────────────────────────

    def generate_smart_packet(self, trigger_event=None) -> Dict[str, Any]:
        """
        Aggregates all Part snapshots into a compressed Smart Context Packet.
        This is what the Specialist AIs receive instead of raw noisy data.
        """
        now = datetime.now()
        stale_cutoff = now - timedelta(seconds=120)   # Ignore data older than 2 min

        with self._lock:
            # ── Filter stale snapshots ────────────
            active = {
                k: v for k, v in self._part_snapshots.items()
                if v.get('ts', datetime.min) > stale_cutoff
            }

            # ── Aggregate direction votes ─────────
            direction_votes = [
                v['direction'] for v in active.values()
                if 'direction' in v
            ]
            # Parts that say VALID (e.g. Part11_Confidence) act as BULLISH confirmers
            # because a 'VALID' result means the existing dominant signal is confirmed.
            valid_confirmations = sum(
                1 for v in active.values()
                if v.get('valid') is True and 'direction' not in v
            )
            bullish = direction_votes.count('BULLISH') + valid_confirmations
            bearish = direction_votes.count('BEARISH')
            neutral = direction_votes.count('NEUTRAL')
            total_votes = len(direction_votes) + valid_confirmations

            if bullish > bearish and bullish > neutral:
                dominant_trend = 'BULLISH'
            elif bearish > bullish and bearish > neutral:
                dominant_trend = 'BEARISH'
            else:
                dominant_trend = 'NEUTRAL'

            # ── Average confidence ────────────────
            conf_scores = []
            for v in active.values():
                c = v.get('confidence')
                if c is not None and isinstance(c, (int, float)):
                    # Normalise: if value <= 1.5, treat as 0-1 ratio and scale to %
                    conf_scores.append(c * 100.0 if c <= 1.5 else c)
            avg_confidence = round(sum(conf_scores) / len(conf_scores), 1) if conf_scores else 0.0

            # ── Health ────────────────────────────
            recent_errors = [
                e for e in self._recent_errors
                if e['ts'] > stale_cutoff
            ]
            critical_count = sum(1 for e in recent_errors if e.get('severity') == 'CRITICAL')
            system_health = 'CRITICAL' if critical_count > 0 else 'DEGRADED' if recent_errors else 'HEALTHY'

            # ── Part summary ──────────────────────
            part_summary = {}
            for part, data in active.items():
                part_summary[part] = {
                    'direction': data.get('direction', 'UNKNOWN'),
                    'confidence': data.get('confidence'),
                    'raw': data.get('raw', '')
                }

        packet = {
            "timestamp":        now.isoformat(),
            "trigger_event":    trigger_event or "WATCHER_POLL",
            "active_parts":     len(active),
            "total_votes":      total_votes,
            "system_health":    system_health,
            "recent_error_count": len(recent_errors),
            "aggregate": {
                "dominant_trend":   dominant_trend,
                "bullish_votes":    bullish,
                "bearish_votes":    bearish,
                "neutral_votes":    neutral,
                "avg_confidence":   avg_confidence,
            },
            "part_snapshots":   part_summary
        }

        logger.debug(
            f"SmartPacket: {dominant_trend} ({bullish}B/{bearish}Be/{neutral}N) "
            f"conf={avg_confidence}% parts={len(active)} health={system_health}"
        )
        return packet

    # ─────────────────────────────────────────
    # Background Watcher Loop
    # ─────────────────────────────────────────

    def _watcher_loop(self):
        logger.info("WatcherAI background loop started — monitoring all Parts via THOUGHTS bus")
        while self.is_running:
            try:
                # Prune very old errors every cycle
                cutoff = datetime.now() - timedelta(seconds=300)
                with self._lock:
                    self._recent_errors = [
                        e for e in self._recent_errors if e['ts'] > cutoff
                    ]
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Watcher loop error: {e}")
                time.sleep(5.0)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(
                target=self._watcher_loop,
                daemon=True,
                name="WatcherAI_Thread"
            )
            self.thread.start()
            logger.info("WatcherAI ONLINE — listening to Part2, Part3, Part5, Part8, Part11 THOUGHTS")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("WatcherAI stopped.")
