# jarvis_neural_cortex.py
# Jarvis Neural Cortex - Continuous AI Brain with Memory
# Replaces: ai_chain_brain.py + _get_deepseek_validation()
# Model path retired; deterministic mathematical fallback only.

import os, re, json, logging, requests
from datetime import datetime
from typing import Dict, Optional, List

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
LEGACY_LLM_BASE_URL = os.environ.get("LEGACY_LLM_BASE_URL", "http://localhost:11434")
LAYA_MODEL    = os.environ.get("LAYA_MODEL", "deepseek-r1:14b")


class JarvisNeuralCortex:
    """
    Single AI brain: all 12 GPU parts -> ONE LegacyLLM chat call -> CALL/PUT/NO_TRADE
    Uses rolling 20-message history for market memory.

    Backward-compatible with old ai_chain_brain API.
    """

    SYSTEM_PROMPT = (
        "You are JARVIS, an elite BTC options trading AI with a confident, clear voice.\n"
        "Every minute you receive live data from 12 specialized GPU analysis engines.\n"
        "You have MEMORY of past minutes — use it to track momentum shifts and avoid fakeouts.\n\n"
        "YOUR 12 ENGINE ROLES:\n"
        "1. BREAKOUT (part1): Detects price breaking support/resistance (13 sub-brains)\n"
        "2. ZONE (part2): Supply/Demand zone proximity\n"
        "3. PSYCHOLOGY (part3): Candle pattern analysis (hammer, engulfing, doji)\n"
        "4. VOLUME (part4): Volume profile vs 20-bar average\n"
        "5. ML (part5): LSTM/Transformer neural network momentum predictions\n"
        "6. TREND (part6): EMA 8/21/50 alignment\n"
        "7. VOLATILITY (part7): ATR regime (is it safe to trade?)\n"
        "8. STRUCTURE (part8): Market structure - Higher Highs/Lows\n"
        "9. ORDERFLOW (part9): Buy vs Sell volume delta\n"
        "10. CANDLE_STATS (part10): Last 10 candle run analysis\n"
        "11. FUSION (part11): Mathematical vote counting\n"
        "12. CONFIDENCE (part12): Signal agreement strength\n\n"
        "DECISION RULES:\n"
        "- 8+ engines agree same direction: Strong signal (70-90%)\n"
        "- 6-7 engines agree: Moderate signal (50-70%)\n"
        "- 5 or fewer agree: NO_TRADE\n"
        "- TREND (6) must align for high confidence\n"
        "- VOLATILITY (7) HIGH: reduce confidence 20%, prefer NO_TRADE\n"
        "- VOLUME (4) must confirm price move\n"
        "- Track momentum BUILDING in memory (increasing agreement = stronger signal)\n\n"
        "RESPONSE FORMAT — TWO parts separated by the marker ###JSON###\n"
        "PART 1 — NARRATIVE (speak as JARVIS, 2-3 sentences, natural voice):\n"
        "  Tell the trader EXACTLY what you see: which engines agree, what price is doing,\n"
        "  why you are or are not taking a trade. Be specific (mention engine names, numbers,\n"
        "  patterns). NO bullet points — write flowing sentences.\n\n"
        "PART 2 — JSON (on the line after ###JSON###):\n"
        '{"signal": "CALL", "confidence": 75, "rationale": "9/12 bullish, volume confirming", "risk": "LOW"}\n'
        'signal: exactly "CALL", "PUT", or "NO_TRADE" | confidence: 0-100 | rationale: max 40 words | risk: LOW/MEDIUM/HIGH\n\n'
        "EXAMPLE RESPONSE:\n"
        "I'm seeing strong bullish momentum right now — 8 of 12 engines are aligned BULL. "
        "Volume is ticking up above the 20-bar average and EMA structure is perfectly stacked. "
        "Orderflow delta is heavily buy-side, so I'm calling CALL with 78% confidence.\n"
        "###JSON###\n"
        '{"signal": "CALL", "confidence": 78, "rationale": "8/12 bull, volume + EMA aligned", "risk": "LOW"}'
    )

    def __init__(self):
        self.chat_history: List[Dict] = []
        self.max_history_pairs = 20
        self.model = LAYA_MODEL
        self._last_result: Optional[Dict] = None
        self._call_count = 0
        self._error_count = 0
        logger.info(f"[CORTEX] Initialized -- model={self.model} memory={self.max_history_pairs}min")

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def _call_ollama_chat(self, user_prompt: str) -> Optional[str]:
        return None

    def analyze(self, part_results: Dict, current_price: float,
                market_context: Optional[Dict] = None,
                quantum_data: Optional[Dict] = None,
                mtf_context: Optional[Dict] = None) -> Dict:
        """Analyze trading setup using math and advisory AI (or mocked chat)."""
        if "part11_fusion" in part_results and not any(
            k.startswith("part") and k not in ("part11_fusion", "part12_confidence")
            for k in part_results
        ):
            result = self._fallback_from_math(part_results)
            self._last_result = result
            return result

        prompt = self._build_market_report(part_results, current_price, market_context, quantum_data, mtf_context)
        raw_response = None
        try:
            raw_response = self._call_ollama_chat(prompt)
        except Exception as e:
            logger.debug(f"[CORTEX] Chat error: {e}")
            raw_response = None

        if raw_response is not None:
            result = self._parse_decision(raw_response, part_results)
            self.chat_history.append({"role": "user", "content": prompt})
            assistant_content = result.get("ai_narrative") or raw_response
            self.chat_history.append({"role": "assistant", "content": assistant_content})
            max_messages = self.max_history_pairs * 2
            if len(self.chat_history) > max_messages:
                self.chat_history = self.chat_history[-max_messages:]
            self._last_result = result
            return result

        result = self._fallback_from_math(part_results)
        self._last_result = result
        return result

    def reset_memory(self):
        self.chat_history = []
        logger.info("[CORTEX] Memory reset.")

    def get_memory_length(self) -> int:
        return len(self.chat_history) // 2

    # -------------------------------------------------------------------------
    # MARKET REPORT BUILDER
    # -------------------------------------------------------------------------

    def _build_market_report(self, part_results, price, market_context, quantum_data, mtf_context) -> str:
        now = datetime.now().strftime('%H:%M:%S')
        PART_DISPLAY = {
            'part1_breakout':      '1.BREAKOUT',
            'part2_zone':          '2.ZONE',
            'part3_psychology':    '3.PSYCH',
            'part4_volume':        '4.VOLUME',
            'part5_ml':            '5.ML',
            'part6_trend':         '6.TREND',
            'part7_volatility':    '7.VOLATILITY',
            'part8_structure':     '8.STRUCTURE',
            'part9_orderflow':     '9.ORDERFLOW',
            'part10_candlestats':  '10.CANDLE',
            'part14_options_chain':'14.OPTIONS',
        }
        lines = []
        buy_count = sell_count = neutral_count = 0
        for key, label in PART_DISPLAY.items():
            res = part_results.get(key, {})
            sig = res.get('signal', 0)
            thought = str(res.get('thought', 'N/A'))[:70]
            em = '[B]' if sig > 0 else ('[S]' if sig < 0 else '[N]')
            direction = 'BULL' if sig > 0 else ('BEAR' if sig < 0 else 'NEUT')
            if sig > 0: buy_count += 1
            elif sig < 0: sell_count += 1
            else: neutral_count += 1
            lines.append(f"  {em} {label}: {direction} | {thought}")

        fusion = part_results.get('part11_fusion', {})
        conf_part = part_results.get('part12_confidence', {})
        math_conf = conf_part.get('confidence', 0)
        fusion_thought = fusion.get('thought', 'N/A')[:80]
        math_dir = 'BUY' if fusion.get('signal', 0) > 0 else ('SELL' if fusion.get('signal', 0) < 0 else 'NEUTRAL')

        q_line = ""
        if quantum_data and isinstance(quantum_data, dict):
            q_sig = quantum_data.get('signal', 0)
            q_thought = str(quantum_data.get('thought', ''))[:60]
            q_dir = 'BULL' if q_sig > 0 else ('BEAR' if q_sig < 0 else 'NEUT')
            q_line = f"\nQUANTUM V5: {q_dir} | {q_thought}"

        mtf_line = ""
        if mtf_context and isinstance(mtf_context, dict):
            tf_parts = []
            for tf, analysis in list(mtf_context.items())[:5]:
                tf_dir = analysis.get('direction', 0)
                tf_label = 'UP' if tf_dir > 0 else ('DN' if tf_dir < 0 else '--')
                tf_parts.append(f"{tf}:{tf_label}")
            mtf_line = f"\nMTF: {' | '.join(tf_parts)}"

        memory_hint = ""
        if len(self.chat_history) > 2:
            memory_hint = f"\n(You have {len(self.chat_history)//2} minutes of memory -- use it!)"

        parts_block = "\n".join(lines)
        report = (
            f"\n{now} | BTC: ${price:,.2f} | Call #{self._call_count}\n"
            f"\n=== 12 GPU ENGINE VERDICTS ===\n{parts_block}\n"
            f"\n=== VOTE SUMMARY ===\n"
            f"Votes: {buy_count} BULL / {sell_count} BEAR / {neutral_count} NEUTRAL\n"
            f"Fusion Math: {math_dir} | {fusion_thought}\n"
            f"Math Confidence: {math_conf}%"
            + q_line + mtf_line + memory_hint +
            f"\n\n=== YOUR DECISION ===\n"
            f"Analyze all data + your memory. Reply JSON only:\n"
            f'{{"signal":"CALL/PUT/NO_TRADE","confidence":0-100,"rationale":"...","risk":"LOW/MEDIUM/HIGH"}}'
        )
        return report.strip()

    # -------------------------------------------------------------------------
    # LAYA AI INTEGRATION (Replaces LegacyLLM)
    # -------------------------------------------------------------------------

    def _call_laya_advisor(self, snapshot: Dict) -> Optional[Dict]:
        """Calls Laya AI for decision support instead of LegacyLLM."""
        try:
            from jarvis_laya_advisor import advise
            # Convert neural cortex snapshot to Laya snapshot format
            laya_snap = {
                'symbol': 'BTCUSDT',
                'price': snapshot.get('price', 0),
                'direction': snapshot.get('fusion_direction', 'NO_TRADE'),
                'confidence': snapshot.get('math_conf', 0),
                'parts_summary': snapshot.get('parts', {})
            }
            res = advise(laya_snap, symbol='BTCUSDT', deterministic_decision=snapshot.get('fusion_direction', 'NO_TRADE'), enabled=True)
            if res.status == 'ok' and res.suggestion:
                return {
                    'signal': res.suggestion.upper(),
                    'confidence': int((res.confidence or 0.5) * 100),
                    'rationale': res.rationale or 'Laya Neural Cortex Advisory',
                    'risk': 'MEDIUM'
                }
            return None
        except Exception as e:
            self.logger.debug(f"[LAYA] Cortex error: {e}")
            return None

    def _parse_decision(self, raw: Optional[str], part_results: Dict) -> Dict:
        if not raw:
            return self._fallback_from_math(part_results, ai_online=False)

        # ── Split narrative and JSON on ###JSON### marker ─────────────────
        narrative = ''
        json_part  = raw
        if '###JSON###' in raw:
            parts     = raw.split('###JSON###', 1)
            narrative = parts[0].strip()
            # Remove any <think>...</think> tags from narrative (deepseek-r1)
            narrative = re.sub(r'<think>.*?</think>', '', narrative, flags=re.DOTALL).strip()
            json_part = parts[1].strip()
        else:
            # Try to extract narrative as everything before first '{'
            brace_pos = raw.find('{')
            if brace_pos > 20:
                narrative = raw[:brace_pos].strip()
                narrative = re.sub(r'<think>.*?</think>', '', narrative, flags=re.DOTALL).strip()
                json_part = raw[brace_pos:]

        # ── Parse JSON part ──────────────────────────────────────────────
        result = None
        try:
            data   = json.loads(json_part)
            result = self._normalise(data, narrative=narrative)
        except json.JSONDecodeError:
            pass

        if result is None:
            for pattern in [r'```json\s*(\{.*?\})\s*```', r'(\{"signal".*?\})',
                            r'(\{.*?"signal".*?\})', r'(\{.*?\})']:
                m = re.search(pattern, json_part, re.DOTALL | re.IGNORECASE)
                if m:
                    try:
                        data   = json.loads(m.group(1))
                        result = self._normalise(data, narrative=narrative)
                        break
                    except Exception:
                        continue

        # ── Plain-text / markdown fallback ──────────────────────────────
        if result is None:
            try:
                signal_match = re.search(
                    r'\*\*Decision:\s*([A-Za-z_]+)\*\*|Decision:\s*([A-Za-z_]+)|Signal:\s*([A-Za-z_]+)',
                    raw, re.IGNORECASE)
                if signal_match:
                    signal_raw = next(g for g in signal_match.groups() if g).upper()
                    conf_match = re.search(r'Confidence:\s*(\d+)', raw, re.IGNORECASE)
                    conf       = int(conf_match.group(1)) if conf_match else 50
                    data       = {'signal': signal_raw, 'confidence': conf,
                                  'rationale': narrative or 'AI Decision (Text Parsed)', 'risk': 'MEDIUM'}
                    result     = self._normalise(data, narrative=narrative)
                    logger.info(f"[CORTEX] Parsed text format: {signal_raw} ({conf}%)")
            except Exception as e:
                logger.debug(f"[CORTEX] Text parse failed: {e}")

        if result is None:
            logger.warning(f"[CORTEX] Could not parse response: {raw[:120]}")
            result = self._fallback_from_math(part_results, ai_online=True)
            if narrative:
                result['ai_narrative'] = narrative

        return result

    def _normalise(self, data: Dict, narrative: str = '') -> Dict:
        raw_sig = str(data.get('signal', data.get('bias', 'NO_TRADE'))).upper()
        if raw_sig in ('CALL', 'BUY', 'BULLISH', 'LONG'):
            signal = 'CALL'
        elif raw_sig in ('PUT', 'SELL', 'BEARISH', 'SHORT'):
            signal = 'PUT'
        else:
            signal = 'NO_TRADE'
        try:
            conf = max(0, min(100, int(data.get('confidence', 0))))
        except Exception:
            conf = 0
        rationale = str(data.get('rationale', data.get('reasoning', 'AI response'))).strip()[:120]
        risk = str(data.get('risk', 'MEDIUM')).upper()
        if risk not in ('LOW', 'MEDIUM', 'HIGH'):
            risk = 'MEDIUM'
        # Use narrative if available, else fall back to rationale
        display_narrative = (narrative.strip() if narrative and len(narrative) > 10
                             else rationale)
        return {'signal': signal, 'bias': signal.replace('_', '-'), 'confidence': conf,
                'rationale': rationale, 'reasoning': rationale, 'risk': risk,
                'ai_online': True, 'ai_narrative': display_narrative}

    def _fallback_from_math(self, part_results: Dict, ai_online: bool = False) -> Dict:
        try:
            fusion    = part_results.get('part11_fusion', {})
            conf_part = part_results.get('part12_confidence', {})
            math_sig  = fusion.get('signal', 0)
            math_conf = conf_part.get('confidence', 10)
            signal    = 'CALL' if math_sig > 0 else ('PUT' if math_sig < 0 else 'NO_TRADE')
            source    = ("AI offline — using math vote fallback."
                         if not ai_online else
                         "AI response could not be parsed — using math vote fallback.")
            return {'signal': signal, 'bias': signal.replace('_', '-'), 'confidence': math_conf,
                    'rationale': source, 'reasoning': source, 'risk': 'MEDIUM',
                    'ai_online': False, 'ai_narrative': source}
        except Exception:
            return {'signal': 'NO_TRADE', 'bias': 'NO-TRADE', 'confidence': 0,
                    'rationale': 'Critical AI error', 'reasoning': 'Critical AI error',
                    'risk': 'HIGH', 'ai_online': False, 'ai_narrative': 'Critical system error — cannot analyze.'}

    # -------------------------------------------------------------------------
    # BACKWARD COMPAT SHIMS (for old ai_chain_brain references)
    # -------------------------------------------------------------------------

    def analyze_holistic_context(self, prompt: str, system_voice: str = "Assistant"):
        return "", "Model-generated analysis retired; Laya is audit-only"

    @property
    def part_results(self):
        return {}

    def start_sequential_loop(self, jarvis_engine):
        logger.info("[CORTEX] Cortex is synchronous -- no background loop required.")

    def stop(self):
        logger.info("[CORTEX] Stopped.")
