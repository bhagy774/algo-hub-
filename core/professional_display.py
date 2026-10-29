#!/usr/bin/env python3
"""
JARVIS God-Mode Terminal Display
Real JARVIS experience - AI speaks, market explained, clear signals
"""

import os
import re
import sys
import platform
import textwrap
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ══ Windows: Force UTF-8 + ANSI ══════════════════════════════════════════════
if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# ══ ANSI Color Palette ═══════════════════════════════════════════════════════
R   = '\033[91m'    # Red
G   = '\033[92m'    # Green
Y   = '\033[93m'    # Yellow
B   = '\033[94m'    # Blue
M   = '\033[95m'    # Magenta / Purple
C   = '\033[96m'    # Cyan
W   = '\033[97m'    # White
DG  = '\033[90m'    # Dark Gray
BD  = '\033[1m'     # Bold
DM  = '\033[2m'     # Dim
UL  = '\033[4m'     # Underline
RST = '\033[0m'     # Reset All

W_LINE = 92         # Terminal box width

def _clr(text, *codes):
    return ''.join(codes) + str(text) + RST

def _strip(text):
    return re.sub(r'\033\[[0-9;]*m', '', str(text))

def _pad(text, width):
    """Pad text (accounting for invisible ANSI codes) to visual width."""
    clean_len = len(_strip(text))
    return text + ' ' * max(0, width - clean_len)

def _bar(pct, width=36):
    """Progress bar: green filled, dark gray empty."""
    pct = max(0, min(100, pct))
    filled = int(pct / 100 * width)
    empty  = width - filled
    if pct >= 70:
        col = G
    elif pct >= 50:
        col = Y
    else:
        col = R
    return _clr('█' * filled, col) + _clr('░' * empty, DG)

def _center_visual(text, width):
    """Center text using visual (non-ANSI) length."""
    clean = len(_strip(text))
    pad   = max(0, width - clean)
    l, r  = pad // 2, pad - pad // 2
    return ' ' * l + text + ' ' * r


class ProfessionalSignalDisplay:
    """JARVIS God-Mode Terminal — every cycle feels alive."""

    def __init__(self):
        self.W = W_LINE
        self._cycle = 0

    # ─── box helpers ────────────────────────────────────────────────────────
    def _top(self):
        return _clr('╔' + '═' * (self.W - 2) + '╗', C)

    def _bot(self):
        return _clr('╚' + '═' * (self.W - 2) + '╝', C)

    def _sep(self, char='═'):
        return _clr('╠' + char * (self.W - 2) + '╣', C)

    def _thin(self):
        return _clr('╟' + '─' * (self.W - 2) + '╢', C)

    def _row(self, content=''):
        """Wrap content in box sides, padding to correct visual width."""
        inner = _pad(content, self.W - 4)
        return _clr('║', C) + ' ' + inner + ' ' + _clr('║', C)

    def _blank(self):
        return self._row('')

    # ─── main display ───────────────────────────────────────────────────────
    def display_full_signal(self, signal_data, current_price=None, part_results=None,
                            symbol=None, model_name=None):
        """
        Render the full JARVIS god-mode terminal display.

        signal_data  – dict with direction, confidence, ai_reason, entry_price …
        current_price – live price (float)
        part_results  – dict  {part_name: {signal: -1/0/1, thought: str, …}}
        symbol        – active trading symbol (e.g. 'NEARUSD', 'BTCUSDT')
        model_name    – active LegacyLLM model name
        """
        self._cycle += 1

        # ── Extract core values ──────────────────────────────────────────
        direction  = str(signal_data.get('direction', '') or
                         signal_data.get('trade_signal', {}).get('direction', 'NO_TRADE')).upper()
        if direction in ('BUY',):    direction = 'CALL'
        if direction in ('SELL',):   direction = 'PUT'
        if not direction:            direction = 'NO_TRADE'

        conf_raw = signal_data.get('confidence', 0)
        if not conf_raw:
            try:
                cs = signal_data.get('trade_signal', {}).get('confidence_score', '0/100')
                conf_raw = int(str(cs).split('/')[0])
            except Exception:
                conf_raw = 0
        confidence = int(conf_raw)

        price = float(current_price or signal_data.get('entry_price', 0) or
                      signal_data.get('trade_signal', {}).get('entry_price', 0) or 0)

        # AI narrative – prefer the fuller narrative key
        ai_reason = (signal_data.get('ai_narrative') or
                     signal_data.get('ai_reason') or
                     signal_data.get('rationale') or
                     signal_data.get('reasoning') or
                     signal_data.get('trade_signal', {}).get('ai_synthesis', '') or
                     'Analyzing market conditions…')

        # TP / SL
        ts  = signal_data.get('trade_signal', {})
        tp1 = ts.get('take_profit_1') or signal_data.get('tp1')
        sl  = ts.get('stop_loss')     or signal_data.get('sl')
        exp = ts.get('recommended_expiry', '3M')

        # Market context
        mctx      = signal_data.get('market_context', {})
        trend     = mctx.get('trend', '─')
        vol_stat  = mctx.get('volatility_status', mctx.get('volatility', '─'))
        session   = mctx.get('session', '─')

        # Part results – accept from multiple keys
        if part_results is None:
            part_results = (signal_data.get('parts_data') or
                            signal_data.get('detailed_scores') or {})

        # ── Signal colors / labels ───────────────────────────────────────
        if direction == 'CALL':
            sig_col, sig_bg   = G, BD + G
            sig_arrow         = '▲'
            signal_label      = _clr(' ██  CALL  ▲  BUY  ██ ', BD + G)
            border_col        = G
        elif direction == 'PUT':
            sig_col, sig_bg   = R, BD + R
            sig_arrow         = '▼'
            signal_label      = _clr(' ██  PUT   ▼  SELL ██ ', BD + R)
            border_col        = R
        else:
            sig_col, sig_bg   = Y, BD + Y
            sig_arrow         = '━'
            signal_label      = _clr(' ▬▬  WAIT  ━  HOLD ▬▬ ', BD + Y)
            border_col        = Y

        # Trend emoji
        tu = str(trend).upper()
        trend_icon = ('📈' if 'UP' in tu else '📉' if 'DOWN' in tu else '📊')

        # ── Build output ─────────────────────────────────────────────────
        self._clear()
        out = []
        ap  = out.append   # shorthand

        # ═══ HEADER ═══════════════════════════════════════════════════
        now_str  = datetime.now().strftime('%H:%M:%S')
        date_str = datetime.now().strftime('%d %b %Y')

        ap(self._top())
        # Use active symbol from param; fallback to env var or BTC/USDT
        _sym = symbol or __import__('os').environ.get('JARVIS_DEFAULT_SYMBOL', 'BTC/USDT')
        _model = model_name or __import__('os').environ.get('LEGACY_LLM_MODEL', 'deepseek-r1:14b')
        header = (
            _clr('  🤖 JARVIS', BD + C) +
            _clr(' NEURAL CORTEX', C) +
            _clr('  │  ', DG) +
            _clr(_sym, BD + W) +
            _clr('  │  ', DG) +
            _clr(date_str, DG) +
            '  ' +
            _clr(now_str, BD + Y) +
            _clr('  │  ', DG) +
            _clr(_model, M) +
            _clr(f'  │  Cycle #{self._cycle}', DG)
        )
        ap(self._row(header))

        if price > 0:
            price_row = (
                _clr('  💰 ', DG) + _clr(f'${price:,.2f}', BD + Y) +
                _clr('   │   ', DG) +
                trend_icon + '  ' + _clr(trend, G if 'UP' in tu else R if 'DOWN' in tu else Y) +
                _clr('   │   ', DG) +
                _clr('VOL: ', DG) + _clr(str(vol_stat), Y) +
                _clr('   │   ', DG) +
                _clr('SESSION: ', DG) + _clr(str(session).upper(), C)
            )
            ap(self._row(price_row))

        ap(self._sep())

        # ═══ 12 ENGINE BRAIN ANALYSIS ══════════════════════════════════
        ap(self._row(_clr('  ⚡ 12-ENGINE BRAIN ANALYSIS', BD + C)))
        ap(self._thin())

        PARTS = [
            ('part1_breakout',      '1', 'BREAKOUT DETECTOR'),
            ('part2_zone',          '2', 'ZONE INTELLIGENCE'),
            ('part3_psychology',    '3', 'CANDLE PSYCHOLOGY'),
            ('part4_volume',        '4', 'VOLUME PROFILE   '),
            ('part5_ml',            '5', 'ML NEURAL NET    '),
            ('part6_trend',         '6', 'TREND ENGINE     '),
            ('part7_volatility',    '7', 'VOLATILITY SHIELD'),
            ('part8_structure',     '8', 'MARKET STRUCTURE '),
            ('part9_orderflow',     '9', 'ORDER FLOW DELTA '),
            ('part10_candlestats',  'A', 'CANDLE STATS     '),
            ('part11_fusion',       'B', 'FUSION VOTE MATH '),
            ('part12_confidence',   'C', 'CONFIDENCE ENGINE'),
        ]

        bull_n = bear_n = neut_n = 0

        for key, num, label in PARTS:
            res = part_results.get(key, {}) if isinstance(part_results, dict) else {}

            if isinstance(res, dict):
                sig    = res.get('signal', 0)
                thought = str(res.get('thought', res.get('reasoning', 'Standby…')))
            elif isinstance(res, (int, float)):
                sig, thought = float(res), 'Signal computed'
            else:
                sig, thought = 0, 'Offline'

            try:
                sig = float(sig)
            except Exception:
                sig = 0.0

            if sig > 0:
                bull_n += 1
                icon = _clr('▲ BULL', BD + G)
                dot  = _clr('●', G)
            elif sig < 0:
                bear_n += 1
                icon = _clr('▼ BEAR', BD + R)
                dot  = _clr('●', R)
            else:
                neut_n += 1
                icon = _clr('━ NEUT', DG)
                dot  = _clr('○', DG)

            # Truncate thought to fit
            t_clean = _strip(thought)
            max_t   = 43
            if len(t_clean) > max_t:
                t_clean = t_clean[:max_t] + '…'
            t_dim = _clr(t_clean, DM)

            row = (
                f'  {dot} ' +
                _clr(f'[{num}]', DG) + '  ' +
                _clr(f'{label}', W) + '   ' +
                icon + '   ' +
                _clr('│', DG) + '  ' +
                t_dim
            )
            ap(self._row(row))

        ap(self._thin())

        # Vote summary row
        total_v  = bull_n + bear_n + neut_n
        bull_pct = int(bull_n / max(total_v, 1) * 100)
        bear_pct = int(bear_n / max(total_v, 1) * 100)

        vote_row = (
            _clr('  VOTES  ▶  ', BD + C) +
            _clr(f'▲ {bull_n} BULL', BD + G) + '  ' +
            _clr(f'▼ {bear_n} BEAR', BD + R) + '  ' +
            _clr(f'━ {neut_n} NEUT', DG) +
            _clr('   │   ', DG) +
            _clr('MAJORITY: ', DG) +
            _clr(f'{bull_pct}% BULLISH', BD + G if bull_pct >= 60 else BD + R if bear_pct >= 60 else BD + Y)
        )
        ap(self._row(vote_row))
        ap(self._sep())

        # ═══ JARVIS SPEAKS ═════════════════════════════════════════════
        ap(self._row(_clr('  🧠 JARVIS SAYS:', BD + M)))
        ap(self._thin())

        # Wrap AI narrative
        inner_w = self.W - 6
        lines_ai = textwrap.wrap(str(ai_reason), width=inner_w)
        if not lines_ai:
            lines_ai = ['Waiting for AI analysis…']
        for ln in lines_ai:
            ap(self._row('  ' + _clr(ln, W)))

        ap(self._sep())

        # ═══ CONFIDENCE BAR ═══════════════════════════════════════════
        ap(self._blank())
        conf_col = G if confidence >= 70 else Y if confidence >= 50 else R
        bar_row  = (
            _clr('  CONFIDENCE  ', DG) +
            _bar(confidence, width=38) +
            '  ' +
            _clr(f'{confidence}%', BD + conf_col)
        )
        ap(self._row(bar_row))

        # ═══ MAIN SIGNAL BANNER ═══════════════════════════════════════
        ap(self._blank())
        # Center the signal label visually
        banner  = _center_visual(signal_label, self.W - 4)
        ap(self._row(banner))
        ap(self._blank())
        ap(self._thin())

        # ═══ TRADE LEVELS ═════════════════════════════════════════════
        if direction in ('CALL', 'PUT') and price > 0:
            is_call = direction == 'CALL'
            if tp1 is None:
                tp1 = price * (1.006 if is_call else 0.994)
            if sl is None:
                sl  = price * (0.997 if is_call else 1.003)

            exp_map = {'1M': 1, '2M': 2, '3M': 3, '5M': 5, 'SCALP': 3, 'DAY_TRADE': 15}
            exp_min = exp_map.get(str(exp).upper(), 3)
            exp_time = (datetime.now() + timedelta(minutes=exp_min)).strftime('%H:%M:%S')

            lvl = (
                _clr('  ENTRY: ', DG) + _clr(f'${price:,.2f}', BD + W) +
                _clr('   TP: ', DG)    + _clr(f'${tp1:,.2f}', BD + G) +
                _clr('   SL: ', DG)    + _clr(f'${sl:,.2f}', BD + R) +
                _clr('   EXPIRY: ', DG) + _clr(f'{exp} → {exp_time}', BD + Y)
            )
            ap(self._row(lvl))
        elif direction == 'NO_TRADE':
            reason = signal_data.get('no_trade_reason', 'Engines not aligned — waiting for setup.')
            ap(self._row(_clr('  ⏸  ' + str(reason)[:self.W - 8], DG)))

        ap(self._bot())
        print('\n'.join(out))
        print()

    def display_compact(self, final_decision):
        """One-line compact signal for quiet cycles."""
        ts  = datetime.now().strftime('%H:%M:%S')
        d   = str(final_decision.get('direction', 'HOLD')).upper()
        c   = final_decision.get('confidence', 0)
        p   = final_decision.get('entry', 0)
        em  = '🟢' if d in ('BUY', 'CALL') else '🔴' if d in ('SELL', 'PUT') else '⚪'
        col = G if d in ('BUY', 'CALL') else R if d in ('SELL', 'PUT') else Y
        print(f"{_clr(f'[{ts}]', DG)} {em} {_clr(d, BD + col)} {_clr('│', DG)} "
              f"{_clr('Conf:', DG)} {_clr(f'{c}%', col)} {_clr('│', DG)} "
              f"{_clr(f'${p:,.0f}', W)}")

    def _clear(self):
        os.system('cls' if platform.system() == 'Windows' else 'clear')


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    d = ProfessionalSignalDisplay()

    fake_parts = {
        'part1_breakout':    {'signal':  1, 'thought': 'Broke 97,200 resistance level — 13 brains confirm'},
        'part2_zone':        {'signal':  1, 'thought': 'Price sitting inside demand zone $97,100-$97,300'},
        'part3_psychology':  {'signal':  1, 'thought': 'Bullish engulfing on 15m, strong body, low wick'},
        'part4_volume':      {'signal':  0, 'thought': 'Volume 1.1x average — slightly elevated, not extreme'},
        'part5_ml':          {'signal':  1, 'thought': 'LSTM predicts +0.4% move over next 3 bars'},
        'part6_trend':       {'signal':  1, 'thought': 'EMA 8 > 21 > 50 perfectly stacked bullish'},
        'part7_volatility':  {'signal':  0, 'thought': 'ATR NORMAL (0.4%) — safe conditions to trade'},
        'part8_structure':   {'signal':  1, 'thought': 'Higher Highs and Higher Lows confirmed on 1h chart'},
        'part9_orderflow':   {'signal':  1, 'thought': 'Buy delta +$2.3M vs Sell -$0.8M — buyers dominating'},
        'part10_candlestats':{'signal': -1, 'thought': 'Last 3 candles mixed, slight bearish wick on top'},
        'part11_fusion':     {'signal':  1, 'thought': '8/10 engines agree BULL, math vote +0.68'},
        'part12_confidence': {'signal':  0, 'thought': 'Confidence score 78% — within trade zone'},
    }

    d.display_full_signal(
        signal_data={
            'direction':    'CALL',
            'confidence':   78,
            'ai_narrative': (
                "I'm seeing strong bullish momentum with 8 of 12 engines aligned. "
                "Volume is ticking up above average, EMA structure is perfectly stacked, "
                "and institutional orderflow is skewed heavily to buy-side. "
                "The demand zone at $97,100 is holding firm. My confidence is 78% — I'm calling CALL."
            ),
            'entry_price': 97432.50,
            'market_context': {
                'trend': 'UPTREND', 'volatility_status': 'NORMAL', 'session': 'london'
            },
            'trade_signal': {
                'take_profit_1': 98015.00,
                'stop_loss':     97140.00,
                'recommended_expiry': '3M',
            },
        },
        current_price=97432.50,
        part_results=fake_parts,
    )
