"""Safe current options expiry parsing; maturity dates are valid observations, not future data."""
from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any


def parse_option_contract(name: Any, today: date | None = None):
    today = today or datetime.now(timezone.utc).date()
    if isinstance(today, datetime): today = today.date()
    if not isinstance(name, str): return None
    p = name.upper().split('-')
    if len(p) != 4 or p[3] not in {'C','P'}: return None
    try:
        if not p[0] or not p[2].isdigit() or int(p[2]) <= 0: return None
        expiry = datetime.strptime(p[1], '%d%b%y').date()
    except (ValueError, TypeError): return None
    if expiry < today: return None
    return {'underlying': p[0], 'expiry': expiry.isoformat(), 'strike': int(p[2]), 'type': p[3]}


def group_current_expiries(instruments: list[dict], underlying: str, now: datetime | None = None):
    now = now or datetime.now(timezone.utc); today = now.date(); target = str(underlying).upper()
    groups = defaultdict(lambda: {'calls': 0, 'puts': 0, 'open_interest': 0.0, 'volume': 0.0, 'contracts': 0, 'missing_fields': 0})
    for inst in instruments or []:
        if not isinstance(inst, dict): continue
        parsed = parse_option_contract(inst.get('instrument_name'), today)
        if not parsed or parsed['underlying'] != target: continue
        g = groups[parsed['expiry']]; g['contracts'] += 1
        g['calls' if parsed['type']=='C' else 'puts'] += 1
        for field in ('open_interest','volume'):
            value = inst.get(field)
            if value is None: g['missing_fields'] += 1
            elif isinstance(value,(int,float)) and value >= 0: g[field] += float(value)
    result = []
    for expiry in sorted(groups):
        g = dict(groups[expiry]); g['expiry'] = expiry; result.append(g)
    return {'underlying': target, 'as_of': now.astimezone(timezone.utc).isoformat(), 'expiry_count': len(result), 'expiries': result}
