#!/usr/bin/env python3
"""
Deribit Options Chain Data Client
Fetches Open Interest, Put-Call Ratio, and OI Walls from Deribit API
"""

import requests
import time
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from options_expiry import group_current_expiries

logger = logging.getLogger(__name__)

class DeribitOptionsClient:
    """Client to fetch and analyze Options Chain data from Deribit"""
    
    def __init__(self, currency='BTC', cache_duration=30, client_id=None, client_secret=None):
        """
        Initialize Deribit client
        
        Args:
            currency: 'BTC' or 'ETH'
            cache_duration: Cache lifetime in seconds (default 30s)
            client_id: Deribit API Client ID (optional, for authentication)
            client_secret: Deribit API Client Secret (optional, for authentication)
        """
        self.base_url = "https://www.deribit.com/api/v2"
        self.test_url = "https://test.deribit.com/api/v2"
        self.currency = currency
        self.cache_duration = cache_duration
        
        # Authentication
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = 0
        
        # Authentication is intentionally lazy.  Constructing an analysis
        # client must not emit a credential-bearing request.
        self._cache = None
        self._cache_time = 0
        
        # Smart Money Tracking
        self._prev_oi = None  # Store previous OI for change detection
        self._oi_history = []  # Track OI changes over time
    
    def _authenticate(self):
        """Authenticate with Deribit and obtain access token"""
        try:
            url = f"{self.base_url}/public/auth"
            params = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data and 'access_token' in data['result']:
                self.access_token = data['result']['access_token']
                self.token_expiry = time.time() + data['result'].get('expires_in', 3600)
                logger.info("[Deribit] Authentication successful")
            else:
                logger.error(f"[Deribit] Auth failed: {data}")
                
        except requests.RequestException as e:
            logger.error(f"[Deribit] Auth error: {e}")
        except Exception as e:
            logger.error(f"[Deribit] Unexpected auth error: {e}")
    
    def _get_headers(self):
        """Get headers with auth token if available"""
        headers = {}
        if self.access_token and time.time() < self.token_expiry:
            headers['Authorization'] = f'Bearer {self.access_token}'
        elif self.client_id and self.client_secret:
            # Token expired, re-authenticate
            self._authenticate()
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
        return headers
        
    def summarize_current_expiries(self, instruments=None) -> Dict:
        """Return all listed, unexpired matching-underlying expiries as of now."""
        if instruments is None:
            try:
                response = requests.get(f"{self.base_url}/public/get_book_summary_by_currency", params={"currency": self.currency, "kind": "option"}, timeout=10)
                response.raise_for_status(); instruments = response.json().get("result", [])
            except Exception as exc:
                return {"underlying": self.currency.upper(), "as_of": datetime.now(timezone.utc).isoformat(), "expiry_count": 0, "expiries": [], "error": str(exc)}
        return group_current_expiries(instruments, self.currency)

    def get_option_chain(self, use_cache=True) -> Optional[Dict]:
        """
        Fetch current Option Chain data
        
        Returns:
            {
                'strikes': [95000, 96000, ...],
                'call_oi': [1200, 3400, ...],
                'put_oi': [450, 890, ...],
                'pcr': 0.85,
                'max_pain': 96500,
                'resistance_wall': 98000,
                'support_wall': 95000,
                'timestamp': 1234567890
            }
        """
        # Check cache
        if use_cache and self._cache and (time.time() - self._cache_time) < self.cache_duration:
            return self._cache
            
        try:
            # Fetch all option instruments for the currency
            url = f"{self.base_url}/public/get_book_summary_by_currency"
            params = {
                'currency': self.currency,
                'kind': 'option'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' not in data:
                logger.error(f"Invalid Deribit response: {data}")
                return None
                
            instruments = data['result']
            if not isinstance(instruments, list):
                logger.error("Invalid Deribit result type")
                return None

            # Parse and aggregate by strike
            strike_data = {}
            for inst in instruments:
                try:
                    if not isinstance(inst, dict):
                        continue
                    name = inst['instrument_name']
                    parts = name.split('-')
                    
                    # Format: BTC-28DEC24-96000-C (or -P)
                    if len(parts) != 4:
                        continue
                        
                    strike = int(parts[2])
                    option_type = parts[3]  # 'C' or 'P'
                    oi = float(inst.get('open_interest', 0) or 0)
                    if not math.isfinite(oi) or oi < 0:
                        continue

                    if strike not in strike_data:
                        strike_data[strike] = {'call_oi': 0, 'put_oi': 0}
                    
                    if option_type == 'C':
                        strike_data[strike]['call_oi'] += oi
                    elif option_type == 'P':
                        strike_data[strike]['put_oi'] += oi
                        
                except (ValueError, IndexError, KeyError) as e:
                    logger.debug(f"Skipping instrument {inst.get('instrument_name')}: {e}")
                    continue
            
            if not strike_data:
                logger.warning("No option data parsed from Deribit")
                return None
            
            # Sort by strike
            sorted_strikes = sorted(strike_data.keys())
            call_oi_list = [strike_data[s]['call_oi'] for s in sorted_strikes]
            put_oi_list = [strike_data[s]['put_oi'] for s in sorted_strikes]
            
            # Calculate metrics
            total_call_oi = sum(call_oi_list)
            total_put_oi = sum(put_oi_list)
            pcr = total_put_oi / (total_call_oi + 1e-9)
            
            max_pain = self._calculate_max_pain(sorted_strikes, call_oi_list, put_oi_list)
            resistance_wall, support_wall = self._find_oi_walls(sorted_strikes, call_oi_list, put_oi_list)
            
            # Calculate OI changes (Smart Money Detection)
            oi_changes = {}
            if self._prev_oi:
                for strike in sorted_strikes:
                    if strike in self._prev_oi:
                        call_change = strike_data[strike]['call_oi'] - self._prev_oi[strike]['call_oi']
                        put_change = strike_data[strike]['put_oi'] - self._prev_oi[strike]['put_oi']
                        oi_changes[strike] = {
                            'call_change': call_change,
                            'put_change': put_change,
                            'call_pct': (call_change / (self._prev_oi[strike]['call_oi'] + 1e-9)) * 100,
                            'put_pct': (put_change / (self._prev_oi[strike]['put_oi'] + 1e-9)) * 100
                        }
            
            result = {
                'strikes': sorted_strikes,
                'call_oi': call_oi_list,
                'put_oi': put_oi_list,
                'pcr': round(pcr, 3),
                'max_pain': max_pain,
                'resistance_wall': resistance_wall,
                'support_wall': support_wall,
                'oi_changes': oi_changes,  # NEW: Track changes
                'timestamp': int(time.time())
            }
            
            # Store current OI for next comparison
            self._prev_oi = strike_data.copy()
            
            # Update cache
            self._cache = result
            self._cache_time = time.time()
            
            logger.info(f"[OI] PCR: {pcr:.2f} | Support: {support_wall} | Resistance: {resistance_wall}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Deribit API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching option chain: {e}")
            return None
    
    def _calculate_max_pain(self, strikes: List[int], call_oi: List[float], put_oi: List[float]) -> int:
        """Calculate Max Pain (price with maximum total loss for option holders)"""
        if not strikes:
            return 0
            
        max_pain_strike = strikes[0]
        min_total_value = float('inf')
        
        for test_strike in strikes:
            total_value = 0
            
            for i, strike in enumerate(strikes):
                # ITM Call value
                if strike < test_strike:
                    total_value += call_oi[i] * (test_strike - strike)
                    
                # ITM Put value
                if strike > test_strike:
                    total_value += put_oi[i] * (strike - test_strike)
            
            if total_value < min_total_value:
                min_total_value = total_value
                max_pain_strike = test_strike
        
        return max_pain_strike
    
    def _find_oi_walls(self, strikes: List[int], call_oi: List[float], put_oi: List[float]) -> Tuple[int, int]:
        """Find significant OI concentration (walls)"""
        if not strikes:
            return 0, 0
            
        # Find peak Call OI (Resistance)
        max_call_idx = call_oi.index(max(call_oi))
        resistance_wall = strikes[max_call_idx]
        
        # Find peak Put OI (Support)
        max_put_idx = put_oi.index(max(put_oi))
        support_wall = strikes[max_put_idx]
        
        return resistance_wall, support_wall
    
    def validate_signal(self, current_price: float, signal_direction: str) -> int:
        """
        Validate a CALL/PUT signal against Option Chain
        
        Args:
            current_price: Current spot price
            signal_direction: 'CALL' or 'PUT'
            
        Returns:
            1: Strong confirmation (OI supports signal)
            0: Neutral (no clear OI edge)
            -1: Conflict (OI opposes signal)
        """
        oi_data = self.get_option_chain()
        if not oi_data:
            logger.warning("No OI data available for validation")
            return 0
        
        pcr = oi_data['pcr']
        support_wall = oi_data['support_wall']
        resistance_wall = oi_data['resistance_wall']
        
        # Rule 1: PCR-based sentiment
        if signal_direction == 'CALL':
            if pcr > 1.2:  # High Put OI -> Oversold -> Favor CALL
                logger.info(f"[OI CONFIRM] High PCR ({pcr}) supports CALL")
                return 1
            elif pcr < 0.8:  # High Call OI -> Overbought -> Conflict
                logger.warning(f"[OI CONFLICT] Low PCR ({pcr}) opposes CALL")
                return -1
        
        elif signal_direction == 'PUT':
            if pcr < 0.8:  # High Call OI -> Overbought -> Favor PUT
                logger.info(f"[OI CONFIRM] Low PCR ({pcr}) supports PUT")
                return 1
            elif pcr > 1.2:  # High Put OI -> Oversold -> Conflict
                logger.warning(f"[OI CONFLICT] High PCR ({pcr}) opposes PUT")
                return -1
        
        # Rule 2: Proximity to OI Walls
        dist_to_resistance = abs(current_price - resistance_wall) / max(current_price, 1) if current_price > 0 else 0
        dist_to_support = abs(current_price - support_wall) / max(current_price, 1) if current_price > 0 else 0
        
        if signal_direction == 'CALL' and dist_to_support < 0.02:  # < 2% from support
            logger.info(f"[OI CONFIRM] Price near support wall ({support_wall})")
            return 1
        
        if signal_direction == 'PUT' and dist_to_resistance < 0.02:  # < 2% from resistance
            logger.info(f"[OI CONFIRM] Price near resistance wall ({resistance_wall})")
            return 1
        
        return 0  # Neutral

    def detect_smart_money(self, current_price: float) -> Dict:
        """
        Detect Smart Money (Institutional) Flow
        
        Returns:
            {
                'detected': True/False,
                'direction': 'CALL'/'PUT'/None,
                'confidence': 0-100,
                'strike': strike where flow detected,
                'details': explanation string
            }
        """
        oi_data = self.get_option_chain()
        if not oi_data or not oi_data.get('oi_changes'):
            return {'detected': False, 'direction': None, 'confidence': 0, 'details': 'No OI change data'}
        
        oi_changes = oi_data['oi_changes']
        max_call_change = 0
        max_put_change = 0
        call_strike = None
        put_strike = None
        
        # Find largest OI changes
        for strike, changes in oi_changes.items():
            # Significant change = > 15% increase
            if changes['call_pct'] > 15 and changes['call_change'] > max_call_change:
                max_call_change = changes['call_change']
                call_strike = strike
                
            if changes['put_pct'] > 15 and changes['put_change'] > max_put_change:
                max_put_change = changes['put_change']
                put_strike = strike
        
        # Determine Smart Money direction
        if max_call_change > max_put_change and call_strike:
            # Big players buying CALLS
            confidence = min(95, 70 + int(max_call_change / 100))
            details = f"🐋 SMART MONEY: +{int(max_call_change)} BTC Calls at ${call_strike} ({oi_changes[call_strike]['call_pct']:.1f}%)"
            logger.info(f"[SMART MONEY] Call accumulation detected at ${call_strike}")
            return {
                'detected': True,
                'direction': 'CALL',
                'confidence': confidence,
                'strike': call_strike,
                'details': details
            }
            
        elif max_put_change > max_call_change and put_strike:
            # Big players buying PUTS
            confidence = min(95, 70 + int(max_put_change / 100))
            details = f"🐋 SMART MONEY: +{int(max_put_change)} BTC Puts at ${put_strike} ({oi_changes[put_strike]['put_pct']:.1f}%)"
            logger.info(f"[SMART MONEY] Put accumulation detected at ${put_strike}")
            return {
                'detected': True,
                'direction': 'PUT',
                'confidence': confidence,
                'strike': put_strike,
                'details': details
            }
        
        return {'detected': False, 'direction': None, 'confidence': 0, 'details': 'No significant flow'}
    
    def get_greeks_and_iv(self) -> Optional[Dict]:
        """
        Fetch Greeks (Delta, Gamma, Theta, Vega) and Implied Volatility
        for ATM (At-The-Money) options
        
        Returns:
            {
                'call_delta': 0.55,
                'put_delta': -0.45,
                'call_gamma': 0.02,
                'call_theta': -15.5,
                'call_vega': 125.0,
                'call_iv': 65.5,  # Implied Volatility %
                'put_iv': 67.2,
                'atm_strike': 97000
            }
        """
        try:
            # Get current BTC price to find ATM strike
            url = f"{self.base_url}/public/ticker"
            params = {'instrument_name': f'{self.currency}-PERPETUAL'}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' not in data:
                return None
                
            current_price = float(data['result']['last_price'])
            if not math.isfinite(current_price) or current_price <= 0:
                return None

            # Find nearest ATM strike (round to nearest 1000)
            atm_strike = round(current_price / 1000) * 1000
            
            # Get all option instruments
            url2 = f"{self.base_url}/public/get_instruments"
            params2 = {'currency': self.currency, 'kind': 'option'}
            response2 = requests.get(url2, params=params2, timeout=10)
            response2.raise_for_status()
            instruments_data = response2.json()
            
            if 'result' not in instruments_data:
                return None
            
            # Find ATM Call and Put
            call_instrument = None
            put_instrument = None
            
            for inst in instruments_data['result']:
                name = inst['instrument_name']
                if str(atm_strike) in name:
                    if name.endswith('-C'):
                        call_instrument = name
                    elif name.endswith('-P'):
                        put_instrument = name
                    
                    if call_instrument and put_instrument:
                        break
            
            if not call_instrument or not put_instrument:
                logger.warning(f"ATM options not found for strike ${atm_strike}")
                return None
            
            # Fetch Greeks for Call
            url3 = f"{self.base_url}/public/ticker"
            params3 = {'instrument_name': call_instrument}
            response3 = requests.get(url3, params=params3, timeout=10)
            call_data = response3.json()['result']
            
            # Fetch Greeks for Put
            params4 = {'instrument_name': put_instrument}
            response4 = requests.get(url3, params=params4, timeout=10)
            put_data = response4.json()['result']
            
            result = {
                'call_delta': call_data.get('greeks', {}).get('delta', 0),
                'put_delta': put_data.get('greeks', {}).get('delta', 0),
                'call_gamma': call_data.get('greeks', {}).get('gamma', 0),
                'call_theta': call_data.get('greeks', {}).get('theta', 0),
                'call_vega': call_data.get('greeks', {}).get('vega', 0),
                'call_iv': call_data.get('mark_iv', 0),
                'put_iv': put_data.get('mark_iv', 0),
                'atm_strike': atm_strike,
                'current_price': current_price
            }
            
            logger.info(f"[GREEKS] Delta: {result['call_delta']:.2f} | IV: {result['call_iv']:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Greeks fetch error: {e}")
            return None
    
    def get_volume_profile(self) -> Optional[Dict]:
        """
        Fetch Volume distribution across strikes
        
        Returns:
            {
                strike: {'volume': 12500, 'oi': 3400},
                ...
            }
        """
        try:
            url = f"{self.base_url}/public/get_book_summary_by_currency"
            params = {'currency': self.currency, 'kind': 'option'}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' not in data:
                return None
            
            volume_profile = {}
            for inst in data['result']:
                try:
                    name = inst['instrument_name']
                    parts = name.split('-')
                    if len(parts) != 4:
                        continue
                    
                    strike = int(parts[2])
                    volume = inst.get('volume', 0)
                    oi = inst.get('open_interest', 0)
                    
                    if strike not in volume_profile:
                        volume_profile[strike] = {'volume': 0, 'oi': 0}
                    
                    volume_profile[strike]['volume'] += volume
                    volume_profile[strike]['oi'] += oi
                    
                except (ValueError, IndexError):
                    continue
            
            # Find most active strike
            if volume_profile:
                max_vol_strike = max(volume_profile, key=lambda x: volume_profile[x]['volume'])
                logger.info(f"[VOLUME] Most active strike: ${max_vol_strike} ({volume_profile[max_vol_strike]['volume']:.0f} BTC)")
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return None
    
    def analyze_full_market(self, current_price: float) -> Dict:
        """
        MASTER METHOD: Analyze ALL available data
        
        Returns comprehensive market analysis combining:
        - OI and PCR
        - OI Changes (Smart Money)
        - Greeks and IV
        - Volume Profile
        """
        result = {
            'timestamp': int(time.time()),
            'current_price': current_price
        }
        
        # 1. OI Data
        oi_data = self.get_option_chain()
        if oi_data:
            result['pcr'] = oi_data['pcr']
            result['support'] = oi_data['support_wall']
            result['resistance'] = oi_data['resistance_wall']
            result['max_pain'] = oi_data['max_pain']
            result['oi_changes'] = oi_data.get('oi_changes', {})
        
        # 2. Smart Money
        smart_money = self.detect_smart_money(current_price)
        result['smart_money'] = smart_money
        result['smart_money_bias'] = smart_money.get('direction', 'NEUTRAL')
        
        # 3. Greeks & IV
        greeks = self.get_greeks_and_iv()
        if greeks:
            result['greeks'] = greeks
            result['iv_signal'] = self._analyze_iv(greeks)
        
        # 4. Volume
        volume = self.get_volume_profile()
        if volume:
            result['volume_profile'] = volume
            result['most_active_strike'] = max(volume, key=lambda x: volume[x]['volume'])
        
        return result
    
    def _analyze_iv(self, greeks_data: Dict) -> str:
        """Analyze Implied Volatility for signals"""
        iv = greeks_data.get('call_iv', 0)
        
        if iv > 80:
            return "HIGH_VOLATILITY - Big move expected"
        elif iv < 30:
            return "LOW_VOLATILITY - Range-bound market"
        else:
            return "NORMAL_VOLATILITY - Standard conditions"


    def get_institutional_bias(self, current_price: float) -> Dict:
        """
        Calculate a granular Institutional Bias Score (-10 to +10)
        based on all available data points.
        """
        try:
            oi_data = self.get_option_chain()
            if not oi_data:
                return {'score': 0, 'bias': 'NEUTRAL', 'reasons': ['No OI data available']}
            
            pcr = oi_data['pcr']
            max_pain = oi_data['max_pain']
            support = oi_data['support_wall']
            resistance = oi_data['resistance_wall']
            
            score = 0
            reasons = []
            
            # 1. PCR Analysis (Standard: 0.7 - 1.0)
            if pcr < 0.7:
                score += 2
                reasons.append(f"Bullish: Low PCR ({pcr:.2f}) indicates Call dominance")
            elif pcr > 1.2:
                score -= 2
                reasons.append(f"Bearish: High PCR ({pcr:.2f}) indicates Put dominance")
            
            # 2. Max Pain Gravity
            # Price tends to move towards Max Pain near expiration
            pain_diff_pct = (max_pain - current_price) / max(current_price, 1) if current_price > 0 else 0
            if abs(pain_diff_pct) > 0.01: # Significant difference
                if pain_diff_pct > 0:
                    score += 2
                    reasons.append(f"Bullish: Price ${current_price:.0f} is below Max Pain ${max_pain}")
                else:
                    score -= 2
                    reasons.append(f"Bearish: Price ${current_price:.0f} is above Max Pain ${max_pain}")
            
            # 3. Wall Proximity
            dist_to_support = (current_price - support) / max(current_price, 1) if current_price > 0 else 0
            dist_to_resistance = (resistance - current_price) / max(current_price, 1) if current_price > 0 else 0
            
            if abs(dist_to_support) < 0.02:
                score += 3
                reasons.append(f"Strong Bullish: Price near Major Support Wall ${support}")
            
            if abs(dist_to_resistance) < 0.02:
                score -= 3
                reasons.append(f"Strong Bearish: Price near Major Resistance Wall ${resistance}")
            
            # 4. Smart Money Flow (Live OI Changes)
            sm = self.detect_smart_money(current_price)
            if sm['detected']:
                flow_score = 4 if sm['direction'] == 'CALL' else -4
                score += flow_score
                reasons.append(f"Institutional Flow: {sm['details']}")
            
            # 5. Delta/Gamma Sentiment (if available)
            greeks = self.get_greeks_and_iv()
            if greeks:
                delta = greeks.get('call_delta', 0)
                if delta > 0.6: # Deep ITM Calls dominance
                    score += 1
                    reasons.append("Bullish: High ATM Call Delta")
                elif delta < 0.4:
                    score -= 1
                    reasons.append("Bearish: Low ATM Call Delta")

            # Finalize
            score = max(-10, min(10, score))
            bias = 'BULLISH' if score >= 3 else ('BEARISH' if score <= -3 else 'NEUTRAL')
            
            import math as _m
            _coverage = self.summarize_current_expiries()
            _vals = (pcr, max_pain, support, resistance, current_price)
            _fin = all(isinstance(v, (int, float)) and _m.isfinite(v) for v in _vals)
            _exp_ok = isinstance(_coverage, dict) and not _coverage.get('error') and _coverage.get('expiry_count', 0) > 0
            _reasons = []
            if not _fin:
                _reasons.append('non_finite_oi_metrics')
            if not _exp_ok:
                _reasons.append('expiry_coverage_unavailable')
            _validation = {'usable': bool(_fin and _exp_ok), 'complete': bool(_fin and _exp_ok),
                           'reasons': _reasons, 'provider': 'deribit', 'source': 'public_get_book_summary_by_currency'}
            return {
                'score': score,
                'bias': bias,
                'reasons': reasons,
                'raw_data': {
                    'pcr': pcr,
                    'max_pain': max_pain,
                    'support': support,
                    'resistance': resistance,
                    'current_price': current_price,
                    'expiry_coverage': _coverage,
                    'options_validation': _validation,
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating institutional bias: {e}")
            return {'score': 0, 'bias': 'NEUTRAL', 'reasons': [f"Error: {str(e)}"]}

    def _calculate_greeks_manual(self, S, K, T, r, sigma, option_type='C'):
        """Manual Black-Scholes Greeks calculation for profile mapping"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return {'delta': 0, 'gamma': 0}
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            normal_pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
            normal_cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
            gamma = normal_pdf / (S * sigma * math.sqrt(T))
            if option_type == 'C':
                delta = normal_cdf
            else:
                delta = normal_cdf - 1
            return {'delta': delta, 'gamma': gamma}
        except Exception:
            return {'delta': 0, 'gamma': 0}

    def get_market_regime_analysis(self, current_price: float) -> Dict:
        """
        Differentiate between WHALE dominance and RETAIL behavior.
        Uses Concentration index and Volume/OI profiles.
        """
        try:
            profile = self.get_volume_profile()
            if not profile: return {"regime": "UNKNOWN", "thought": "No volume data for regime analysis"}
            
            # 1. WHALE DETECTION: Concentration Analysis (HHI-like)
            total_vol = sum(v['volume'] for v in profile.values())
            vols = [v['volume'] for v in profile.values() if v['volume'] > 0]
            if not vols: return {"regime": "RETAIL", "thought": "Low volume environment"}
            
            sorted_vols = sorted(vols, reverse=True)
            top_3_pct = sum(sorted_vols[:3]) / (total_vol + 1e-9)
            
            # 2. GEX Analysis (Gamma Exposure Approximation)
            greeks_atm = self.get_greeks_and_iv()
            atm_iv = (greeks_atm.get('call_iv', 0) / 100) if (greeks_atm and greeks_atm.get('call_iv')) else 0.50
            
            total_gex = 0
            for strike, data in profile.items():
                if abs(strike - current_price) / current_price > 0.1: continue
                # Approx 7 days to expiry
                g = self._calculate_greeks_manual(current_price, strike, 7/365, 0.05, atm_iv)
                total_gex += (data['oi'] * g['gamma'] * current_price * 0.01)
            
            regime = "WHALE_ACTIVE" if top_3_pct > 0.45 else "RETAIL_DOMINANT"
            
            # 3. RETAIL TRAP DETECTION
            traps = []
            for strike, data in profile.items():
                # OTM high volume but low OI change is usually a retails gambling zone
                if data['volume'] > (total_vol * 0.08) and abs(strike - current_price) / current_price > 0.02:
                    traps.append(f"${strike}")
            
            thought = f"REGIME: {regime}. "
            if regime == "WHALE_ACTIVE":
                thought += f"Institutional concentration high ({top_3_pct*100:.1f}% in top 3 strikes). GEX: {total_gex:.1f}."
            else:
                thought += f"Retail liquidity dominant. Traps clustered around {', '.join(traps[:2]) if traps else 'none'}."
            
            return {
                "regime": regime,
                "concentration": top_3_pct,
                "gex": total_gex,
                "retail_traps": traps,
                "thought": thought
            }
        except Exception as e:
            return {"regime": "NEUTRAL", "thought": f"Regime analysis failure: {str(e)}"}

# Test/Demo
if __name__ == "__main__":
    import sys
    import io
    
    # Force UTF-8 for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    logging.basicConfig(level=logging.INFO)
    
    client = DeribitOptionsClient(currency='BTC')
    
    print("\n" + "="*60)
    print("      JARVIS DEEP DERIBIT ANALYSIS")
    print("="*60)
    
    # Get current price
    try:
        ticker_url = "https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL"
        import requests as req
        price_resp = req.get(ticker_url, timeout=10)
        current_price = price_resp.json()['result']['last_price']
        print(f"BTC Spot Price: ${current_price:.2f}")
    except:
        current_price = 87000
        print(f"Using Fallback Price: ${current_price}")
    
    print("\nAnalyzing Option Chain...")
    bias_data = client.get_institutional_bias(current_price)
    
    print(f"\nINSTITUTIONAL BIAS: {bias_data['bias']} (Score: {bias_data['score']}/10)")
    print("-" * 60)
    for reason in bias_data['reasons']:
        print(f" • {reason}")
    print("-" * 60)
    
    print("\n[FULL STATS]")
    stats = bias_data['raw_data']
    print(f" PCR: {stats['pcr']:.3f}")
    print(f" Max Pain: ${stats['max_pain']}")
    print(f" Support: ${stats['support']}")
    print(f" Resistance: ${stats['resistance']}")
    print("="*60 + "\n")
