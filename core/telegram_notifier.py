"""
telegram_notifier.py - Stub module for Telegram notifications
Jarvis system can run without Telegram. This module provides
safe fallback functions so the system doesn't crash on import.
"""
import logging

logger = logging.getLogger(__name__)


def send_trading_signal(signal: dict, chat_id: str = None, token: str = None) -> bool:
    """
    Send a trading signal notification via Telegram.
    This is a stub - configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID 
    in environment variables to enable real notifications.
    """
    try:
        import os
        bot_token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        telegram_chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        
        if not bot_token or not telegram_chat_id:
            logger.debug("[Telegram] Not configured - skipping notification")
            return False
        
        import requests
        direction = signal.get("direction", "N/A")
        score = signal.get("score", 0)
        price = signal.get("entry_price", 0)
        
        msg = (
            f"🤖 *JARVIS SIGNAL*\n"
            f"Direction: `{direction}`\n"
            f"Confidence: `{score}%`\n"
            f"Entry: `${price:,}`\n"
        )
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": telegram_chat_id,
            "text": msg,
            "parse_mode": "Markdown"
        }, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.debug(f"[Telegram] Notification skipped: {e}")
        return False


def send_message(text: str, chat_id: str = None, token: str = None) -> bool:
    """Send a plain text message via Telegram."""
    return send_trading_signal({"direction": text, "score": 0, "entry_price": 0}, chat_id, token)

def send_daily_report(stats: dict, chat_id: str = None, token: str = None) -> bool:
    """Send daily performance summary via Telegram."""
    try:
        import os
        bot_token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        telegram_chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        
        if not bot_token or not telegram_chat_id:
            logger.debug("[Telegram] Not configured - skipping daily report")
            return False
            
        import requests
        
        msg = (
            f"📊 *JARVIS DAILY REPORT*\n"
            f"======================\n"
            f"📈 Trades Taken: `{stats.get('trades_taken', 0)}`\n"
            f"🏆 Win Rate: `{stats.get('win_rate', 0):.1f}%`\n"
            f"💰 Total P&L: `${stats.get('pnl', 0):.2f}`\n\n"
            
            f"🥇 Best Trade: `${stats.get('best_trade', 0):.2f}`\n"
            f"💔 Worst Trade: `${stats.get('worst_trade', 0):.2f}`\n\n"
            
            f"🛡️ PreSim Vetoes: `{stats.get('presim_vetoes', 0)}`\n"
            f"⚠️ Data Rejects: `{stats.get('dv_rejects', 0)}`\n\n"
            
            f"🧠 Top Engines:\n"
            f"{stats.get('top_engines', 'N/A')}\n\n"
            
            f"⏱️ Uptime: `{stats.get('uptime', 'N/A')}`\n"
            f"❌ Errors: `{stats.get('errors', 0)}`"
        )
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": telegram_chat_id,
            "text": msg,
            "parse_mode": "Markdown"
        }, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.debug(f"[Telegram] Daily report skipped: {e}")
        return False
