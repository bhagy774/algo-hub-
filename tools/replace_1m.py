import sys

content = open('jarvis_FIXED.py', 'r', encoding='utf-8').read()
content = content.replace("'1m'", "'5m'")
content = content.replace("last_1m_entry", "last_5m_entry")
content = content.replace("_refine_entry_with_1m", "_refine_entry_with_5m")
content = content.replace("entry_gate_1m", "entry_gate_5m")
content = content.replace("1m_entry", "5m_entry")
content = content.replace("1m confirm", "5m confirm")
content = content.replace("1m trigger", "5m trigger")
content = content.replace("1M-OFF", "5M-OFF")
content = content.replace("1M-FALLBACK", "5M-FALLBACK")
content = content.replace("1m:", "5m:")
content = content.replace("1m-swing", "5m-swing")

open('jarvis_FIXED.py', 'w', encoding='utf-8').write(content)
print("Replaced 1m with 5m in jarvis_FIXED.py")
