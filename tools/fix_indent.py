with open('mode_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

shadow_xgb_start = 286 # index 286 is line 287
shadow_xgb_end = 310 # index 310 is line 311 (non-inclusive)

shadow_xgb_lines = lines[shadow_xgb_start:shadow_xgb_end]
del lines[shadow_xgb_start:shadow_xgb_end]

# Find where final_brain_votes starts
insert_idx = 0
for i, line in enumerate(lines):
    if 'Synthesize final vote per brain across ALL timeframes' in line:
        insert_idx = i - 1
        break

lines[insert_idx:insert_idx] = shadow_xgb_lines

with open('mode_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Moved shadow XGBoost block.")
