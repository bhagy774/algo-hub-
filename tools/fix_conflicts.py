import sys

def resolve_conflicts(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out_lines = []
    i = 0
    conflict_idx = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('<<<<<<< HEAD'):
            conflict_idx += 1
            # Read until =======
            head_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('======='):
                head_lines.append(lines[i])
                i += 1
            
            # Read until >>>>>>> pr107
            i += 1
            pr107_lines = []
            while i < len(lines) and not lines[i].startswith('>>>>>>>'):
                pr107_lines.append(lines[i])
                i += 1
                
            if conflict_idx == 1:
                # Conflict 1: Keep both helper function blocks
                out_lines.extend(head_lines)
                out_lines.append('\n')
                out_lines.extend(pr107_lines)
            elif conflict_idx == 2:
                # Conflict 2: We want to keep BOTH. First HEAD (the XGB shadow loop), then PR107 (the Parts 4-12 loop)
                out_lines.extend(head_lines)
                out_lines.extend(pr107_lines)
            elif conflict_idx == 3:
                # Conflict 3: Reason strings and shadow report
                # Head: "reason": "NO_29_BRAIN_CONSENSUS", \n "xgb_shadow": ...
                # PR107: "reason": f"NO_{len(votes)}_BRAIN_CONSENSUS"
                # Keep PR107 reason and HEAD shadow report
                shadow_line = [l for l in head_lines if 'xgb_shadow' in l][0]
                reason_line = [l for l in pr107_lines if 'reason' in l][0].rstrip()
                out_lines.append(reason_line + ',\n')
                out_lines.append(shadow_line)
            elif conflict_idx == 4:
                # Conflict 4: Reason strings and shadow report
                # Similar logic
                shadow_line = [l for l in head_lines if 'xgb_shadow' in l][0]
                reason_line = [l for l in pr107_lines if 'reason' in l][0].rstrip()
                out_lines.append(reason_line + ',\n')
                out_lines.append(shadow_line)
            
            i += 1 # skip >>>>>>>
        else:
            out_lines.append(line)
            i += 1

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

resolve_conflicts('mode_engine.py')
print("Python resolution script done.")
