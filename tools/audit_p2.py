# Now let's check if mode_engine passes the correct dict structure to MetaFusionBrain
# From mode_engine.py lines 378-391:
# mf_res = self.p1_meta_fusion.analyze_meta_fusion({
#     'trend': t_res, 'volatility': v_res, 'strength': st_res, 'risk': rk_res,
#     'reversal': rv_res, 'regime': reg_res, 'deepseek': ds_res, 'evolution': evo_res,
#     'memory': mem_res, 'self_healing': heal_res, 'mini_r1': r1_res, 'mini_v3': v3_res
# })

# MetaFusionBrain iterates: for brain_name, brain_data in all_brain_support.items()
# and accesses brain_data.get('support_score', 0)
# All Part 1 brains return support_score, so this looks correct!

# But we need to check: does MetaFusionBrain need exactly those key names?
# Looking at: all_brain_support has 'trend', 'volatility', 'strength', etc.
# These are all the P1 brains' results - and each result dict has 'support_score'
# So MetaFusionBrain just aggregates support_scores => this is CORRECT

# Now let's check the p2_brains structure
print('Checking p2_brains structure...')

with open('part2_FIXED.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find AdvancedAnalysisSystem.brains attribute
import re
# Find the brains dict initialization
brains_matches = re.findall(r'brains\s*=\s*\{[^}]+\}', content, re.DOTALL)
for m in brains_matches[:3]:
    print(m[:500])
    print('---')
