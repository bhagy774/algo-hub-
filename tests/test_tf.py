import glob, re

tfs = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
pattern = re.compile(r'mtf_context\.get\([\'\"](\w+)[\'\"]\)')
pattern2 = re.compile(r'mtf_data\.get\([\'\"](\w+)[\'\"]\)')
pattern3 = re.compile(r'native_frames\[[\'\"](\w+)[\'\"]\]')
pattern4 = re.compile(r'parts_by_timeframe\[[\'\"](\w+)[\'\"]\]')
pattern5 = re.compile(r'timeframes.*?\[(.*?)\]')

results = {}
for f in glob.glob('c:\\jarvis\\*FIXED*.py'):
    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        name = f.split('\\')[-1]
        matches = set(pattern.findall(content) + pattern2.findall(content) + pattern3.findall(content) + pattern4.findall(content))
        if matches:
            print(f'{name}: {sorted(list(matches))}')

print("DONE")
