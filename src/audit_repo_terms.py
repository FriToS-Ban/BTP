"""
Script to audit the codebase for:
1. Invented dates (e.g. 'August 2026')
2. Unhedged terms ('prove', 'proves', 'proving', 'proof', 'demonstrates conclusively', 'irreducible')
3. Unverified commercial brand names or processing assertions ('Custom 465', 'Ferrium', 'peak-aged', 'unaged')
"""

import os
import glob
import re

patterns = {
    'forbidden_words': re.compile(r'\b(prove|proves|proving|proof|demonstrates conclusively|irreducible)\b', re.IGNORECASE),
    'unverified_names': re.compile(r'\b(Custom 465|Ferrium|peak-aged|unaged)\b', re.IGNORECASE),
    'invented_dates': re.compile(r'\b(August 2026)\b', re.IGNORECASE)
}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
files = glob.glob(os.path.join(BASE_DIR, '**', '*'), recursive=True)

matches = {k: [] for k in patterns}

for f in files:
    if f.endswith(('.md', '.py', '.txt', '.csv', '.ipynb')) and 'site-packages' not in f and '.git' not in f and 'audit_repo_terms.py' not in f:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                for idx, line in enumerate(fp, 1):
                    for k, pat in patterns.items():
                        m = pat.search(line)
                        if m:
                            matches[k].append((os.path.relpath(f, BASE_DIR), idx, m.group(0), line.strip()))
        except Exception:
            pass

print("=== CODEBASE AUDIT SEARCH RESULTS ===")
total = sum(len(m) for m in matches.values())
print(f"Total findings: {total}")
for k, mlist in matches.items():
    print(f"\n{k.upper()}: {len(mlist)} matches")
    for item in mlist:
        print(f"  [{item[0]}:L{item[1]}] found '{item[2]}': {item[3][:100]}")
