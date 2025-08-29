#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Alphas and corresponding directories
ENTRIES = [
    (0.0, Path('./logs/mda2_a00/mda_summary.json')),
    (0.1, Path('./logs/mda2_a01/mda_summary.json')),
    (0.2, Path('./logs/mda2_a02/mda_summary.json')),
    (0.3, Path('./logs/mda2_a03/mda_summary.json')),
    (0.4, Path('./logs/mda2_a04/mda_summary.json')),
    (0.5, Path('./logs/mda2_a05/mda_summary.json')),
]

rows = []
for alpha, path in ENTRIES:
    if not path.exists():
        continue
    with open(path, 'r') as f:
        data = json.load(f)
    counts = data.get('amplified_counts', {})
    samples = max(int(data.get('samples', 100)), 1)
    intel = counts.get('intelligent', 0)
    indep = counts.get('independent', 0)
    rows.append({'alpha': alpha, 'trait': 'intelligent', 'percent': 100.0 * intel / samples})
    rows.append({'alpha': alpha, 'trait': 'independent', 'percent': 100.0 * indep / samples})

if not rows:
    raise SystemExit('No summaries found; run amplification first.')

# Dataframe
_df = pd.DataFrame(rows)
_df.sort_values(by='alpha', inplace=True)

# Plot
sns.set_theme(style='whitegrid', context='talk', font_scale=1.0)
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=_df, x='alpha', y='percent', hue='trait', marker='o', linewidth=2.5, markersize=8)
ax.set_title('Model-Diff Amplification: Trait Percentage vs Alpha', pad=12)
ax.set_xlabel('Alpha (amplification strength)')
ax.set_ylabel('Percentage of answers (%)')
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_ylim(0, 100)
ax.legend(title='Trait', loc='best', frameon=True)

plt.tight_layout()
out_path = Path('./logs/mda2_alpha_sweep.png')
plt.savefig(out_path, dpi=200)
print(f'Saved plot to: {out_path}')
