import pandas as pd
df = pd.read_csv('submission_history.csv')
md = '# Orbit Wars Submission History\n\n'
md += '| Ref | Date | Score | Description | Status |\n'
md += '|-----|------|-------|-------------|--------|\n'
for _, row in df.iterrows():
    desc = str(row['description']).replace('\n', ' ')
    md += f'| {row["ref"]} | {row["date"]} | {row["publicScore"]} | {desc} | {row["status"]} |\n'

md += """
## Performance Analysis (v128 vs v129/v130)

### v128 (Ref: 53860097, Score: 1138.0) - High Water Mark
- **Script**: `scripts_archive/v128_merged_main.py`
- **ROI Threshold**: 1.35 (Conservative)
- **Reinforcement Beta**: 2.2 (High safety)
- **Strategy**: Single-source strikes with robust adaptive scaling and ring conquest.
- **Why it worked**: Strong focus on high-value, safe captures. Avoided ship bleeding.

### v129 (Ref: 53861922, Score: 1046.8) - First Multi-Source + Sun-Skimming
- **ROI Threshold**: 1.12 (Highly Aggressive)
- **Added**: Multi-source synchronization and sun-skimming.
- **Score Impact**: ~90 point drop.
- **Observation**: The low ROI threshold likely caused the agent to take too many risks. Multi-source strikes are powerful but expensive; sending 3 fleets to one target leaves 3 planets vulnerable.

### v130 (Ref: 53882607, Score: 942.9) - Tuned Multi-Source
- **ROI Threshold**: 1.25 (Moderate)
- **Reinforcement Beta**: 0.0 in 4P (Zero safety)
- **Score Impact**: Further ~100 point drop.
- **Observation**: Disabling safety in 4P was likely a major mistake. In crowded maps, ignoring enemy reinforcement leads to catastrophic losses. Also, the multi-source logic replaced single-source candidates instead of supplementing them, reducing the planner's flexibility.

### v131 (Latest) - Hybrid Strategy
- **Script**: `scripts_archive/v131_hybrid_main.py`
- **Safety Restored**: Set `reinforce_size_beta` back to 2.2 for all modes.
- **Conservative ROI**: Returned to `roi_threshold = 1.35`.
- **Supplemented Logic**: Modified `plan_lite_waves` to include BOTH single-source and multi-source candidates in the same greedy pool.
- **Mechanical Advantage**: Retained Sun-Skimming logic.
"""

with open('SUBMISSION_HISTORY.md', 'w') as f:
    f.write(md)
