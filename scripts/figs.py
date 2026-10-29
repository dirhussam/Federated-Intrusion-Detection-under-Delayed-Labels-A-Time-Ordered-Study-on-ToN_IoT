#!/usr/bin/env python3
"""Paper figures (vector PDF, IEEE single column)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(_ROOT, 'results'); OUT = os.path.join(_ROOT, 'figs'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'legend.frameon': False, 'pdf.fonttype': 42})
# validated categorical slots, fixed order; line styles give a second encoding for print
S = {'frozen': ('Frozen', '#2a78d6', (0, (1, 1.2))),
     'fedavg_p1': ('FedAvg, every round', '#eb6834', (0, (5, 1.5))),
     'proposed': ('Shared model, triggered', '#1baf7a', (0, (3, 1, 1, 1))),
     'local_adwin_replay': ('Local only', '#eda100', (0, (1, 0))),
     'pfl': ('Personalised (ours)', '#e87ba4', (0, (1, 0)))}

# ---------------------------------------------------------------- timeline
fig, (ax0, ax) = plt.subplots(2, 1, figsize=(7.0, 2.45), sharex=True,
                              gridspec_kw=dict(height_ratios=[0.17, 1], hspace=0.05))
base = pd.read_csv(f'{R}/pred/pfl_d1_s0_rounds.csv')
types = [c[2:] for c in base.columns if c.startswith('n_') and c != 'n_normal']
cnt = base[[f'n_{t}' for t in types]].fillna(0).values
dom = np.where(cnt.sum(1) > 0, np.array(types)[cnt.argmax(1)], 'normal')   # benign-only rounds
t0 = pd.to_datetime(base.ts, unit='s')
hrs = (base.ts - base.ts.iloc[0]) / 3600.0
x = base['round'].values
shade = ['#f0efec', '#dcdbd6']
prev, start, i = None, 0, 0
blocks = []
for j, d in enumerate(dom):
    if d != prev:
        if prev is not None:
            blocks.append((prev, start, j))
        prev, start = d, j
blocks.append((prev, start, len(dom)))
for n, (d, a, b) in enumerate(blocks):
    ax0.axvspan(x[a] - 0.5, x[b - 1] + 0.5, color=shade[n % 2], lw=0)
    full = {'scanning': 'Scanning', 'dos': 'DoS', 'injection': 'Injection', 'ddos': 'DDoS',
            'password': 'Password', 'xss': 'XSS', 'ransomware': 'Ransomware', 'backdoor': 'Backdoor',
            'mitm': 'MITM', 'normal': 'Benign'}
    if b - a >= 3:
        ax0.text((x[a] + x[b - 1]) / 2, 0.5, full.get(d, d),
                 ha='center', va='center', fontsize=6.6, color='#0b0b0b')
    ax0.axvline(x[a] - 0.5, color='#fcfcfb', lw=1.2)
ax0.set_yticks([]); ax0.set_ylim(0, 1)
for s in ax0.spines.values():
    s.set_visible(False)
for key, (lab, col, ls) in S.items():
    d = pd.read_csv(f'{R}/pred/{key}_d1_s0_rounds.csv')
    yv = d['macro_f1'].rolling(3, min_periods=1, center=True).mean()
    ax.plot(d['round'], yv, color=col, lw=1.4 if key == 'pfl' else 1.0, ls=ls, label=lab,
            zorder=3 if key == 'pfl' else 2)
ax.axvline(base['round'].iloc[0] + 32 - 0.5, color='#52514e', lw=0.6, ls=(0, (2, 2)))
ax.text(base['round'].iloc[0] + 32.6, 0.03, 'evaluation starts', fontsize=6.4, color='#52514e', va='bottom')
ax.set_ylabel('Macro-F1 per round'); ax.set_xlabel('Round (8,000 records each, time order)')
ax.set_ylim(0, 1.02); ax.grid(axis='y', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(ncol=5, fontsize=6.6, loc='upper center', bbox_to_anchor=(0.5, -0.24), handlelength=2.8,
          columnspacing=1.6)
ax.set_xlim(x[0] - 1, x[-1] + 1)
fig.savefig(f'{OUT}/timeline.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- trade-off
df = pd.read_csv(f'{R}/all_runs_eval.csv')
d1 = df[df.delay == 1]
show = {'frozen': 'Frozen', 'fedavg_p1': 'FedAvg (every round)', 'fedavg_p10': 'FedAvg (every 10)',
        'periodic_matched_replay': 'Periodic, matched', 'proposed': 'Shared, triggered',
        'proposed_sched': 'Shared, triggered, W=5', 'local_adwin_replay': 'Local only',
        'pfl': 'Personalised (ours)'}
fig, ax = plt.subplots(figsize=(3.45, 2.2))
g = d1.groupby('name')
off = {'local_adwin_replay': (0.012, -0.03), 'pfl': (0.012, 0.012), 'frozen': (0.012, 0.0),
       'fedavg_p1': (-0.015, 0.0), 'fedavg_p10': (0.012, -0.03), 'periodic_matched_replay': (0.012, 0.012),
       'proposed': (0.012, 0.0), 'proposed_sched': (0.012, -0.035)}
for nm, lab in show.items():
    if nm not in g.groups:
        continue
    G = g.get_group(nm)
    mx, my = G.byte_ratio.mean(), G.macro_f1.mean(); sy = G.macro_f1.std()
    hl = nm == 'pfl'
    ax.errorbar(mx, my, yerr=sy, fmt='o', ms=4.5 if hl else 3.5, color='#e87ba4' if hl else '#2a78d6',
                ecolor='#9a9994', elinewidth=0.6, capsize=1.5, mec='#fcfcfb', mew=0.8, zorder=3)
    dx, dy = off.get(nm, (0.012, 0))
    ax.text(mx + dx, my + dy, lab, fontsize=5.8, color='#0b0b0b', ha='right' if dx < 0 else 'left',
            va='center', fontweight='bold' if hl else 'normal')
ax.set_xlabel('Bytes relative to FedAvg with all clients every round')
ax.set_ylabel('Macro-F1, evaluation segment')
ax.set_xlim(-0.04, 1.12); ax.grid(color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.savefig(f'{OUT}/tradeoff.pdf', bbox_inches='tight'); plt.close(fig)
print('ok')
