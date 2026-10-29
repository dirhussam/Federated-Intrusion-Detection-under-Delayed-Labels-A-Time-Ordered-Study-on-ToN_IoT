#!/usr/bin/env python3
"""Aggregate run records into tables, paired tests and figures."""
import json, os, sys, collections
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
recs = []
for f in ('main.jsonl', 'variant.jsonl'):
    p = os.path.join(R, f)
    if os.path.exists(p):
        recs += [json.loads(l) for l in open(p)]
rows = []
for r in recs:
    e = dict(r['eval']); e.update(name=r['name'], delay=r['cfg']['delay'], seed=r['seed'],
                                  tune_f1=r['tune']['macro_f1'])
    rows.append(e)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(R, 'all_runs_eval.csv'), index=False)

cols = ['macro_f1', 'attack_recall', 'fpr', 'pr_auc', 'min_client_f1', 'byte_ratio', 'client_updates',
        'aggregations', 'alarms', 'tune_f1']
g = df.groupby(['delay', 'name'])
summ = g[cols].agg(['mean', 'std']).round(4)
summ['n'] = g.size()
summ.to_csv(os.path.join(R, 'summary_mean_std.csv'))
print(summ.xs('mean', axis=1, level=1).round(3).to_string())

# per attack type recall
tcols = [c for c in df.columns if c.startswith('recall_')]
pt = df.groupby(['delay', 'name'])[tcols].mean().round(3)
pt.to_csv(os.path.join(R, 'per_type_recall.csv'))

# paired tests against the reference method, seed-paired
ref = sys.argv[1] if len(sys.argv) > 1 else 'pfl'
tests = []
for d in sorted(df.delay.unique()):
    A = df[(df.delay == d) & (df.name == ref)].set_index('seed')
    for nm in sorted(df.name.unique()):
        if nm == ref:
            continue
        B = df[(df.delay == d) & (df.name == nm)].set_index('seed')
        s = A.index.intersection(B.index)
        if len(s) < 5:
            continue
        diff = (A.loc[s, 'macro_f1'] - B.loc[s, 'macro_f1']).values
        try:
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = np.nan
        rng = np.random.default_rng(0)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
        tests.append(dict(delay=d, ref=ref, other=nm, n=len(s), mean_diff=diff.mean(),
                          ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                          wins=int((diff > 0).sum()), p_wilcoxon=p))
T = pd.DataFrame(tests)
T.to_csv(os.path.join(R, f'paired_tests_vs_{ref}.csv'), index=False)
print(T.round(4).to_string())
