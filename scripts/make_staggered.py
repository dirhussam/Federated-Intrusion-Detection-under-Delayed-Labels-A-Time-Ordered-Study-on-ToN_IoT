#!/usr/bin/env python3
"""Controlled 'staggered exposure' stream: each attack type reaches the gateways one after another.

For attack type t with first time t0 and duration T_t, gateway k (in a random order per type,
seed 0) loses the records of type t observed before t0 + (j/K) * 0.5 * T_t, where j is its
position in that order. The first gateway sees the whole attack and the last one only its second
half. Nothing is relabelled or synthesised; records are only removed.
Usage: python make_staggered.py IN.pkl OUT.pkl
"""
import sys, zlib, json
import numpy as np, pandas as pd

K = 8
df = pd.read_pickle(sys.argv[1])
client = np.array([zlib.crc32(s.encode()) % K for s in df['dst_ip'].astype(str).values])
rng = np.random.default_rng(0)
drop = np.zeros(len(df), bool); info = {}
for t in sorted(df['type'].unique()):
    if t == 'normal':
        continue
    m = (df['type'] == t).values
    t0, t1 = df['ts'].values[m].min(), df['ts'].values[m].max()
    order = rng.permutation(K)
    for j, k in enumerate(order):
        cut = t0 + (j / K) * 0.5 * (t1 - t0)
        drop |= m & (client == k) & (df['ts'].values < cut)
    info[t] = dict(order=order.tolist(), start=int(t0), end=int(t1))
out = df[~drop].reset_index(drop=True)
out.to_pickle(sys.argv[2])
json.dump(dict(dropped=int(drop.sum()), kept=len(out), types=info,
               counts=out['type'].value_counts().to_dict()), open(sys.argv[2] + '.info.json', 'w'), indent=1)
print(drop.sum(), len(out)); print(out['type'].value_counts())
