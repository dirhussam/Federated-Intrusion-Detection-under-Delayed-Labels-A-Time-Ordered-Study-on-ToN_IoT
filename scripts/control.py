#!/usr/bin/env python3
"""Controlled check of the error monitor: pure class-prior shift vs. a real conditional change.

A warm-started model is frozen. Records of known types from the tuning segment are resampled
i.i.d. into synthetic monitoring streams, so P(X|Y) is fixed by construction inside each type.
  prior:  attack share moves from 0.2 to 0.6 at round 50 (scanning and normal only)
  prior_rev: attack share moves from 0.6 to 0.2 at round 50
  concept: attack share fixed at 0.2, but from round 50 attacks are DoS (unseen in warm-up)
We count increase alarms of a pooled ADWIN and of class-conditional ADWINs fed with the same
Brier errors and the same per-round cap as in the main simulation.
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim import load_stream, MLP, warm_start, ErrorMonitor

D = load_stream(sys.argv[1], 8, 8000); X = np.ascontiguousarray(D['X'])
D['X'] = X
best = json.load(open(sys.argv[2])); out = sys.argv[3]
seg = (D['rnd'] >= D['R_w']) & (D['rnd'] < D['R_t'])
pool = {t: np.flatnonzero(seg & (D['types'] == t)) for t in ('normal', 'scanning', 'dos')}
ROUNDS, PER, SHIFT = 100, 1000, 50
res = []
for seed in range(10):
    net = MLP(X.shape[1]); th = warm_start(net, D, seed, 5)
    err = {t: (net.predict(th, X[i]) - (t != 'normal')) ** 2 for t, i in pool.items()}
    for scen in ('prior', 'prior_rev', 'concept'):
        for rep in range(3):
            rng = np.random.default_rng(seed * 10 + rep)
            mons = {cc: ErrorMonitor(best['delta'], cc, 250, np.random.default_rng(rep)) for cc in (False, True)}
            alarms = {False: [], True: []}
            for r in range(ROUNDS):
                if scen == 'prior':
                    share = 0.6 if r >= SHIFT else 0.2
                elif scen == 'prior_rev':
                    share = 0.2 if r >= SHIFT else 0.6
                else:
                    share = 0.2
                atk = 'dos' if (scen == 'concept' and r >= SHIFT) else 'scanning'
                na = rng.binomial(PER, share)
                e = np.concatenate([rng.choice(err['normal'], PER - na), rng.choice(err[atk], na)])
                y = np.r_[np.zeros(PER - na, int), np.ones(na, int)]
                for cc in (False, True):
                    if mons[cc].update(e, y):
                        alarms[cc].append(r)
            for cc in (False, True):
                a = np.array(alarms[cc])
                post = a[a >= SHIFT]
                res.append(dict(seed=seed, rep=rep, scenario=scen, classcond=cc,
                                alarms_before=int((a < SHIFT).sum()), alarms_after=int(len(post)),
                                first_delay=int(post[0] - SHIFT) if len(post) else None,
                                mean_err=dict((t, float(v.mean())) for t, v in err.items())))
    print(seed, flush=True)
json.dump(res, open(out, 'w'), indent=1)
