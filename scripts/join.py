#!/usr/bin/env python3
"""Cold-start check: one gateway joins at the start of the evaluation segment.
local: it starts from the shared warm-up model (trained without it) and adapts alone.
pfl:   it starts from the current global model and then personalises."""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import f1_score
from sim import load_stream, simulate, warm_start, MLP
D = None
def init(p):
    global D
    D = load_stream(p, 8, 8000); D['X'] = np.ascontiguousarray(D['X'])
def job(a):
    k, seed, best, delay = a
    th0 = warm_start(MLP(D['X'].shape[1]), D, seed, 5, exclude=k)
    out = []
    for name, cfg in [('local_adwin_replay', dict(kind='local')),
                      ('pfl', dict(kind='pfl', lam=best['variant']['lam'], sched=best['variant']['sched']))]:
        cfg.update(lr=best['lr'], delta=best['delta'], delay=delay, join=k)
        P, _, _, _ = simulate(D, cfg, seed, th0)
        rec = dict(client=k, seed=seed, delay=delay, name=name)
        for tag, hi in (('first24', D['R_t'] + 24), ('all', D['R'])):
            m = (D['client'] == k) & (D['rnd'] >= D['R_t']) & (D['rnd'] < hi)
            yy = D['y'][m]; yh = (P[m] >= 0.5).astype(int)
            rec[f'f1_{tag}'] = f1_score(yy, yh, average='macro') if len(np.unique(yy)) == 2 else None
            rec[f'acc_{tag}'] = float((yy == yh).mean()); rec[f'n_{tag}'] = int(m.sum())
        out.append(rec)
    return out
if __name__ == '__main__':
    best = json.load(open(sys.argv[2])); res = []
    if len(sys.argv) > 4 and os.path.exists(sys.argv[4]):
        res = json.load(open(sys.argv[4]))          # resume from an earlier partial run
    have = {(r['seed'], r['client']) for r in res}
    tasks = [(k, s, best, 1) for s in range(3) for k in range(8) if (s, k) not in have]
    with ProcessPoolExecutor(2, initializer=init, initargs=(sys.argv[1],)) as ex:
        for o in ex.map(job, tasks):
            res += o; print(o, flush=True)
            json.dump(res, open(sys.argv[3], 'w'), indent=1)
    json.dump(res, open(sys.argv[3], 'w'), indent=1)
