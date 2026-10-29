#!/usr/bin/env python3
"""Run every configuration x label delay x seed and append one JSON line per run.

Usage: python run_all.py STREAM.pkl OUT.jsonl [--seeds 10] [--jobs 2] [--stage tune|main]
Already-finished (config, delay, seed) triples in OUT.jsonl are skipped, so the script can resume.
"""
import argparse, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

D = None; TH0 = {}


def init(path):
    global D
    from sim import load_stream
    D = load_stream(path, K=8, n_round=8000)
    D['X'] = np.ascontiguousarray(D['X'])


def job(args):
    name, cfg, seed, save_pred = args
    from sim import simulate, metrics, warm_start, MLP, per_round
    key = seed
    if key not in TH0:
        TH0[key] = warm_start(MLP(D['X'].shape[1]), D, seed, 5)
    t = time.time()
    P, ver, log, dp = simulate(D, cfg, seed, TH0[key])
    rec = dict(name=name, cfg=cfg, seed=seed, secs=round(time.time() - t, 1),
               eval=metrics(D, P, log, dp, cfg, 'eval'), tune=metrics(D, P, log, dp, cfg, 'tune'),
               d_theta=dp)
    if save_pred:
        np.save(save_pred, P.astype(np.float32))
        per_round(D, P).to_csv(save_pred.replace('.npy', '_rounds.csv'), index=False)
        json.dump(dict(alarms=log['alarms'], warnings=log['warnings'], updates=log['updates']),
                  open(save_pred.replace('.npy', '_events.json'), 'w'))
    return rec


def configs(stage, best):
    lr, delta = best.get('lr', 0.01), best.get('delta', 0.002)
    base = dict(lr=lr, delta=delta)
    if stage == 'tune':
        out = []
        for l in (1e-3, 1e-2):
            for dl in (1e-3, 1e-2):
                out.append((f'tune_lr{l}_d{dl}', dict(kind='triggered', delay=1, lr=l, delta=dl)))
        for l in (1e-3, 1e-2):
            out.append((f'tune_periodic_lr{l}', dict(kind='periodic', period=1, delay=1, lr=l,
                                                     replay=False, mu=0.0)))
        return out
    if stage == 'tune2':
        out = []
        for lam in (0.01, 0.1, 1.0):
            for W in (1, 5):
                out.append((f'tune2_pfl_lam{lam}_W{W}', dict(base, kind='pfl', delay=1, lam=lam, sched=W)))
        out.append(('tune2_triggered_W5', dict(base, kind='triggered', delay=1, sched=5)))
        out.append(('tune2_local', dict(base, kind='local', delay=1)))
        return out
    if stage == 'stagger':
        out = []
        for d in (1, 5):
            B = dict(base, delay=d)
            out += [('frozen', dict(B, kind='frozen')),
                    ('fedavg_p1', dict(B, kind='periodic', period=1, replay=False, mu=0.0)),
                    ('fedavg_p10', dict(B, kind='periodic', period=10, replay=False, mu=0.0)),
                    ('local_adwin_replay', dict(B, kind='local')),
                    ('proposed', dict(B, kind='triggered'))]
            out += [(f'pfl_lam{lam}', dict(B, kind='pfl', lam=lam, sched=5)) for lam in (0.01, 0.1, 1.0)]
        return out
    if stage == 'variant':
        v = best['variant']; out = []
        for d in (0, 1, 5):
            B = dict(base, delay=d)
            out += [('pfl', dict(B, kind='pfl', lam=v['lam'], sched=v['sched'])),
                    ('pfl_no_replay', dict(B, kind='pfl', lam=v['lam'], sched=v['sched'], replay=False)),
                    ('pfl_pooled_adwin', dict(B, kind='pfl', lam=v['lam'], sched=v['sched'], classcond=False)),
                    ('proposed_sched', dict(B, kind='triggered', sched=v['sched']))]
        return out
    out = []
    for d in (0, 1, 5):
        B = dict(base, delay=d)
        out += [
            ('frozen', dict(B, kind='frozen')),
            ('fedavg_p1', dict(B, kind='periodic', period=1, replay=False, mu=0.0)),
            ('fedavg_p10', dict(B, kind='periodic', period=10, replay=False, mu=0.0)),
            ('local_adwin_replay', dict(B, kind='local')),
            ('proposed', dict(B, kind='triggered')),
            ('abl_pooled_adwin', dict(B, kind='triggered', classcond=False)),
            ('abl_no_replay', dict(B, kind='triggered', replay=False)),
            ('abl_no_heartbeat', dict(B, kind='triggered', heartbeat=None)),
            ('abl_warning_only', dict(B, kind='triggered', trigger='warning')),
            ('abl_error_or_warning', dict(B, kind='triggered', trigger='both')),
        ]
        P = best.get('matched_period', {}).get(str(d))
        if P:
            out += [('periodic_matched_replay', dict(B, kind='periodic', period=P)),
                    ('periodic_matched_noreplay', dict(B, kind='periodic', period=P, replay=False))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stream'); ap.add_argument('out')
    ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--stage', default='main'); ap.add_argument('--best', default=None)
    ap.add_argument('--pred-dir', default=None)
    a = ap.parse_args()
    best = json.load(open(a.best)) if a.best else {}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['name'], r['cfg'].get('delay'), r['seed']))
    seeds = range(3) if a.stage.startswith('tune') else range(a.seeds)
    tasks = []
    for name, cfg in configs(a.stage, best):
        for s in seeds:
            if (name, cfg.get('delay'), s) in done:
                continue
            sp = None
            if a.pred_dir and s == 0:
                os.makedirs(a.pred_dir, exist_ok=True)
                sp = os.path.join(a.pred_dir, f"{name}_d{cfg.get('delay')}_s{s}.npy")
            tasks.append((name, cfg, s, sp))
    # group by seed so each worker reuses its warm start
    tasks.sort(key=lambda t: t[2])
    print(f'{len(tasks)} runs', flush=True)
    with ProcessPoolExecutor(a.jobs, initializer=init, initargs=(a.stream,)) as ex, open(a.out, 'a') as f:
        for i, rec in enumerate(ex.map(job, tasks, chunksize=1)):
            f.write(json.dumps(rec, default=float) + '\n'); f.flush()
            e = rec['eval']
            print(f"[{i+1}/{len(tasks)}] {rec['name']} d={rec['cfg'].get('delay')} s={rec['seed']} "
                  f"F1={e['macro_f1']:.4f} bytes={e['byte_ratio']:.3f} ({rec['secs']}s)", flush=True)


if __name__ == '__main__':
    main()
