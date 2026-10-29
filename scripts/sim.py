#!/usr/bin/env python3
"""Prequential simulation of delayed-label, drift-triggered federated intrusion detection.

Every record is scored by the model deployed at its round before any label from that round is
used. Labels of round r are released at the end of round r + delay. All feature transforms are
fitted on the warm-up segment only.
"""
import argparse, json, os, time, zlib, pickle
import numpy as np, pandas as pd
from river.drift import ADWIN

NUM = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts',
       'dst_ip_bytes', 'dns_qclass', 'dns_qtype', 'dns_rcode', 'http_trans_depth',
       'http_request_body_len', 'http_response_body_len', 'http_status_code']
CAT = ['proto', 'service', 'conn_state', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_version',
       'ssl_cipher', 'ssl_resumed', 'ssl_established', 'http_method', 'http_version',
       'http_orig_mime_types', 'http_resp_mime_types', 'weird_name', 'weird_notice']
MON = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']

# ----------------------------------------------------------------------------- data

def load_stream(path, K, n_round, frac=(0.2, 0.2)):
    cache = f'{path}.K{K}_n{n_round}.cache.pkl'
    if os.path.exists(cache):
        D = pickle.load(open(cache, 'rb'))
        D['X'] = np.load(cache + '.X.npy', mmap_mode='r')
        return D
    D = _build_stream(path, K, n_round, frac)
    np.save(cache + '.X.npy', D['X'])
    pickle.dump({k: v for k, v in D.items() if k != 'X'}, open(cache, 'wb'))
    D['X'] = np.load(cache + '.X.npy', mmap_mode='r')
    return D


def _build_stream(path, K, n_round, frac):
    df = pd.read_pickle(path)
    n = len(df)
    rnd = np.arange(n) // n_round
    R = int(rnd.max()) + 1
    R_w = int(round(frac[0] * R)); R_t = int(round((frac[0] + frac[1]) * R))
    warm = rnd < R_w
    num = np.stack([pd.to_numeric(df[c], errors='coerce').fillna(0).clip(lower=0).values for c in NUM], 1)
    num = np.log1p(num.astype(np.float64))
    mu = num[warm].mean(0); sd = num[warm].std(0); sd[sd < 1e-6] = 1.0
    cols = [((num - mu) / sd).astype(np.float32)]
    names = list(NUM)
    for c in CAT:
        v = df[c].astype(str).values
        vocab = pd.Series(v[warm]).value_counts().index[:30].tolist()
        m = {t: i for i, t in enumerate(vocab)}
        idx = np.array([m.get(t, len(vocab)) for t in v])
        oh = np.zeros((n, len(vocab) + 1), dtype=np.float32); oh[np.arange(n), idx] = 1
        cols.append(oh); names += [f'{c}={t}' for t in vocab] + [f'{c}=OTHER']
    X = np.concatenate(cols, 1).astype(np.float32)
    keep = X[warm].std(0) > 0          # drop columns constant in warm-up (causal)
    X = X[:, keep]; names = [nm for nm, k in zip(names, keep) if k]
    y = df['label'].values.astype(np.int8)
    types = df['type'].astype(str).values
    client = np.array([zlib.crc32(s.encode()) % K for s in df['dst_ip'].astype(str).values])
    mon = num[:, [NUM.index(c) for c in MON]]
    return dict(X=X, y=y, types=types, client=client, rnd=rnd, R=R, R_w=R_w, R_t=R_t, mon=mon,
                names=names, ts=df['ts'].values)

# ----------------------------------------------------------------------------- model

class MLP:
    def __init__(self, d_in, h=(128, 64)):
        self.shapes = [(d_in, h[0]), (h[0],), (h[0], h[1]), (h[1],), (h[1], 1), (1,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.P = sum(self.sizes)

    def init(self, rng):
        th = []
        for s in self.shapes:
            if len(s) == 2:
                th.append(rng.normal(0, np.sqrt(2.0 / s[0]), s).ravel())
            else:
                th.append(np.zeros(s))
        return np.concatenate(th)

    def unpack(self, th):
        out, o = [], 0
        for s, z in zip(self.shapes, self.sizes):
            out.append(th[o:o + z].reshape(s)); o += z
        return out

    def predict(self, th, X):
        W1, b1, W2, b2, W3, b3 = self.unpack(th)
        h1 = np.maximum(X @ W1 + b1, 0); h2 = np.maximum(h1 @ W2 + b2, 0)
        z = (h2 @ W3 + b3).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def grad(self, th, X, y, w):
        """Gradient of sum_i w_i * BCE_i."""
        W1, b1, W2, b2, W3, b3 = self.unpack(th)
        a1 = X @ W1 + b1; h1 = np.maximum(a1, 0)
        a2 = h1 @ W2 + b2; h2 = np.maximum(a2, 0)
        z = (h2 @ W3 + b3).ravel(); p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        dz = ((p - y) * w)[:, None]
        gW3 = h2.T @ dz; gb3 = dz.sum(0)
        d2 = (dz @ W3.T) * (a2 > 0)
        gW2 = h1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W2.T) * (a1 > 0)
        gW1 = X.T @ d1; gb1 = d1.sum(0)
        return np.concatenate([g.ravel() for g in (gW1, gb1, gW2, gb2, gW3, gb3)])


def local_train(net, th0, Xn, yn, Xb, yb, beta, mu, lr, rng, batch=64, epochs=1, mom=0.9, anchor=None):
    """One pass over new data; each minibatch is paired with an equal-size replay minibatch.
    The proximal term pulls towards `anchor` (default: the starting point th0)."""
    th = th0.copy(); v = np.zeros_like(th)
    anchor = th0 if anchor is None else anchor
    n = len(yn); steps = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            ib = perm[s:s + batch]
            g = net.grad(th, Xn[ib], yn[ib], np.full(len(ib), 1.0 / len(ib)))
            if beta > 0 and Xb is not None and len(yb) > 0:
                jb = rng.integers(0, len(yb), len(ib))
                g += beta * net.grad(th, Xb[jb], yb[jb], np.full(len(jb), 1.0 / len(jb)))
            if mu > 0:
                g += mu * (th - anchor)
            v = mom * v + g; th -= lr * v; steps += 1
    return th, steps

# ----------------------------------------------------------------------------- monitors

class Reservoir:
    """Class-stratified reservoir of verified example indices."""
    def __init__(self, cap, rng):
        self.cap = cap // 2; self.buf = {0: [], 1: []}; self.seen = {0: 0, 1: 0}; self.rng = rng

    def add(self, idx, y):
        for i, c in zip(idx, y):
            c = int(c); self.seen[c] += 1; b = self.buf[c]
            if len(b) < self.cap:
                b.append(i)
            else:
                j = self.rng.integers(0, self.seen[c])
                if j < self.cap:
                    b[j] = i

    def indices(self):
        return np.array(self.buf[0] + self.buf[1], dtype=np.int64)


class ErrorMonitor:
    """ADWIN on released Brier errors; pooled or one detector per true class.

    Returns True when a detector fires and its mean error went up."""
    def __init__(self, delta, classcond, cap_per_round, rng):
        self.classcond = classcond; self.cap = cap_per_round; self.rng = rng
        self.det = {c: ADWIN(delta=delta) for c in ((0, 1) if classcond else ('all',))}

    def update(self, err, y):
        fired = False
        groups = [(c, err[y == c]) for c in (0, 1)] if self.classcond else [('all', err)]
        for c, e in groups:
            if len(e) > self.cap:
                e = self.rng.choice(e, self.cap, replace=False)
            d = self.det[c]
            for v in e:
                before = d.estimation
                d.update(float(v))
                if d.drift_detected and d.estimation > before:
                    fired = True
        return fired


def js_div(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log(a[nz] / b[nz]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class FeatureMonitor:
    """Max over monitored features of JS divergence between the round histogram and warm-up."""
    def __init__(self, mon_warm, bins=10):
        self.edges = [np.unique(np.quantile(mon_warm[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
                      for j in range(mon_warm.shape[1])]
        self.ref = [self.hist(mon_warm[:, j], j) for j in range(mon_warm.shape[1])]

    def hist(self, x, j):
        h = np.bincount(np.searchsorted(self.edges[j], x, side='right'),
                        minlength=len(self.edges[j]) + 1).astype(float) + 0.5
        return h / h.sum()

    def score(self, M):
        return max(js_div(self.hist(M[:, j], j), self.ref[j]) for j in range(M.shape[1]))

# ----------------------------------------------------------------------------- simulation

DEFAULT = dict(kind='triggered', lam=0.1, sched=1, delay=1, replay=True, beta=1.0, buf=512, mu=0.01, lr=0.01,
               delta=0.002, classcond=True, trigger='error', heartbeat=10, cooldown=2,
               period=None, clip=5.0, wcap=2000, new_cap=4000, err_cap=250, h_bytes=64, b_bytes=4,
               warm_epochs=5, js_q=0.99)


def warm_start(net, D, seed, epochs, exclude=None):
    rng = np.random.default_rng(1000 + seed)
    th = net.init(rng)
    K = D['client'].max() + 1
    w = D['rnd'] < D['R_w']
    per = [np.flatnonzero(w & (D['client'] == k)) for k in range(K)]
    for _ in range(epochs):
        ups, ns = [], []
        for k in range(K):
            if len(per[k]) == 0 or k == exclude:
                continue
            t, _ = local_train(net, th, D['X'][per[k]], D['y'][per[k]], None, None, 0, 0, 0.01, rng)
            ups.append(t - th); ns.append(len(per[k]))
        ns = np.array(ns, float); th = th + sum(a * u for a, u in zip(ns / ns.sum(), ups))
    return th


def simulate(D, cfg, seed, th0=None, log_every=0):
    c = dict(DEFAULT); c.update(cfg)
    rng = np.random.default_rng(seed)
    X, y, cl, rnd = D['X'], D['y'], D['client'], D['rnd']
    K = int(cl.max()) + 1; R, R_w = D['R'], D['R_w']
    net = MLP(X.shape[1])
    if th0 is None:
        th0 = warm_start(net, D, seed, c['warm_epochs'])
    order = np.argsort(rnd, kind='stable')
    bounds = np.searchsorted(rnd[order], np.arange(R + 1))
    by_rc = {}
    for r in range(R):
        ir = order[bounds[r]:bounds[r + 1]]
        for k in range(K):
            by_rc[(r, k)] = ir[cl[ir] == k]
    local = c['kind'] == 'local'
    personal = c['kind'] in ('local', 'pfl')
    models = [th0.copy() for _ in range(K)] if personal else None
    queue = []
    glob = th0.copy()
    # replay buffers start from verified warm-up data
    bufs = [Reservoir(c['buf'], np.random.default_rng(seed * 100 + k)) for k in range(K)]
    J = c.get('join'); jr = D['R_t']
    for k in range(K):
        if k == J:
            continue
        idx = np.concatenate([by_rc[(r, k)] for r in range(R_w)])
        bufs[k].add(idx, y[idx])
    mons = [ErrorMonitor(c['delta'], c['classcond'], c['err_cap'], np.random.default_rng(seed * 7 + k))
            for k in range(K)]
    fmons, thr = [], []
    for k in range(K):
        idx = np.concatenate([by_rc[(r, k)] for r in range(R_w)])
        fm = FeatureMonitor(D['mon'][idx]); fmons.append(fm)
        sc = [fm.score(D['mon'][by_rc[(r, k)]]) for r in range(R_w) if len(by_rc[(r, k)]) >= 50]
        thr.append(np.quantile(sc, c['js_q']) if sc else np.inf)
    P = np.full(len(y), np.nan); ver = np.full(len(y), -1)
    pending = [[] for _ in range(K)]
    since = np.zeros(K, int); cool = np.zeros(K, int)
    version = 0
    log = dict(up=0, agg=0, health=0, steps=0, alarms=[], warnings=[], updates=[], clipped=0,
               norms=[], agg_rounds=[], health_per_round=0 if c['kind'] in ('frozen', 'periodic', 'local') else K,
               local=local)
    for r in range(R_w, R):
        absent = J is not None and r < jr
        if J is not None and r == jr and personal:
            models[J] = (th0 if local else glob).copy()   # a new gateway joins
        # 1) score with deployed models
        for k in range(K):
            if absent and k == J:
                continue
            ik = by_rc[(r, k)]
            if len(ik):
                P[ik] = net.predict(models[k] if personal else glob, X[ik]); ver[ik] = version
        # 2) release labels of round r - delay
        rr = r - c['delay']
        alarm = np.zeros(K, bool); warn = np.zeros(K, bool)
        for k in range(K):
            if absent and k == J:
                continue
            ik = by_rc[(r, k)]
            if len(ik) >= 50 and fmons[k].score(D['mon'][ik]) > thr[k]:
                warn[k] = True; log['warnings'].append((r, k))
            if rr >= R_w:
                ir = by_rc[(rr, k)]
                if len(ir):
                    pending[k].append(ir)
                    bufs[k].add(ir, y[ir])
                    if c['kind'] in ('triggered', 'local', 'pfl'):
                        if mons[k].update((P[ir] - y[ir]) ** 2, y[ir]):
                            alarm[k] = True; log['alarms'].append((r, k))
        # 3) decide who trains
        since += 1
        if c['kind'] == 'frozen':
            cohort = []
        elif c['kind'] == 'periodic':
            cohort = list(range(K)) if (r - R_w + 1) % c['period'] == 0 else []
        else:
            sig = {'error': alarm, 'warning': warn, 'both': alarm | warn}[c['trigger']]
            hb = since >= c['heartbeat'] if c['heartbeat'] else np.zeros(K, bool)
            req = (sig | hb) & (cool == 0)
            cohort = [k for k in np.flatnonzero(req) if not (absent and k == J)]
            cool = np.maximum(cool - 1, 0)
            log['health'] += 0 if local else K
        # 4) local training
        for k in cohort:
            if not pending[k]:
                continue
            idx = np.concatenate(pending[k])[-c['new_cap']:]
            bidx = bufs[k].indices() if c['replay'] else None
            Xb = X[bidx] if bidx is not None else None
            yb = y[bidx] if bidx is not None else None
            beta = c['beta'] if c['replay'] else 0.0
            if c['kind'] == 'pfl':
                # personalised deployed model, kept close to the current global model (Ditto-style)
                models[k], st = local_train(net, models[k], X[idx], y[idx], Xb, yb, beta, c['lam'],
                                            c['lr'], rng, anchor=glob)
                log['steps'] += st
            base = models[k] if local else glob
            th, st = local_train(net, base, X[idx], y[idx], Xb, yb, beta, c['mu'], c['lr'], rng)
            log['steps'] += st
            pending[k] = []; since[k] = 0; cool[k] = c['cooldown']
            delta = th - base; nrm = np.linalg.norm(delta); log['norms'].append(nrm)
            if nrm > c['clip']:
                delta *= c['clip'] / nrm; log['clipped'] += 1
            if local:
                models[k] = base + delta
            else:
                queue.append((delta, min(len(idx), c['wcap'])))
            log['up'] += 0 if local else 1
            log['updates'].append((r, int(k)))
        # 5) at each scheduling boundary, aggregate the queued common-version updates and
        #    broadcast the new global model to all clients
        if queue and (r - R_w + 1) % c['sched'] == 0:
            ws = np.array([w for _, w in queue], float); ws /= ws.sum()
            glob = glob + sum(a * u for a, (u, _) in zip(ws, queue)); queue = []
            version += 1; log['agg'] += 1; log['agg_rounds'].append(r)
    return P, ver, log, net.P


# ----------------------------------------------------------------------------- metrics

def metrics(D, P, log, d_theta, cfg, seg='eval'):
    from sklearn.metrics import f1_score, average_precision_score, recall_score
    c = dict(DEFAULT); c.update(cfg)
    rnd, y, types, cl = D['rnd'], D['y'], D['types'], D['client']
    lo, hi = (D['R_t'], D['R']) if seg == 'eval' else (D['R_w'], D['R_t'])
    m = (rnd >= lo) & (rnd < hi)
    p = P[m]; yy = y[m]; yh = (p >= 0.5).astype(int); tt = types[m]
    out = dict(macro_f1=f1_score(yy, yh, average='macro'),
               attack_recall=recall_score(yy, yh, zero_division=0),
               fpr=float(((yh == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum())),
               pr_auc=average_precision_score(yy, p) if yy.min() != yy.max() else np.nan,
               brier=float(np.mean((p - yy) ** 2)))
    for t in np.unique(tt):
        if t != 'normal':
            out[f'recall_{t}'] = float(yh[tt == t].mean())
    K = int(cl.max()) + 1
    cf = []
    for k in range(K):
        mk = cl[m] == k
        if mk.sum() > 100 and len(np.unique(yy[mk])) == 2:
            cf.append(f1_score(yy[mk], yh[mk], average='macro'))
    out['min_client_f1'] = float(np.min(cf)) if cf else np.nan
    # communication inside this segment, corrected byte model:
    # uplink per participating client + broadcast of every new global model to all K clients
    # + one health message per client and round (triggered methods only)
    inseg = lambda rr: lo <= rr < hi
    nupd = sum(1 for (rr, k) in log['updates'] if inseg(rr))
    up = 0 if log['local'] else nupd
    agg = sum(1 for rr in log['agg_rounds'] if inseg(rr))
    b = c['b_bytes']; nr = hi - lo
    bytes_ = b * d_theta * (up + K * agg) + c['h_bytes'] * log['health_per_round'] * nr
    full = 2 * b * d_theta * K * nr
    out.update(uplinks=up, aggregations=agg, bytes=int(bytes_), byte_ratio=bytes_ / full,
               client_updates=nupd, rounds=nr,
               alarms=sum(1 for (rr, k) in log['alarms'] if inseg(rr)),
               warnings=sum(1 for (rr, k) in log['warnings'] if inseg(rr)),
               clip_frac=log['clipped'] / max(1, len(log['norms'])), sgd_steps=log['steps'])
    return out


def per_round(D, P):
    from sklearn.metrics import f1_score
    rows = []
    for r in range(D['R_w'], D['R']):
        m = D['rnd'] == r
        yy = D['y'][m]; yh = (P[m] >= 0.5).astype(int)
        row = dict(round=r, ts=int(D['ts'][m].min()),
                   macro_f1=f1_score(yy, yh, average='macro') if len(np.unique(yy)) == 2 else np.nan,
                   acc=float((yy == yh).mean()))
        for t in np.unique(D['types'][m]):
            mt = D['types'][m] == t
            row[f'n_{t}'] = int(mt.sum()); row[f'hit_{t}'] = float((yh[mt] == yy[mt]).mean())
        rows.append(row)
    return pd.DataFrame(rows)
