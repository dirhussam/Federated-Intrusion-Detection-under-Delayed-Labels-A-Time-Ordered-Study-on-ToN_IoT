import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import *
t=time.time(); D=load_stream(sys.argv[1], K=8, n_round=8000); print('load',time.time()-t, D['X'].shape, D['R'],D['R_w'],D['R_t'])
import collections
for k in range(8):
    m=D['client']==k; print(k, m.sum(), collections.Counter(D['types'][m]).most_common(4))
net=MLP(D['X'].shape[1]); t=time.time(); th0=warm_start(net,D,0,5); print('warm',time.time()-t, net.P)
for cfg in [dict(kind='frozen'), dict(kind='periodic',period=1,replay=False,mu=0), dict(kind='triggered'), dict(kind='local')]:
    t=time.time(); P,ver,log,dp=simulate(D,cfg,0,th0); m=metrics(D,P,log,dp,cfg)
    print(cfg, round(time.time()-t,1)); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items()})
    if log['norms']: print('norm pct', np.percentile(log['norms'],[10,50,90]))
