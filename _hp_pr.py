from src import env_guard
import sys, numpy as np, torch
from src.evaluation.real_events import load_events
from src.run_gate_check import forward_file, load_model
from src.run_seed_variance_probe import _mask_from
ev=load_events(); dev='cuda' if torch.cuda.is_available() else 'cpu'
for tag in sys.argv[1:]:
    model, apps, _ = load_model(f'results/{tag}.pt', dev)
    jh = apps.index('hotplate')
    print(f"[{tag}]  {'파일':8s}{'구간':9s}{'참ON':>6s}{'정밀도':>9s}{'재현율':>9s}{'F1':>9s}")
    for s in ('test_4','test_5','test_6'):
        d = forward_file(model, s, dev, stride=30)
        n=int(ev[s]['cycles'])
        truth_all=_mask_from(ev[s]['intervals']['hotplate'].get('on'), n, d['targets'])
        for lo,lab in ((0.,'<1300'),(1300.,'>=1300')):
            m=(d['p_observed']>=lo) if lo else (d['p_observed']<1300.)
            t=truth_all[m]; p=d['gate'][m,jh]>0.5
            tp=float((p&t).sum()); pr=tp/max(p.sum(),1); rc=tp/max(t.sum(),1)
            print(f"{'':10s}{s:8s}{lab:9s}{int(t.sum()):>6d}{pr:>9.3f}{rc:>9.3f}"
                  f"{2*pr*rc/max(pr+rc,1e-9):>9.3f}")
