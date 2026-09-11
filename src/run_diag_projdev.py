"""편차가 **판별 방향에 얼마나 실리나** — 최악치가 아니라 실제 사영 (13.84.31 정정)."""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.stats import trim_mean
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, ODD, FS, build_dict

D, names = build_dict()
def col(a):
    cs = [c for c, nm in enumerate(names) if nm.split(':')[0] == a]
    return D[:, cs].mean(1)
C = {a: col(a) for a in ('laptop_charger', 'beam_projector', 'minipc')}
EVEN = [1, 3, 5, 7, 9, 11, 13]                       # h2..h14 — **크기만** 쓴다(위상=플러그 방향)

def feat(v, even):
    f = [v[ODD].real, v[ODD].imag]
    if even:
        f.append(np.abs(v[EVEN]))
    return np.concatenate(f)

print('① 축을 넓히면 증폭률이 줄어드나  (||g_minipc||, W/A — 작을수록 좋다)')
for even in (False, True):
    tag = '홀수 16차원' if not even else '홀수16 + 짝수크기7 = 23차원'
    row = []
    for sub in (('laptop_charger', 'minipc'), ('beam_projector', 'minipc'),
                ('laptop_charger', 'beam_projector', 'minipc')):
        A = np.stack([feat(C[a], even) for a in sub], 1)
        g = np.linalg.pinv(A)[sub.index('minipc')]
        row.append('%s %7.1f' % ('+'.join(x[:5] for x in sub), np.linalg.norm(g)))
    print('   %-28s %s' % (tag, ' · '.join(row)))

print('\n② 실제 편차가 판별 방향에 얼마나 실리나 (최악치 ||g||·||δ|| 대 실제 |g·δ|)')
ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
def cmean(stem, want):
    r = load_nilm_npz('processed_data/composite_eval/%s.npz' % stem)
    H = np.asarray(r['harmonics_complex']); n = len(H)
    mk = {}
    for a in APPS:
        m = np.zeros(n, bool)
        for t0, t1 in ev[stem]['intervals'].get(a, {}).get('on', []):
            m[int(t0 * FS):int(min(t1 * FS, n))] = True
        mk[a] = m
    sel = np.ones(n, bool)
    for a in APPS:
        if a in ev[stem]['appliances_present']:
            sel &= (mk[a] if a in want else ~mk[a])
    idx = np.nonzero(sel)[0]; idx = idx[(idx > 5 * FS) & (idx < n - 5 * FS)]
    return trim_mean(H[idx], 0.2, axis=0) if len(idx) > 60 else None

for even in (False, True):
    A3 = np.stack([feat(C[a], even) for a in ('laptop_charger', 'beam_projector', 'minipc')], 1)
    g3 = np.linalg.pinv(A3)[2]
    A2 = np.stack([feat(C[a], even) for a in ('laptop_charger', 'minipc')], 1)
    g2 = np.linalg.pinv(A2)[1]
    print('   [%s]' % ('홀수 16차원' if not even else '짝수 크기 포함 23차원'))
    for stem, base, add, who, g in (('test_3', {'laptop_charger'}, {'laptop_charger', 'beam_projector'}, '프로젝터', g3),
                                    ('test_3', set(), {'laptop_charger'}, '충전기', g2),
                                    ('test_4', set(), {'laptop_charger', 'beam_projector'}, '충전기+프로젝터', g3)):
        a_, b_ = cmean(stem, base), cmean(stem, add)
        if a_ is None or b_ is None:
            continue
        dv = feat(b_ - a_, even)
        ref = feat(C['beam_projector'] if who == '프로젝터' else C['laptop_charger'], even)
        if who == '충전기+프로젝터':
            ref = feat(C['laptop_charger'] + C['beam_projector'], even)
        w = float(dv @ ref / (ref @ ref))
        d = dv - w * ref
        worst = np.linalg.norm(g) * np.linalg.norm(d)
        actual = abs(float(g @ d))
        print('      %-8s %-16s ||δ|| %6.1f mA · 최악 %6.1f W · **실제 %6.1f W** (%.0f%%)'
              % (stem, who, 1000 * np.linalg.norm(d) / np.sqrt(2), worst, actual,
                 100 * actual / max(worst, 1e-9)))
