"""자리별 고조파 지문 — 지문은 기기의 성질만이 아니다 (13.59)
=================================================================
⚠ **처방으로는 반증됐다** (13.59.6). 이 표로 2단계 `L_harm` 을 조건화하면 자리 D
  프로젝터가 **더 깎인다** (짝지은 6판 중 5판, 15.1W -> 3.2/5.8/3.2W, 참 44.6).
  격리 지문의 자리 보정은 격리->복합 전이 실패를 못 고치기 때문이다 —
  조합 홀드아웃에서 **현장 적합**이 오차를 −79% 줄이는데 자리 보정은 −7% 다
  (13.59.7). `--sig-site` 는 기본으로 꺼져 있고, 되풀이 방지를 위해 남긴다.

  **측정 도구로는 유효하다.** 아래 D/E 비 표가 13.58 의 기전을 실측으로 확정한다.

`harmonic_signatures` 는 격리 녹화를 **자리 구분 없이 뭉쳐** 중앙값을 낸다. 그런데
13.58 이 확정한 대로 SMPS 의 고차 전류는 그 콘센트의 계통 임피던스에 크게 반응한다 —
격리 녹화를 자리로 갈라 재면 자리 D(Z≈1.15~1.19Ω) 의 와트당 전류가 자리 E(0.42Ω) 대비

    기기        h9     h11    h13    h15      h13 위상차
    프로젝터    0.772  0.523  0.473  0.623    +113.3°
    충전기      0.809  0.549  0.300  0.301     +45.7°
    미니PC      0.944  0.817  0.647  0.388     +20.4°

**셋이 다 다르다.** 그래서 13.50 이 시도한 "자리별 회전 하나" 로는 못 고친다 —
보정은 기기마다 다르고 차수마다 다르다. h1 은 셋 다 1.07~1.08 로 같은데(전압비
229/216 = 1.06 의 몫이다) 고차에서만 갈린다.

왜 이것이 배분을 고치는가: 프로젝터의 격리 녹화 3개 중 2개가 자리 E 라 뭉친 지문은
2/3 가 E 다. 그것으로 자리 D 의 창을 예측하면 h13 을 **+71% 과대**로 낸다. `L_harm`
은 그 초과를 **프로젝터 전력을 깎아** 메우고, 그 40W 가 충전기로 간다 (13.58.3).

    옛:  pred_h = Σ_k Σ_s (게이트·혼합·p_states)_ks · sig[k,s,h]
    새:  pred_h = Σ_k Σ_s (게이트·혼합·p_states)_ks · sig[**자리**, k,s,h]

⚠ **자리(D/E)로 가른다. 세션(D1/D2/E1/E2)이 아니다.** 세션으로 가르면 구멍이 난다 —
  프로젝터는 D2 에만, 충전기는 D1 에만 자리 D 녹화가 있다. 자리 안에서 세션 간 전압은
  0.7V(D1 216.5 대 D2 215.8) 차라 h1 이 0.3% 움직인다.

⚠ 표본이 얇은 칸은 **뭉친 지문으로 되돌린다**. 자리 하나에서만 녹화된 기기(포트·오븐·
  드라이기 …)는 통째로 되돌아가므로 그 기기들의 동작은 이전과 **정확히 같다**.
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np

from src.model.net import MAX_STATES, harmonic_signatures, harmonic_signatures_by_state
from src.preprocessing.file_registry import site_of

#: 자리 축. 순서가 곧 `site_index` 다.
SITES: Tuple[str, ...] = ("D", "E")
#: 이 사이클 수보다 얇은 칸은 뭉친 지문으로 되돌린다. `harmonic_signatures_by_state`
#: 와 같은 문턱이다 — 얇은 표본을 억지로 따로 맞추면 그 칸이 잡음을 배운다.
MIN_CYCLES = 200


def site_index(stems: Sequence[str]) -> np.ndarray:
    """녹화 stem 목록 -> `SITES` 의 색인 (N,) int64. 모르는 자리는 −1."""
    look = {s: i for i, s in enumerate(SITES)}
    return np.asarray([look.get(site_of(s), -1) for s in stems], dtype=np.int64)


def _per_site_cells(pool, appliances: Sequence[str], by_state: bool
                    ) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
    """(자리, 기기, 상태) -> (사이클 고조파, 전력). `by_state` 가 거짓이면 상태는 −1.

    ⚠ **문턱을 늘리는 쪽과 맞춰야 한다.** `harmonic_signatures` 는 통전(p90 의 절반)
    문턱이고 `harmonic_signatures_by_state` 는 **1.0W** 다 — 상태 축은 저전력 상태를
    담는 것이 목적이라 통전 문턱을 쓰면 그 상태가 통째로 빈다. 처음에 여기서 양쪽에
    통전 문턱을 써서 프로젝터 상태 1 을 지웠고, 자리 지문이 그 자리를 덮어
    와트당 h13 이 2.5~3.8 mA/W 어긋났다 (13.59.5).
    """
    cells: Dict[Tuple[int, int, int], Tuple[List, List]] = {}
    look = {s: i for i, s in enumerate(SITES)}
    for j, app in enumerate(appliances):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            continue
        thr = 1.0 if by_state else max(0.5 * pool.get_steady_power_w(app), 1.0)
        for a in acts:
            si = look.get(site_of(a.source_file), -1)
            if si < 0:
                continue
            m0 = a.target_power_w > thr
            if not m0.any():
                continue
            if not by_state:
                d = cells.setdefault((si, j, -1), ([], []))
                d[0].append(a.net_harmonics_complex[m0])
                d[1].append(a.target_power_w[m0])
                continue
            st = getattr(a, "state_id", None)
            if st is None:
                continue
            st = np.asarray(st)
            for s in np.unique(st[m0]).astype(int):
                if not 0 < s < MAX_STATES:
                    continue
                m = m0 & (st == s)
                if m.any():
                    d = cells.setdefault((si, j, int(s)), ([], []))
                    d[0].append(a.net_harmonics_complex[m])
                    d[1].append(a.target_power_w[m])
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in cells.items()}


def _median_per_w(c: np.ndarray, p: np.ndarray, n_harm: int) -> np.ndarray:
    per_w = c[:, :n_harm] / np.maximum(p[:, None], 1e-6)
    out = np.zeros((n_harm, 2), dtype=np.float32)
    out[:, 0] = np.median(np.real(per_w), axis=0)
    out[:, 1] = np.median(np.imag(per_w), axis=0)
    return out


def site_signatures(pool, appliances: Sequence[str], n_harm: int = 15,
                    min_cycles: int = MIN_CYCLES) -> Tuple[np.ndarray, np.ndarray]:
    """(자리, 기기) 와트당 지문 (Ns, K, H, 2) 와 따로 맞췄는지 (Ns, K) bool.

    얇은 칸은 `harmonic_signatures` 의 뭉친 값이다 — 즉 **자리 녹화가 없는 기기는
    이전과 완전히 같다**.
    """
    base = harmonic_signatures(pool, appliances, n_harm)                 # (K,H,2)
    sig = np.repeat(base[None], len(SITES), axis=0)                      # (Ns,K,H,2)
    used = np.zeros((len(SITES), len(appliances)), dtype=bool)
    for (si, j, _), (c, p) in _per_site_cells(pool, appliances, False).items():
        if len(c) < min_cycles:
            continue
        sig[si, j] = _median_per_w(c, p, n_harm)
        used[si, j] = True
    return sig.astype(np.float32), used


def site_signatures_by_state(pool, appliances: Sequence[str], n_harm: int = 15,
                             min_cycles: int = MIN_CYCLES
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """(자리, 기기, 상태) 지문 (Ns, K, S, H, 2) 와 **직접 잰** 칸 (Ns, K, S) bool.

    되돌림이 상태 축을 **절대 지우지 않는다**:

        ① 자리x상태 표본이 두꺼우면 그것을 그대로 쓴다
        ② 아니면 뭉친 상태별 지문에 **자리 비**를 곱한다
               sig[z,k,s,h] = sig_state[k,s,h] · (sig_site[z,k,h] / sig_pool[k,h])
        ③ 자리 지문이 없는 기기는 뭉친 상태별 지문 그대로다

    ②가 요점이다. 처음에는 자리 지문으로 상태 축을 **덮었는데**, 그러면 표본이 얇은
    저전력 상태(프로젝터 상태 1)가 고전력 지문을 뒤집어쓴다 — 13.11 이 드라이기에서
    확인한 그 결함이다. 비로 걸면 상태 모양은 그대로 두고 자리 효과만 얹는다.
    """
    state_base, _ = harmonic_signatures_by_state(pool, appliances, n_harm)   # (K,S,H,2)
    site_base, site_used = site_signatures(pool, appliances, n_harm, min_cycles)
    pooled = harmonic_signatures(pool, appliances, n_harm)                   # (K,H,2)
    sig = np.repeat(state_base[None], len(SITES), axis=0)                    # (Ns,K,S,H,2)
    # ② 자리 비 — 복소로 곱한다. 분모가 0 에 가까운 차수는 비를 1 로 둔다.
    pc = pooled[..., 0] + 1j * pooled[..., 1]                                # (K,H)
    floor = 1e-3 * np.abs(pc[:, :1]).clip(min=1e-12)                         # h1 의 1/1000
    for si in range(len(SITES)):
        sc = site_base[si, ..., 0] + 1j * site_base[si, ..., 1]              # (K,H)
        r = np.where(np.abs(pc) > floor, sc / np.where(np.abs(pc) > floor, pc, 1.0), 1.0)
        for j in range(len(appliances)):
            if not site_used[si, j]:
                continue
            b = state_base[j, ..., 0] + 1j * state_base[j, ..., 1]           # (S,H)
            nb = b * r[j][None]
            sig[si, j, ..., 0], sig[si, j, ..., 1] = nb.real, nb.imag
    # ① 자리x상태 표본이 두꺼우면 직접 잰 값으로 덮는다.
    used = np.zeros((len(SITES), len(appliances), MAX_STATES), dtype=bool)
    for (si, j, s), (c, p) in _per_site_cells(pool, appliances, True).items():
        if len(c) < min_cycles:
            continue
        sig[si, j, s] = _median_per_w(c, p, n_harm)
        used[si, j, s] = True
    return sig.astype(np.float32), used


def signature_bank(pool, appliances: Sequence[str], n_harm: int = 15,
                   min_cycles: int = MIN_CYCLES
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """손실이 창별로 고를 지문 묶음.

        sig      (Ns+1, K, H, 2)
        sig_st   (Ns+1, K, S, H, 2)
        used     (Ns, K)   따로 맞춘 칸

    **마지막 줄은 뭉친 지문**이다. 자리를 모르는 창을 거기로 보내면 그 창은
    이전과 정확히 같이 돈다 (`bank_index` 가 −1 을 그 줄로 보낸다).
    """
    ss, used = site_signatures(pool, appliances, n_harm, min_cycles)
    st, _ = site_signatures_by_state(pool, appliances, n_harm, min_cycles)
    base = harmonic_signatures(pool, appliances, n_harm)
    base_st, _ = harmonic_signatures_by_state(pool, appliances, n_harm)
    return (np.concatenate([ss, base[None]], 0).astype(np.float32),
            np.concatenate([st, base_st[None]], 0).astype(np.float32), used)


def bank_index(stems: Sequence[str]) -> np.ndarray:
    """녹화 stem 목록 -> `signature_bank` 의 줄 번호 (N,) int64. 모르는 자리는 뭉친 줄."""
    idx = site_index(stems)
    return np.where(idx < 0, len(SITES), idx).astype(np.int64)


def coverage_table(pool, appliances: Sequence[str], n_harm: int = 15) -> str:
    """어느 칸을 따로 맞췄는지 사람이 읽는 표. 규칙 14 — 잰 것과 되돌린 것을 갈라 찍는다."""
    _, used = site_signatures(pool, appliances, n_harm)
    _, used_s = site_signatures_by_state(pool, appliances, n_harm)
    lines = [f"  {'기기':20s}" + "".join(f"{'자리 ' + s:>12s}" for s in SITES)
             + f"{'상태별 칸':>12s}"]
    for j, app in enumerate(appliances):
        cells = "".join(f"{('따로' if used[si, j] else '뭉침'):>12s}"
                        for si in range(len(SITES)))
        lines.append(f"  {app:20s}{cells}{int(used_s[:, j].sum()):12d}")
    return "\n".join(lines)
