"""
Phase 1 — 특징 기반 baseline (분해 과제)
=========================================
설계 문서 4.1절: **CNN 부터 시작하지 말 것.** 비교 대상이 없으면 CNN 성능이 좋은지
나쁜지 판단할 수 없다. 근거 둘 — (1) 이 데이터로 수작업 특징 + 소형 MLP 가 9종
판별 98.7% 를 냈고, (2) 거의 같은 하드웨어의 선행 연구에서 RF > SVM > CNN 이었다.

**단 0.3절의 98.7% 는 '한 기기만 든 창' 을 분류한 것이다.** 우리 과제는 분해다.
그래서 여기 baseline 도 분해 과제로 세운다 — 같은 창에서 9종 전력을 동시에 회귀.

    기기별 회귀기 9개   HistGradientBoostingRegressor  -> 전력 (W)
    기기별 분류기 9개   HistGradientBoostingClassifier -> on/off

LightGBM 대신 sklearn 을 쓴다. 사실상 같은 히스토그램 기반 GBM 이고 의존성이
늘지 않는다.

[학습/평가 분리]
학습 창은 `SegmentPool(time_split="train")` — 각 녹화의 앞 80% — 에서만 만든다.
평가는 `processed_data/holdout` 의 얼린 셋(뒤 20%)으로 한다. 12.3절 참조.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import numpy as np

from src.baseline.features import extract, feature_names

# 워커마다 한 번씩 만들어 재사용한다 (풀 적재가 2.5초라 매번 만들면 안 된다)
_GEN = None
_DEGRADE = 0        # >0 이면 마지막 이 사이클만 60Hz 로 남기고 앞은 다운샘플


def degrade_to_multiscale(x: np.ndarray, fine_cycles: int, coarse_hz: float = 2.0) -> np.ndarray:
    """창의 뒤 `fine_cycles` 만 60Hz 로 남기고 그 앞은 `coarse_hz` 로 낮춘다.

    설계 문서 1.1절의 2갈래(세밀 10초 @60Hz + 광역 @1~2Hz)가 **실제로 전 구간
    60Hz 만큼의 정보를 주는지** 확인하려고 만든 것이다. 다운샘플한 배열에도
    같은 특징 추출기를 돌려, 정보량만 줄이고 나머지는 동일하게 둔다.

    블록 평균을 그대로 유지(hold)하므로, 2Hz 로 샘플한 뒤 계단 보간한 신호와 같다.
    """
    b, c, w = x.shape
    n_coarse = w - fine_cycles
    if n_coarse <= 0:
        return x
    out = x.copy()
    blk = max(1, int(round(60.0 / coarse_hz)))
    k = n_coarse // blk
    if k:
        head = out[:, :, :k * blk].reshape(b, c, k, blk)
        out[:, :, :k * blk] = head.mean(axis=3, keepdims=True).repeat(blk, axis=3).reshape(b, c, k * blk)
    if k * blk < n_coarse:
        tail = out[:, :, k * blk:n_coarse]
        out[:, :, k * blk:n_coarse] = tail.mean(axis=2, keepdims=True)
    return out


def _init_worker(npz_dir: str, window_cycles: int, time_split: str, seed_base: int,
                 target_lookahead_cycles: int, degrade_fine_cycles: int = 0) -> None:
    global _GEN, _DEGRADE
    _DEGRADE = int(degrade_fine_cycles)
    import os
    from src.synthesis.dataset import NILMBatchGenerator
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer

    np.random.seed((seed_base + os.getpid()) % (2 ** 31))
    pool = SegmentPool(npz_dir=npz_dir, time_split=time_split)
    _GEN = NILMBatchGenerator(
        segment_pool=pool, window_size_cycles=window_cycles,
        synthesizer=LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False),
        compute_gt_harmonics=False,
        target_lookahead_cycles=target_lookahead_cycles,
    )


def _make_chunk(n: int) -> Tuple[np.ndarray, ...]:
    """창 n 개를 만들어 **특징만** 돌려준다. 원시 창을 반환하면 IPC 가 병목이 된다."""
    g = _GEN
    xs = np.empty((n, 33, g.window_size), np.float32)
    k = len(g.appliance_list)
    yp = np.empty((n, k), np.float32)
    yo = np.empty((n, k), np.int8)
    for i in range(n):
        smp, _ = g._synthesize_window()
        t = g._format_targets(smp)
        xs[i] = g._format_inputs(smp)
        yp[i], yo[i] = t["y_power"], t["y_on"]
    if _DEGRADE > 0:
        xs = degrade_to_multiscale(xs, _DEGRADE)
    return extract(xs, target_index=g.target_index), yp, yo


def build_training_set(
    n_windows: int = 150_000,
    npz_dir: str = "processed_data/npz",
    window_cycles: int = 600,
    time_split: str = "train",
    seed: int = 0,
    n_workers: int = 11,
    chunk: int = 500,
    target_lookahead_cycles: int = 60,
    degrade_fine_cycles: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """학습용 (특징, 전력, on/off) 를 만든다."""
    import multiprocessing as mp

    from src.synthesis.segment_pool import SegmentPool
    apps = SegmentPool(npz_dir=npz_dir, time_split=time_split).get_appliance_types()

    tasks = [chunk] * (n_windows // chunk)
    if n_windows % chunk:
        tasks.append(n_windows % chunk)

    t0 = time.time()
    args = (npz_dir, window_cycles, time_split, seed, target_lookahead_cycles,
            degrade_fine_cycles)
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=args) as pool:
            parts = []
            for i, r in enumerate(pool.imap_unordered(_make_chunk, tasks), 1):
                parts.append(r)
                if i % 50 == 0:
                    done = sum(len(p[0]) for p in parts)
                    print(f"  {done:>7,}/{n_windows:,}  ({done/(time.time()-t0):,.0f} win/s)", flush=True)
    else:
        _init_worker(*args)
        parts = [_make_chunk(t) for t in tasks]

    F = np.concatenate([p[0] for p in parts])
    yp = np.concatenate([p[1] for p in parts])
    yo = np.concatenate([p[2] for p in parts])
    print(f"  완료 {len(F):,}창 / {time.time()-t0:.1f}s ({len(F)/(time.time()-t0):,.0f} win/s)")
    return F, yp, yo, apps


@dataclass
class BaselineModel:
    """기기별 GBM 묶음."""
    appliances: List[str]
    regressors: list
    classifiers: list
    feature_names: List[str]
    config: dict

    def predict(self, F: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """(전력 W, on 확률) 을 낸다. 전력은 on 판정으로 게이팅한다.

        회귀기는 켜진 창에서만 학습했으므로 꺼진 창에서의 출력은 외삽이다.
        반드시 게이팅해야 한다. 2.4절과 같은 구조이며, 게이팅이 없으면
        꺼진 기기에도 전력이 새어 들어 4.3절의 FA 가 올라간다.
        """
        p = np.stack([r.predict(F) for r in self.regressors], axis=1)
        on = np.stack([c.predict_proba(F)[:, 1] for c in self.classifiers], axis=1)
        return np.maximum(p, 0.0) * (on > threshold), on

    def save(self, path) -> None:
        import pickle
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "BaselineModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def train(
    F: np.ndarray, y_power: np.ndarray, y_on: np.ndarray, appliances: List[str],
    max_iter: int = 400, learning_rate: float = 0.06, max_leaf_nodes: int = 63,
    early_stopping: bool = True, random_state: int = 0, verbose: bool = True,
) -> BaselineModel:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    regs, clfs = [], []
    for j, app in enumerate(appliances):
        t0 = time.time()
        on = y_on[:, j].astype(bool)

        # 회귀기는 **켜진 창에서만** 학습한다.
        #
        # 전체 창으로 학습하면 안 된다. 타깃의 85% 가 0 이라 MAE 최적 상수가
        # 중앙값(=0)이고, 회귀기가 곧장 "전부 0" 으로 수렴해 조기종료한다.
        # (실제로 그렇게 되어 회귀 20회 만에 멈추고 RE 가 전부 1.000 이 나왔다)
        #
        # 켜진 창만 쓰면 타깃이 정격 근처에 몰려 정상적으로 학습되고,
        # 꺼진 창은 분류기가 게이팅으로 처리한다. 이 분해는 2.4절의
        # `P̂_i = sigmoid(on_i)·p_i` 와 같은 구조라 CNN 과 비교도 공정하다.
        r = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=max_iter, learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes, early_stopping=early_stopping,
            validation_fraction=0.1, n_iter_no_change=20, random_state=random_state)
        if int(on.sum()) >= 50:
            r.fit(F[on], y_power[on, j])
        else:                                    # 양성이 거의 없으면 학습을 포기한다
            r.fit(F[:100], y_power[:100, j])

        c = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            early_stopping=early_stopping, validation_fraction=0.1, n_iter_no_change=20,
            random_state=random_state)
        c.fit(F, y_on[:, j].astype(np.int8))
        regs.append(r); clfs.append(c)
        if verbose:
            print(f"  {app:18s} 회귀 {r.n_iter_:>4d}회 (양성 {int(on.sum()):>6,}창) "
                  f"/ 분류 {c.n_iter_:>4d}회 | 양성률 {100*on.mean():5.1f}% "
                  f"| {time.time()-t0:5.1f}s", flush=True)

    return BaselineModel(
        appliances=appliances, regressors=regs, classifiers=clfs,
        feature_names=feature_names(),
        config={"n_train": int(len(F)), "max_iter": max_iter,
                "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
                "loss": "absolute_error", "random_state": random_state},
    )


def permutation_importance_fast(
    model: BaselineModel, F: np.ndarray, y_power: np.ndarray,
    n_repeats: int = 1, subsample: int = 4000, random_state: int = 0,
) -> Dict[str, List[Tuple[str, float]]]:
    """기기별로 어떤 특징이 실제로 쓰였는지. 설계 근거 검증에 쓴다.

    예: 핫플레이트가 `blk_transitions`(0.5초 블록 전이 횟수)에 의존한다면
    0.4절의 "릴레이 펄스가 판별 단서" 가 실제로 작동한다는 뜻이다.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(F), size=min(subsample, len(F)), replace=False)
    Fs, ys = F[idx], y_power[idx]
    out: Dict[str, List[Tuple[str, float]]] = {}
    for j, app in enumerate(model.appliances):
        base = float(np.mean(np.abs(model.regressors[j].predict(Fs) - ys[:, j])))
        gains = []
        for c in range(Fs.shape[1]):
            tot = 0.0
            for _ in range(n_repeats):
                Fp = Fs.copy()
                Fp[:, c] = Fp[rng.permutation(len(Fp)), c]
                tot += float(np.mean(np.abs(model.regressors[j].predict(Fp) - ys[:, j])))
            gains.append((model.feature_names[c], tot / n_repeats - base))
        out[app] = sorted(gains, key=lambda kv: -kv[1])[:6]
    return out
