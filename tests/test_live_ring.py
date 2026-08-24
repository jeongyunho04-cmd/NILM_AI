"""실시간 입력 버퍼 회귀 테스트 — 순서 뒤바뀜을 견디는가

펌웨어는 선택적 재전송을 하므로 CSV 안에서 행이 `37 -> 28 -> 38 -> 29` 처럼
거꾸로 온다 (`nilm_receiver.py` 의 `REORDER_MAX` 주석). 옛 `run_live` 는
파일 순서대로 deque 에 append 해서 **4.5초 전 사이클을 '가장 최근' 자리에**
앉혔다. 세밀 갈래는 뒤 10초만 보고 타깃은 그 안의 539번째라(`inputs.py`)
오염되는 자리가 정확히 타깃 근방이었다.

12.21.4 의 교훈대로 회귀 테스트를 박아 둔다. 이 성질이 조용히 깨지면
지표는 멀쩡해 보이는데 입력이 틀려 있게 된다.
"""
import numpy as np
import pytest

from src.run_live import CYCLE_HZ, CycleRing


def cycle(v: float) -> np.ndarray:
    """채널 전체가 같은 값인 사이클. 자리만 확인하면 되므로 이걸로 충분하다."""
    return np.full(33, v, np.float32)


def fill(ring: CycleRing, ks) -> None:
    for k in ks:
        ring.push(cycle(float(k)), k / CYCLE_HZ)


def test_in_order_window_is_time_ordered():
    r = CycleRing(size=10)
    fill(r, range(10))
    assert r.ready()
    np.testing.assert_allclose(r.window()[0], np.arange(10, dtype=np.float32))


def test_out_of_order_row_lands_in_its_own_slot():
    """늦게 온 행이 제 시각 자리에 들어가고 머리를 밀지 않는다."""
    r = CycleRing(size=10)
    fill(r, [0, 1, 2, 3, 6, 7])          # 4, 5 가 아직 안 왔다
    assert r.push(cycle(4.0), 4 / CYCLE_HZ) == "backfill"
    assert r.push(cycle(5.0), 5 / CYCLE_HZ) == "backfill"
    assert r.k_head == 7                  # 되꽂아도 머리는 그대로
    fill(r, [8, 9])
    np.testing.assert_allclose(r.window()[0], np.arange(10, dtype=np.float32))


def test_shuffled_input_gives_the_same_window_as_clean_input():
    """이 파일의 핵심. 순서를 섞어도 창이 **동일**해야 한다."""
    n = 60
    clean = CycleRing(size=n)
    fill(clean, range(n))

    order = list(range(n))
    for i in range(0, n - 9, 7):          # 펌웨어 재전송 흉내: 9칸 뒤로 미룬다
        order.remove(i)
        order.insert(order.index(i + 9), i)
    shuffled = CycleRing(size=n)
    fill(shuffled, order)

    assert shuffled.stats()["n_backfill"] > 0, "테스트가 실제로 순서를 안 섞었다"
    np.testing.assert_allclose(shuffled.window(), clean.window())


def test_row_older_than_the_window_is_dropped():
    """창 밖으로 밀려난 행은 버린다. 새 자리에 덮어쓰면 미래를 오염시킨다."""
    r = CycleRing(size=10)
    fill(r, range(10, 20))
    assert r.push(cycle(-1.0), 3 / CYCLE_HZ) == "stale"
    assert r.stats()["n_stale"] == 1
    assert not np.any(r.window() == -1.0)


def test_gap_is_filled_with_the_previous_valid_cycle():
    """유실로 빈 자리는 직전 유효값으로 채운다 (전처리의 60Hz 보간에 해당)."""
    r = CycleRing(size=10, min_fill=0.5)
    fill(r, [0, 1, 2, 5, 6, 7, 8, 9])     # 3, 4 유실
    assert r.ready()
    w = r.window()[0]
    assert w[3] == 2.0 and w[4] == 2.0    # 앞으로 끌고 온다
    np.testing.assert_allclose(w[[0, 1, 2, 5, 6, 7, 8, 9]],
                               np.array([0, 1, 2, 5, 6, 7, 8, 9], np.float32))


def test_advancing_the_head_clears_the_slot_it_reuses():
    """한 바퀴 돌아 재사용하는 자리에 옛 값이 남으면 60초 전 사이클이 섞인다."""
    r = CycleRing(size=4)
    fill(r, [0, 1, 2, 3])
    r.push(cycle(4.0), 4 / CYCLE_HZ)      # 자리 0 을 재사용한다
    np.testing.assert_allclose(r.window()[0], np.array([1, 2, 3, 4], np.float32))


def test_falls_back_to_arrival_order_without_time():
    """`--no-reorder` 와 t_s 결측 시의 옛 동작. 회귀 비교용으로 살려 둔다."""
    r = CycleRing(size=5, use_time=False)
    for v in [3.0, 1.0, 2.0]:             # 시각을 무시하고 온 순서대로 쌓는다
        r.push(cycle(v), None)
    assert r.k_head == 2
    assert r.stats()["n_backfill"] == 0


def test_ready_requires_a_nearly_full_window():
    """예열이 안 끝났는데 예측을 내면 빈 자리가 0 으로 들어간다."""
    r = CycleRing(size=100, min_fill=0.98)
    fill(r, range(50))
    assert not r.ready()
    fill(r, range(50, 100))
    assert r.ready()


@pytest.mark.parametrize("shift", [1, 9, 32])
def test_stats_report_the_worst_reversal(shift):
    """되꽂은 폭을 보고해야 '이 세션의 입력이 얼마나 온전했나' 를 알 수 있다."""
    r = CycleRing(size=200)
    fill(r, range(100))
    r.push(cycle(0.0), (99 - shift) / CYCLE_HZ)
    s = r.stats()
    assert s["max_backfill_cycles"] == shift
    assert s["max_backfill_s"] == pytest.approx(shift / CYCLE_HZ)


# ── 세션 이어붙임 (보드 리셋) ──────────────────────────────────────────────
# 실측 데이터에서 실제로 일어난다: test.2.csv 2회, test3.csv 1회,
# oven_1 / hotplate_1 / noise_selfpower / beam_projector_2 각 1회.
# 되감긴 행을 '늦은 행' 으로 버리면 k_head 가 영영 안 내려가 **리셋 이후
# 전부를 잃는다** — 옛 append 동작보다 나쁘다. 그래서 이 테스트가 필요하다.
#
# 머리가 창 밖으로 충분히 나가 있어야 리셋과 되꽂음이 구별된다. 실제로도
# 리셋은 긴 녹화 뒤에 일어나므로(t_s 가 수백 초에서 0 으로) 이 조건이 맞다.

def test_board_reset_restarts_the_buffer_instead_of_dropping_everything():
    r = CycleRing(size=100, reset_after=5)
    fill(r, range(300))                    # 머리를 창 밖으로 보낸다
    assert r.ready()
    fill(r, range(0, 100))                 # t_s 가 0 으로 되감긴다
    assert r.stats()["n_seam"] == 1
    assert r.k_head == 99                  # 새 세션의 머리를 잡았다
    assert r.ready()
    np.testing.assert_allclose(r.window()[0], np.arange(100, dtype=np.float32))


def test_window_never_spans_two_sessions():
    """이어붙인 자리에서 창이 두 세션에 걸치면 60초가 통째로 거짓이 된다."""
    r = CycleRing(size=100, reset_after=5)
    fill(r, range(300))
    fill(r, range(0, 40))                  # 새 세션이 40 사이클만 쌓였다
    assert not r.ready(), "예열이 안 끝났는데 예측을 내면 옛 세션이 섞인다"
    fill(r, range(40, 100))
    assert r.ready()
    assert not np.any(r.window()[0] > 99)  # 옛 세션(100~299)이 한 칸도 없다


def test_a_single_stray_row_does_not_trigger_a_reset():
    """한 줄짜리 이상값에 60초 버퍼를 버리면 안 된다."""
    r = CycleRing(size=100, reset_after=5)
    fill(r, range(300))
    assert r.push(cycle(-1.0), 0.0) == "stale"
    fill(r, [300])                         # 정상 행이 이어진다
    assert r.stats()["n_seam"] == 0
    assert r.k_head == 300
    assert not np.any(r.window() == -1.0)


def test_rows_held_during_reset_detection_are_replayed_not_lost():
    """리셋 판정 전까지 들고 있던 행도 새 세션의 데이터다. 버리면 안 된다."""
    r = CycleRing(size=100, reset_after=5)
    fill(r, range(300))
    fill(r, range(0, 5))                   # 정확히 reset_after 개
    assert r.stats()["n_seam"] == 1
    assert r.k_head == 4
    np.testing.assert_allclose(r.window()[0][-5:], np.arange(5, dtype=np.float32))
    assert r.stats()["n_stale"] == 0       # 되살렸으므로 버린 것으로 세지 않는다


# ── 12.63 기착 절제 ────────────────────────────────────────────────────────

def test_pedestal_ablation_only_touches_named_appliance():
    """기착 절제가 지목한 가전만 줄이고 나머지는 바이트 그대로여야 한다."""
    import numpy as np
    from src.synthesis.segment_pool import SegmentPool

    base = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2)
    abl = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2,
                      ablate_pedestal_apps=["beam_projector"])
    for app in base.appliance_activations:
        b = base.appliance_activations[app]
        c = abl.appliance_activations[app]
        assert len(b) == len(c), app
        if app == "beam_projector":
            continue
        for x, y in zip(b, c):
            assert np.array_equal(x.target_power_w, y.target_power_w), app


def test_pedestal_ablation_removes_the_low_tail():
    """프로젝터 활성화의 끝이 저전력 꼬리가 아니라 정상 구간이 돼야 한다."""
    import numpy as np
    from src.synthesis.segment_pool import SegmentPool

    abl = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2,
                      ablate_pedestal_apps=["beam_projector"])
    for a in abl.appliance_activations["beam_projector"]:
        tp = np.asarray(a.target_power_w, np.float64)
        pk = np.median(tp[tp > 0.5 * tp.max()])
        # 끝값이 정상의 25% 미만인 꼬리가 0.5초 이상 남아 있으면 안 된다
        j = len(tp)
        while j > 0 and 0.0 < tp[j - 1] < 0.25 * pk:
            j -= 1
        assert len(tp) - j < 30, f"{a.source_file}: 꼬리 {(len(tp)-j)/60:.2f}s 남음"


def test_pedestal_ablation_keeps_arrays_consistent():
    """잘린 뒤에도 모든 배열 길이와 duration_cycles 가 맞아야 한다."""
    from src.synthesis.segment_pool import SegmentPool

    abl = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2,
                      ablate_pedestal_apps=["beam_projector"])
    for a in abl.appliance_activations["beam_projector"]:
        n = a.duration_cycles
        assert len(a.target_power_w) == n
        assert len(a.is_on) == n
        assert len(a.state_id) == n
        assert len(a.net_power_features) == n
        assert len(a.net_harmonics_complex) == n
        assert len(a.net_harmonics_ri) == n
        assert a.inrush_cycles <= n


def test_level_scramble_only_touches_named_appliance():
    """전력 준위 조작이 지목한 가전만 흔들어야 한다 (12.64절)."""
    import numpy as np
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.augmentor import DataAugmentor

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2)
    base = DataAugmentor()
    scr = DataAugmentor(level_scramble={"beam_projector": (0.64, 1.42)})
    for app in ("laptop_charger", "minipc"):
        a = pool.appliance_activations[app][0]
        np.random.seed(3); x = base.augment_activation(a).target_power_w
        np.random.seed(3); y = scr.augment_activation(a).target_power_w
        assert np.array_equal(x, y), app


def test_level_scramble_widens_the_named_appliance():
    """프로젝터의 준위 분포가 실제로 넓어져야 한다."""
    import numpy as np
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.augmentor import DataAugmentor

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="holdout", holdout_frac=0.2)
    acts = pool.appliance_activations["beam_projector"]
    spans = []
    for aug in (DataAugmentor(), DataAugmentor(level_scramble={"beam_projector": (0.64, 1.42)})):
        np.random.seed(11)
        v = []
        for _ in range(120):
            g = aug.augment_activation(acts[np.random.randint(len(acts))])
            tp = np.asarray(g.target_power_w, np.float64)
            if tp.max() > 0:
                v.append(np.median(tp[tp > 0.5 * tp.max()]))
        lo, hi = np.percentile(v, [10, 90])
        spans.append(hi - lo)
    assert spans[1] > 2.5 * spans[0], f"폭 {spans[0]:.1f} -> {spans[1]:.1f}"


def test_pair_accuracy_isolates_discrimination_not_detection():
    """쌍 정확도는 둘 중 하나만 켜진 창만 보고, 더 큰 쪽을 고른다 (12.65절)."""
    import numpy as np
    from src.run_pair_ablation import pair_accuracy

    apps = ["beam_projector", "laptop_charger", "minipc"]
    # 창 0: 프로젝터만 (맞음) / 1: 충전기만 (맞음) / 2: 프로젝터만 (틀림)
    # 창 3: 둘 다 ON -> 표본에서 빠져야 한다 / 4: 둘 다 OFF -> 빠져야 한다
    on = np.array([[1,0,0],[0,1,0],[1,0,0],[1,1,0],[0,0,1]], np.int8)
    pred = np.array([[40.,1.,0.],[1.,40.,0.],[1.,40.,0.],[40.,1.,0.],[0.,0.,9.]], np.float32)
    r = pair_accuracy(pred, on, apps)
    assert r["n"] == 3, r
    assert abs(r["acc"] - 2/3) < 1e-9, r
    assert abs(r["acc_proj"] - 0.5) < 1e-9, r
    assert abs(r["acc_chg"] - 1.0) < 1e-9, r


def test_even_dither_targets_even_orders_only():
    """짝수차 지터가 홀수차를 건드리지 않아야 한다 (12.69절)."""
    import numpy as np
    from src.synthesis.augmentor import DataAugmentor

    c = np.ones((50, 15), np.complex64)
    aug = DataAugmentor(harmonic_dither_even_amp=1.4)
    np.random.seed(0)
    out = np.asarray(aug._apply_harmonic_dither(c))
    dev = np.abs(np.abs(out[0]) - 1.0)
    # 1차는 언제나 불변, 홀수차(3,5,...)도 이 설정에서는 불변
    for k in (1, 3, 5, 7, 9, 11, 13, 15):
        assert dev[k - 1] < 1e-6, f"{k}차가 움직였다: {dev[k-1]}"
    assert dev[1] > 1e-3, "2차가 안 움직였다"


def test_even_dither_merges_the_pair_ratio():
    """even_amp 1.4 가 프로젝터↔충전기 |I2|/|I1| 을 실제로 겹치게 해야 한다."""
    import numpy as np
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.augmentor import DataAugmentor

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", holdout_frac=0.2)

    def dp(aug):
        vals = {}
        for app in ("beam_projector", "laptop_charger"):
            np.random.seed(5)
            acts = pool.appliance_activations[app]
            v = []
            for _ in range(120):
                g = aug.augment_activation(acts[np.random.randint(len(acts))])
                m = np.abs(np.asarray(g.net_harmonics_complex))
                tp = np.asarray(g.target_power_w)
                on = tp > 0.5 * tp.max()
                if on.sum() >= 30:
                    v.append(np.median(m[on, 1] / (m[on, 0] + 1e-9)))
            vals[app] = np.log(np.array(v))
        a, b = vals["beam_projector"], vals["laptop_charger"]
        return abs(b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)

    base = dp(DataAugmentor())
    dith = dp(DataAugmentor(harmonic_dither_even_amp=1.4))
    assert base > 3.0, f"기준 d' 이 이미 낮다: {base:.2f}"
    assert dith < 1.5, f"지터 뒤 d' 이 안 내려갔다: {dith:.2f}"


def test_harm_odd_only_masks_even_orders():
    """L_harm 의 짝수차 마스크가 홀수차만 남겨야 한다 (12.75절)."""
    import torch
    from src.model.losses import NILMLoss

    off = NILMLoss(s_i=torch.ones(9), harm_scale=torch.ones(15), harm_odd_only=False)
    on = NILMLoss(s_i=torch.ones(9), harm_scale=torch.ones(15), harm_odd_only=True)
    assert off.harm_mask.sum().item() == 15
    assert on.harm_mask.sum().item() == 8          # 1,3,5,7,9,11,13,15 차
    for j in (0, 2, 4, 6, 8, 10, 12, 14):          # 홀수차 (0-based 짝수 인덱스)
        assert on.harm_mask[j].item() == 1.0
    for j in (1, 3, 5, 7, 9, 11, 13):              # 짝수차
        assert on.harm_mask[j].item() == 0.0


def test_harm_odd_only_preserves_loss_scale():
    """마스크를 걸어도 손실 규모가 유지돼야 w_harm 의 뜻이 안 바뀐다."""
    import torch
    from src.model.losses import NILMLoss

    a = NILMLoss(s_i=torch.ones(9), harm_scale=torch.ones(15), harm_odd_only=False)
    b = NILMLoss(s_i=torch.ones(9), harm_scale=torch.ones(15), harm_odd_only=True)
    def f(e, m):
        return (e * m[None, :, None]).mean() / m.mean().clamp(min=1e-6)

    flat = torch.full((4, 15, 2), 3.0)         # 모든 차수의 오차가 같은 인공 입력
    assert torch.allclose(f(flat, a.harm_mask), torch.tensor(3.0))
    assert torch.allclose(f(flat, b.harm_mask), torch.tensor(3.0)), "마스크가 규모를 바꿨다"

    even = torch.zeros(4, 15, 2)               # 짝수차에만 오차가 있는 입력
    even[:, 1::2] = 5.0
    assert f(even, b.harm_mask).item() == 0.0, "마스크가 짝수차를 안 지웠다"
    assert f(even, a.harm_mask).item() > 0.0
