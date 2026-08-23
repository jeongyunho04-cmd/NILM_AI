"""
평가 하네스 / 시간 홀드아웃 자동 검증
======================================
평가 하네스는 모델보다 먼저 만들었으므로, 여기 테스트가 하네스의 유일한 안전망이다.
"지표가 실제로 그것을 재는가" 를 인공 예측기로 확인한다.
"""
from pathlib import Path
import json

import numpy as np
import pytest

from src.evaluation.holdout import build_holdout, load_holdout
from src.evaluation.metrics import (
    ON_POWER_THRESHOLD_W,
    format_table,
    resistive_confusion,
    score_appliances,
    summarize,
    total_power_residual,
)
from src.evaluation.real_events import build_on_off_truth, load_events, score_events, score_on_off
from src.evaluation.sealing import (
    SealedDatasetError,
    assert_not_sealed,
    filter_sealed,
    is_sealed,
    unseal,
)
from src.synthesis.segment_pool import SegmentPool

NPZ = "processed_data/npz"


# ── 시간 기반 홀드아웃 ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def pools():
    return {s: SegmentPool(npz_dir=NPZ, time_split=s) for s in ("all", "train", "holdout")}


def test_time_split_covers_every_appliance(pools):
    """9종 전부가 학습에도 홀드아웃에도 있어야 한다.

    파일 단위로 나누면 이게 불가능하다 - 에어컨·오븐·포트·프로젝터는 원본이
    1개씩이라 홀드아웃으로 빼면 학습에서 통째로 사라진다.
    """
    for app in pools["all"].get_appliance_types():
        for split in ("train", "holdout"):
            acts = pools[split].appliance_activations.get(app, [])
            assert acts, f"{app} 이 {split} 에 없습니다"


def test_time_split_is_disjoint_in_total_duration(pools):
    """학습 + 홀드아웃이 전체를 넘지 않아야 한다 (겹치면 누수다)."""
    for app in pools["all"].get_appliance_types():
        cyc = {s: sum(a.duration_cycles for a in pools[s].appliance_activations.get(app, []))
               for s in ("all", "train", "holdout")}
        assert cyc["train"] + cyc["holdout"] <= cyc["all"] * 1.02, (
            f"{app}: train {cyc['train']} + holdout {cyc['holdout']} > all {cyc['all']}"
        )
        assert cyc["train"] > cyc["holdout"], f"{app}: 홀드아웃이 학습보다 큽니다"


def test_time_split_noise_reference_is_stable(pools):
    """노이즈 기준 상수가 구간에 따라 크게 흔들리면 안 된다.

    흔들리면 학습/홀드아웃 사이에 '일반화' 가 아니라 '캘리브레이션 불일치' 가
    섞여 들어와 비교가 오염된다. 노이즈는 정상 과정이라 안정적이어야 한다.
    """
    for name in pools["all"].noise_references:
        a = pools["train"].noise_references[name].median_phasor
        b = pools["holdout"].noise_references[name].median_phasor
        # 기본파 기준 절대 차이. 가장 작은 기기(미니PC I1 약 0.14A)의 1% 미만이어야 한다.
        assert abs(a[0] - b[0]) < 0.0014, f"{name} 노이즈 기준이 구간에 따라 흔들립니다"


def test_time_split_rejects_bad_arguments():
    with pytest.raises(ValueError):
        SegmentPool(npz_dir=NPZ, time_split="뒤쪽")
    with pytest.raises(ValueError):
        SegmentPool(npz_dir=NPZ, holdout_frac=1.5)


def test_default_is_unchanged(pools):
    """time_split 기본값은 기존 동작과 같아야 한다 (회귀 방지)."""
    assert pools["all"].time_split == "all"
    assert SegmentPool(npz_dir=NPZ).time_split == "all"


# ── 봉인 ────────────────────────────────────────────────────────────────────
def test_sealing_blocks_final_test_file():
    assert is_sealed("test") and is_sealed("data/test.csv") and is_sealed("test.npz")
    for name in ("test", "data/test.csv", "processed_data/composite_eval/test.npz"):
        with pytest.raises(SealedDatasetError):
            assert_not_sealed(name)


def test_sealing_allows_validation_files():
    for name in ("test.2", "test3", "data/test.2.csv", "test3.npz"):
        assert not is_sealed(name)
        assert_not_sealed(name)          # 예외가 나면 실패


def test_sealing_filter_and_unseal(tmp_path, monkeypatch):
    assert filter_sealed(["test", "test.2", "test3"]) == ["test.2", "test3"]
    monkeypatch.setattr("src.evaluation.sealing.SEAL_LOG_PATH", tmp_path / "SEAL_BROKEN.json")
    with unseal("단위 테스트"):
        assert_not_sealed("test")        # 블록 안에서는 통과
    with pytest.raises(SealedDatasetError):
        assert_not_sealed("test")        # 블록을 나오면 다시 막힌다
    log = json.loads((tmp_path / "SEAL_BROKEN.json").read_text(encoding="utf-8"))
    assert log[0]["reason"] == "단위 테스트"


def test_sealing_requires_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.sealing.SEAL_LOG_PATH", tmp_path / "s.json")
    with pytest.raises(ValueError):
        with unseal("   "):
            pass


# ── 지표 ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def toy():
    """3기기 x 400창. 고부하 1대 + 저부하 2대."""
    rng = np.random.default_rng(0)
    apps = ["oven", "minipc", "beam_projector"]
    y = np.zeros((400, 3))
    y[:200, 0] = 1200 + rng.normal(0, 10, 200)      # 오븐: 앞 절반만 켜짐
    y[:, 1] = np.where(rng.random(400) < 0.5, 18.0, 0.0)
    y[:, 2] = np.where(rng.random(400) < 0.5, 50.0, 0.0)
    return apps, y, {"oven": 1209.0, "minipc": 17.6, "beam_projector": 50.6}


def test_perfect_predictor_scores_perfectly(toy):
    apps, y, s = toy
    sc = score_appliances(y, y.copy(), apps, s)
    for x in sc:
        assert x.mae_w < 1e-9
        assert x.f1 == pytest.approx(1.0)
        # 켜졌지만 임계(5W) 아래인 창은 'off' 로 세므로 FA 가 정확히 0 은 아닐 수 있다
        assert x.fa_w < ON_POWER_THRESHOLD_W


def test_zero_predictor_has_no_recall(toy):
    apps, y, s = toy
    sc = score_appliances(y, np.zeros_like(y), apps, s)
    for x in sc:
        assert x.f1 == 0.0 and x.fa_w == 0.0
        assert x.sae == pytest.approx(-1.0)


def test_fa_detects_high_to_low_transfer(toy):
    """고부하 오차를 저부하로 흘리면 FA(고부하 동시)가 잡아야 한다.

    이것이 4.3절 오귀속 지표의 존재 이유다. 전가된 기기의 MAE 는 거의 안 변하므로
    전체 MAE 만 보면 문제가 보이지 않는다.
    """
    apps, y, s = toy
    p = y.copy()
    leak = 0.03 * y[:, 0]
    p[:, 0] *= 0.97
    p[:, 1] += leak                       # 오븐 오차 전부를 미니PC 로

    base = {x.appliance: x for x in score_appliances(y, y.copy(), apps, s)}
    bad = {x.appliance: x for x in score_appliances(y, p, apps, s)}

    assert bad["minipc"].fa_high_w > 20.0, "전가가 FA(고부하 동시)에 안 잡혔습니다"
    assert bad["minipc"].transfer_w > 20.0, "전가량이 안 잡혔습니다"
    assert bad["minipc"].fa_high_rel > 0.15, "4.3절 목표치를 넘겼는데 통과로 나옵니다"
    # 오븐 자신의 상대 오차는 3% 라 눈에 안 띈다 - 그것이 요점이다
    assert bad["oven"].re_on < 0.05
    assert bad["oven"].mae_w > base["oven"].mae_w


def test_conservation_residual_is_blind_to_transfer(toy):
    """총전력 잔차는 전가를 못 본다. 3.3절이 L_harm 을 필수로 둔 이유다."""
    apps, y, _ = toy
    p = y.copy()
    p[:, 0] *= 0.97
    p[:, 1] += 0.03 * y[:, 0]
    obs = y.sum(axis=1)
    assert total_power_residual(p, obs)["mean_abs_w"] < 1e-9
    assert total_power_residual(y, obs)["mean_abs_w"] < 1e-9


def test_re_detects_systematic_underprediction(toy):
    """켜져 있을 때의 과소 예측은 FA 에 안 잡히고 RE 에만 잡힌다 (0.6절)."""
    apps, y, s = toy
    p = y.copy()
    p[:, 1] *= 0.66                      # 미니PC 를 34% 낮게 (실측 sim-to-real 편차)
    sc = {x.appliance: x for x in score_appliances(y, p, apps, s)}
    assert sc["minipc"].fa_w < 1e-9, "꺼진 구간은 건드리지 않았는데 FA 가 올랐습니다"
    assert sc["minipc"].re_on == pytest.approx(0.34, abs=0.02)
    assert sc["minipc"].bias_on_w < 0, "과소 예측인데 편향이 음수가 아닙니다"


def test_resistive_confusion_catches_misattribution():
    apps = ["electiric_kettle", "oven", "hair_dryer"]
    y = np.zeros((300, 3))
    y[:100, 0] = 1500.0
    y[100:200, 1] = 1200.0
    y[200:, 2] = 1000.0
    assert resistive_confusion(y, y.copy(), apps)["accuracy"] == pytest.approx(1.0)
    swap = y.copy()
    swap[:100] = swap[:100][:, [1, 0, 2]]        # 포트를 오븐으로 오인
    cm = resistive_confusion(y, swap, apps)
    assert cm["accuracy"] == pytest.approx(2 / 3, abs=0.02)
    assert cm["matrix"][0][1] == 100             # 참=포트, 예측=오븐


def test_summary_reports_target_pass(toy):
    apps, y, s = toy
    r = summarize(score_appliances(y, y.copy(), apps, s))
    assert r["fa_target_pass"].endswith("/2")    # toy 에는 저부하 2종만 있다
    assert format_table(score_appliances(y, y.copy(), apps, s)).count("\n") >= 3


# ── 실측 이벤트 ─────────────────────────────────────────────────────────────
def test_real_events_file_is_wellformed():
    """봉인 파일이 섞이지 않았는가 + 각 항목이 채점 가능한 형태인가.

    **정확한 집합 일치로 쓰면 안 된다.** 앞선 판은 `set(ev) == {"test.2", "test3"}`
    였는데, 검증용 복합 부하를 하나 추가할 때마다(2026-08-22 `test_4`) 봉인과 무관하게
    깨졌다. 실제로 지켜야 할 불변식은 '봉인된 것이 없다' 하나다.
    """
    from src.evaluation import sealing

    ev = load_events()
    assert ev, "real_events.json 이 비어 있습니다"
    sealed = [stem for stem in ev if sealing.is_sealed(stem)]
    assert not sealed, f"봉인된 파일이 들어 있습니다: {sealed} (설계 문서 4.3절)"
    for stem, spec in ev.items():
        assert spec["duration_s"] > 0
        assert spec["cycles"] > 0
        for e in spec["events"]:
            assert e["appliance"] and e["kind"] in ("on", "off", "mode")
        # 구간은 파일 길이 안에 있고 순서가 맞아야 한다
        for app, iv in spec["intervals"].items():
            for key in ("on", "uncertain"):
                for t0, t1 in iv.get(key, []):
                    assert 0 <= t0 < t1 <= spec["duration_s"], (
                        f"{stem}/{app}/{key} 구간이 파일 범위를 벗어납니다: [{t0}, {t1}]")


def test_uncertain_regions_are_not_scored():
    """오븐의 팬/조명 구간은 채점에서 빠져야 한다.

    타임라인은 히터 통전만 적어 두었다. 그 사이 구간을 OFF 로 채점하면
    모델이 맞게 예측해도 오답이 된다.
    """
    apps = ["minipc", "beam_projector", "oven"]
    n = int(413.0 * 60)
    on, scorable = build_on_off_truth("test3", apps, n)
    j = apps.index("oven")
    assert on[:, j].any(), "오븐 통전 구간이 없습니다"
    assert (~scorable[:, j]).any(), "오븐 불확실 구간이 표시되지 않았습니다"
    # 통전 구간은 불확실 안에 있어도 채점 대상이어야 한다
    assert scorable[on[:, j], j].all()

    # 미니PC 도 불확실 구간이 있다 (설계 문서 12.25절).
    # 이 단언은 원래 "미니PC 는 전 구간 확실" 이었는데, 그 전제가 틀렸다 —
    # `test3` 첫 63초는 총전력이 4.3W 라 미니PC(최저 유휴 7.8W)가 켜져 있을 수
    # 없었고, 정답이 그것을 ON 으로 적고 있었다. 테스트가 그 오류를 고정하고 있었다.
    jm = apps.index("minipc")
    assert (~scorable[:, jm]).any(), "미니PC 판정보류 구간이 사라졌습니다 (12.25절 정정)"
    assert scorable[on[:, jm], jm].all(), "미니PC ON 구간이 채점에서 빠졌습니다"
    assert not on[:int(63.0 * 60), jm].any(), "미니PC 가 없던 첫 63초가 다시 ON 이 됐습니다"


def test_on_off_scoring_rewards_a_correct_predictor():
    apps = ["minipc", "beam_projector", "oven"]
    n = int(413.0 * 60)
    truth, _ = build_on_off_truth("test3", apps, n)
    good = score_on_off(truth, "test3", apps)
    assert good["minipc"]["f1"] == pytest.approx(1.0)
    assert good["beam_projector"]["f1"] == pytest.approx(1.0)
    bad = score_on_off(np.zeros_like(truth), "test3", apps)
    assert bad["beam_projector"]["f1"] == 0.0
    assert good["oven"]["ignored_uncertain"] > 0


def test_event_delta_p_scoring():
    """빔프로젝터가 t=102.7s 에 +46.7W 로 켜지는 것을 재현하는 예측기는 통과해야 한다."""
    apps = ["minipc", "beam_projector", "oven"]
    n = int(413.0 * 60)
    pred = np.zeros((n, 3))
    pred[int(102.7 * 60):, apps.index("beam_projector")] = 46.7
    pred[int(63.4 * 60):, apps.index("oven")] = 1157.4
    rows = {r["appliance"]: r for r in score_events(pred, "test3", apps)}
    assert abs(rows["beam_projector"]["error_w"]) < 1.0
    assert rows["beam_projector"]["sign_correct"]
    assert abs(rows["oven"]["error_rel"]) < 0.02


def test_real_scoring_refuses_sealed_file():
    with pytest.raises(SealedDatasetError):
        build_on_off_truth("test", ["minipc"], 100)


# ── 고정 홀드아웃 셋 ────────────────────────────────────────────────────────
def test_holdout_build_and_load(tmp_path):
    meta = build_holdout(out_dir=tmp_path / "h", n_windows=40, window_cycles=600,
                         seed=7, progress_every=0)
    hs = load_holdout(tmp_path / "h")
    assert len(hs) == 40
    assert hs.X.shape == (40, 33, 600) and hs.X.dtype == np.float32
    assert hs.y_power.shape == (40, len(hs.appliances))
    assert meta["time_split"] == "holdout"
    from src.model.inputs import target_index as _ti
    assert meta["target_index"] == _ti(600)      # lookahead 를 따라간다
    assert len(meta["content_sha256"]) == 16
    # 부분집합 추출
    sub = hs.subset(hs.recipe == hs.recipe[0])
    assert 0 < len(sub) <= len(hs)


def test_holdout_is_reproducible(tmp_path):
    """같은 시드면 같은 셋이어야 한다. 아니면 실행 간 비교가 무의미해진다."""
    a = build_holdout(out_dir=tmp_path / "a", n_windows=30, seed=99, progress_every=0)
    b = build_holdout(out_dir=tmp_path / "b", n_windows=30, seed=99, progress_every=0)
    assert a["content_sha256"] == b["content_sha256"]
    c = build_holdout(out_dir=tmp_path / "c", n_windows=30, seed=100, progress_every=0)
    assert c["content_sha256"] != a["content_sha256"]


def test_holdout_power_decomposition_holds(tmp_path):
    """타깃 시점에서 P_관측 = Σ활성 + Σ대기 + 계측계 가 성립해야 한다."""
    build_holdout(out_dir=tmp_path / "h", n_windows=60, seed=5, progress_every=0)
    hs = load_holdout(tmp_path / "h")
    recon = hs.y_power.sum(1) + hs.y_standby.sum(1) + hs.p_noise
    assert np.max(np.abs(hs.p_observed - recon)) < 0.5


# ── 타깃 lookahead 배선 ─────────────────────────────────────────────────────
def test_lookahead_reaches_the_placement_logic(pools):
    """생성기의 target_lookahead 가 합성기의 배치 편향까지 전달되어야 한다.

    이 배선이 끊겨 있으면 lookahead 를 바꿔도 활성화는 기본값(끝-1초) 자리를
    계속 겨냥한다. 창 크기·타깃 위치 스윕이 통째로 무의미해지는데,
    지표는 그럴듯하게 나오므로 **조용히 틀린다.**

    [활성 구간의 '중심' 으로 재면 안 된다 - 2026-08-22 에 고쳤다]
    앞선 판은 on-mask 중심의 중앙값을 두 설정에서 비교해 30 사이클 이상 벌어지길
    요구했다. 그런데 창을 통째로 덮는 활성화가 35~40% 라 그쪽 중심은 항상 299.5 에
    고정되고(정보 없음), 남은 표본이 어느 활성화를 뽑았느냐에 따라 중앙값이 흔들린다.
    실제로 풀에 파일 3개를 추가하자 격차가 30.7 -> 20.9 로 떨어져 테스트가 깨졌는데,
    **배선은 멀쩡했다.** 옛 풀에서도 여유가 0.7 사이클(2%)뿐이었다.

    대신 **각 설정이 자기 타깃 인덱스를 덮는 빈도**를 직접 본다. 배치 편향이 겨냥하는
    것이 바로 그 지점이라 포화되지 않고, 풀 구성이 바뀌어도 0.87~0.90 으로 일정하다
    (남의 타깃은 0.68~0.78). 측정한 여유는 최소 0.09 다.
    """
    from src.synthesis.dataset import NILMBatchGenerator

    pool = pools["train"]
    targets = {la: 600 - 1 - la for la in (60, 299)}
    hits = {}
    for la, tgt in targets.items():
        np.random.seed(3)
        g = NILMBatchGenerator(
            segment_pool=pool, window_size_cycles=600, target_lookahead_cycles=la,
            recipe_mix={"high_power_resistive": 1.0})
        assert g.target_index == tgt
        own = other = 0
        other_idx = targets[299 if la == 60 else 60]
        for _ in range(400):
            smp, _ = g._synthesize_window()
            on = np.zeros(600, bool)
            for a in smp.appliance_types:
                on |= smp.gt_is_on[a].astype(bool)
            own += int(on[tgt]); other += int(on[other_idx])
        hits[la] = (own / 400, other / 400)

    # 두 설정 모두 '자기 타깃' 을 '남의 타깃' 보다 자주 덮어야 한다.
    # 배선이 끊기면 둘 다 기본값(539)만 겨냥하므로 la=299 에서 부호가 뒤집힌다.
    for la, (own, other) in hits.items():
        assert own > other + 0.04, (
            f"lookahead={la} (타깃 {targets[la]}) 배치가 따라오지 않습니다: "
            f"자기 타깃 {own:.3f} vs 남의 타깃 {other:.3f} | 전체 {hits}")


def test_lookahead_flows_through_training_set_builder():
    """build_training_set 의 lookahead 가 실제 라벨 시점을 바꾸는가."""
    from src.baseline.train import build_training_set

    out = {}
    for la in (60, 299):
        _, yp, yo, apps = build_training_set(
            n_windows=200, window_cycles=600, time_split="holdout",
            seed=11, n_workers=1, chunk=200, target_lookahead_cycles=la)
        out[la] = yo.mean()
    # 값 자체보다 '다르게 나오는가' 가 핵심이다 (같으면 배선이 끊긴 것)
    assert out[60] != out[299], f"lookahead 가 라벨에 영향을 주지 않습니다: {out}"


def test_real_event_labels_are_physically_possible():
    """정답이 "켜짐" 이라 한 구간의 **총전력**이 그 기기의 최저 소비보다 낮으면 안 된다.

    설계 문서 12.25절. `test.2` 는 정답 ON 의 8.5%, `test3` 는 14.6% 가 이 검사에
    걸렸다 — 총전력이 0.76W 인데 미니PC(최저 유휴 7.8W)가 켜져 있다고 적혀 있었다.
    12.13 의 핫플 granularity 오류와 같은 종류이고, 이 한 줄 검사로 잡힌다.

    `uncertain` 구간은 채점에서 빠지므로 여기서도 뺀다.
    """
    import json
    import numpy as np
    from pathlib import Path
    from src.preprocessing import load_nilm_npz

    ev_path = Path("processed_data/real_events.json")
    if not ev_path.exists():
        pytest.skip("real_events.json 이 없습니다")
    files = json.loads(ev_path.read_text(encoding="utf-8"))["files"]

    # 기기별 최저 소비 (개별 녹화의 p10 중 최솟값). 여유를 두고 보수적으로 잡는다.
    MIN_W = {"minipc": 7.0, "beam_projector": 35.0, "laptop_charger": 15.0,
             "hotplate": 300.0, "oven": 10.0, "electiric_kettle": 800.0,
             "hair_dryer": 300.0, "fan": 15.0, "air_conditioner": 10.0}
    GUARD_CYCLES = 15          # 0.25초. 라벨 시각 해상도(0.5초)의 절반

    bad = []
    for stem, spec in files.items():
        npz = Path("processed_data/composite_eval") / f"{stem}.npz"
        if not npz.exists():
            continue
        p = np.asarray(load_nilm_npz(npz)["power_features"])[:, 0]
        n = len(p)
        for app, iv in spec.get("intervals", {}).items():
            floor = MIN_W.get(app)
            if floor is None or not iv.get("on"):
                continue
            m = np.zeros(n, bool)
            for a, b in iv["on"]:
                i0, i1 = int(a * 60), int(b * 60)
                # 라벨 시각은 seq 기준이라 **0.5초 양자화**돼 있다 (t_s = seq x 0.5).
                # 핫플 통전 펄스는 중앙 1.5초라, 경계가 반 칸만 어긋나도 펄스의
                # 10% 넘게 통전 밖 사이클이 섞인다. 실제로 test_5 는 가드 없이
                # 10.85%, 0.25초 가드로 0.53% 다 — 라벨 오류가 아니라 해상도다.
                # 가드는 펄스 길이의 1/4 를 넘지 않게 해 짧은 펄스를 지우지 않는다.
                guard = min(GUARD_CYCLES, max(0, (i1 - i0) // 4))
                m[i0 + guard:max(i0 + guard, i1 - guard)] = True
            for a, b in iv.get("uncertain", []):
                m[int(a * 60):int(b * 60)] = False
            if not m.any():
                continue
            share = float((p[m] < floor).mean())
            if share > 0.02:
                bad.append(f"{stem}/{app}: ON 구간의 {100*share:.1f}% 에서 "
                           f"총전력이 {floor}W 미만 (최소 {p[m].min():.2f}W)")
    assert not bad, "정답이 신호와 모순됩니다: " + " | ".join(bad)
