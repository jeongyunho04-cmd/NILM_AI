"""
장시간 학습 데이터 생성 및 구성 확인용 진단 차트
==================================================
몇 시간짜리 가정 소비 타임라인을 만들고, 그것이 어떻게 조합되었는지
눈으로 확인할 수 있는 그래프를 함께 생성한다.

# 실행 (기본 2시간):
python -m src.run_long_dataset

# 길이 지정:
python -m src.run_long_dataset --minutes 30

* 출력:
  - synthetic_data/long/<이름>.npz              합성 데이터
  - synthetic_data/long/<이름>_overview.png     전체 구간 개요 4패널
  - synthetic_data/long/<이름>_detail.png       가장 복잡한 구간 확대 3패널
  - synthetic_data/long/<이름>_report.json      구성 통계
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import sys
import time
import numpy as np

# Safe utf-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import LoadSynthesizer, SyntheticLoadSample
from src.synthesis.scenario_generator import ScenarioGenerator

SAMPLING_HZ = 60.0
# 색상은 기기마다 고정한다. 개요와 확대 그래프에서 같은 기기가 같은 색이어야
# 두 그림을 나란히 놓고 읽을 수 있다.
APPLIANCE_COLORS = {
    "air_conditioner": "#1f77b4",
    "beam_projector": "#ff7f0e",
    "electiric_kettle": "#d62728",
    "fan": "#2ca02c",
    "hair_dryer": "#9467bd",
    "hotplate": "#8c564b",
    "laptop_charger": "#e377c2",
    "minipc": "#17becf",
    "oven": "#bcbd22",
}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Malgun Gothic", "Gulim", "DejaVu Sans", "Arial"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """누적합 기반 O(N) 이동평균. 지속 부하를 보기 위한 것이다."""
    if len(x) <= window:
        return np.full(len(x), float(np.mean(x)))
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = (c[window:] - c[:-window]) / window
    # 원래 길이에 맞춰 앞쪽을 첫 값으로 채운다
    return np.concatenate([np.full(window - 1, out[0]), out])


def _active_appliances(sample: SyntheticLoadSample) -> List[str]:
    """실제로 한 번이라도 켜진 가전 (가동 시간이 긴 순)."""
    apps = [a for a in sample.appliance_types if sample.gt_is_on[a].any()]
    return sorted(apps, key=lambda a: -int(sample.gt_is_on[a].sum()))


def plot_overview(sample: SyntheticLoadSample, title: str, output_path: Union[str, Path]) -> str:
    """전체 구간이 어떻게 조합되었는지 보여주는 4패널 개요."""
    plt = _setup_matplotlib()

    n = sample.duration_cycles
    # 2시간이면 43만 점이라 그대로 그리면 선이 뭉개진다. 초 단위로 줄인다.
    step = max(1, n // 7200)
    t_min = sample.t_rel_s[::step] / 60.0
    p_total = sample.power_features[::step, 0]
    apps = _active_appliances(sample)

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 13), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.6, 1.1, 1.1]},
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    # ── 패널 1: 합성 총전력과 기기별 정답을 쌓아 올린 그림 ──────────────────
    ax = axes[0]
    stack = [sample.gt_target_power_w[a][::step] for a in apps]
    if stack:
        ax.stackplot(
            t_min, *stack, labels=apps,
            colors=[APPLIANCE_COLORS.get(a, "#777777") for a in apps], alpha=0.75,
        )
    ax.plot(t_min, p_total, color="black", linewidth=0.9, alpha=0.9, label="관측 총전력 P")
    ax.set_ylabel("전력 (W)", fontsize=11)
    ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        "① 기기별 정답을 쌓으면 관측 총전력이 된다  "
        "(검은 선이 쌓은 면과 일치 = 물리 보존 성립)",
        fontsize=11, loc="left",
    )

    # ── 패널 2: 어느 기기가 언제 켜져 있었나 ────────────────────────────────
    ax = axes[1]
    for i, a in enumerate(apps):
        on = sample.gt_is_on[a][::step].astype(bool)
        plugged = sample.gt_is_plugged[a][::step].astype(bool)
        # 꽂혀만 있는 구간을 옅게, 실제 가동 구간을 진하게
        ax.fill_between(t_min, i - 0.38, i + 0.38, where=plugged,
                        color=APPLIANCE_COLORS.get(a, "#777777"), alpha=0.13, step="mid")
        ax.fill_between(t_min, i - 0.38, i + 0.38, where=on,
                        color=APPLIANCE_COLORS.get(a, "#777777"), alpha=0.95, step="mid")
    ax.set_yticks(range(len(apps)))
    ax.set_yticklabels(apps, fontsize=9)
    ax.set_ylim(-0.6, len(apps) - 0.4)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_title(
        "② 가동 스케줄  (진한 색 = 켜짐 / 옅은 색 = 콘센트에 꽂혀만 있음(대기전력))",
        fontsize=11, loc="left",
    )

    # ── 패널 3: 지속 부하와 멀티탭 한도 ─────────────────────────────────────
    ax = axes[2]
    sustained_full = _moving_average(sample.power_features[:, 0], int(2.0 * SAMPLING_HZ))
    ax.plot(t_min, p_total, color="#999999", linewidth=0.6, alpha=0.6, label="순간 전력")
    ax.plot(t_min, sustained_full[::step], color="#d62728", linewidth=1.2,
            label="지속 부하 (2초 이동평균)")
    limit = sample.metadata.get("sustained_power_limit_w")
    if limit:
        ax.axhline(limit, color="black", linestyle="--", linewidth=1.2,
                   label=f"멀티탭 한도 {limit:.0f}W")
    ax.set_ylabel("전력 (W)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        "③ 지속 부하는 한도 아래로 유지된다  (순간 스파이크는 넘어도 무방)",
        fontsize=11, loc="left",
    )

    # ── 패널 4: 계통 전압 ───────────────────────────────────────────────────
    ax = axes[3]
    base_v = sample.metadata.get("base_voltage_v", 220.0)
    ax.plot(t_min, sample.v_bus[::step], color="#1f77b4", linewidth=0.9,
            label="단자 전압 V_bus (0.5초 계측 해상도)")
    ax.axhline(base_v, color="gray", linestyle="--", alpha=0.6,
               label=f"환경 기저 {base_v:.1f}V ({sample.metadata.get('voltage_environment','?')})")
    ax.set_ylabel("전압 (V)", fontsize=11)
    ax.set_xlabel("시간 (분)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("④ 부하가 걸릴 때마다 전압이 내려간다 (Z_grid 전압 강하)",
                 fontsize=11, loc="left")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def plot_detail(
    sample: SyntheticLoadSample, title: str, output_path: Union[str, Path],
    window_s: float = 180.0,
) -> str:
    """가장 많은 기기가 겹친 구간을 확대해 합성 원리를 보여준다."""
    plt = _setup_matplotlib()

    n = sample.duration_cycles
    apps = _active_appliances(sample)
    if not apps:
        return ""

    # 동시 가동 수가 가장 많은 지점을 찾는다
    concurrency = np.sum([sample.gt_is_on[a] for a in apps], axis=0)
    w = int(window_s * SAMPLING_HZ)
    smooth = _moving_average(concurrency.astype(np.float64), min(w, max(1, n // 4)))
    center = int(np.argmax(smooth))
    s0 = max(0, center - w // 2)
    s1 = min(n, s0 + w)
    s0 = max(0, s1 - w)

    step = max(1, (s1 - s0) // 6000)
    sl = slice(s0, s1, step)
    t_s = sample.t_rel_s[sl]

    # 세 번째 패널만 더 좁은 구간을 그리므로 x축을 공유하지 않는다.
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2.2, 1.3, 1.2]})
    axes[0].sharex(axes[1])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    # ── 패널 1: 총전력과 성분 ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_s, sample.power_features[sl, 0], color="black", linewidth=1.4,
            label="관측 총전력 P", zorder=5)
    for a in apps:
        gp = sample.gt_target_power_w[a][sl]
        if gp.max() > 1.0:
            ax.plot(t_s, gp, color=APPLIANCE_COLORS.get(a, "#777777"),
                    linewidth=1.1, alpha=0.85, label=f"{a}")
    standby = sum(sample.gt_standby_power_w[a][sl] for a in sample.appliance_types)
    ax.plot(t_s, standby + sample.p_noise_w[sl], color="#777777", linewidth=1.0,
            linestyle=":", label="대기전력 합 + 계측계 소비")
    ax.set_ylabel("전력 (W)", fontsize=11)
    ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("① 개별 기기 신호가 더해져 총전력이 된다", fontsize=11, loc="left")

    # ── 패널 2: 고조파 지문 ─────────────────────────────────────────────────
    # 저항 히터는 기본파만 먹으므로, 1kW 히터가 켜져 있어도 3차 고조파에는
    # SMPS 기기(미니PC, 충전기, 프로젝터)의 신호만 남는다.
    ax = axes[1]
    h = sample.harmonics_ri[sl]
    mag = np.sqrt(h[:, :, 0] ** 2 + h[:, :, 1] ** 2)
    ax.plot(t_s, mag[:, 0], color="#333333", linewidth=1.2, label="기본파 I1 (A)")
    ax.plot(t_s, mag[:, 2] * 10, color="#d62728", linewidth=1.1, label="3차 고조파 I3 x10 (A)")
    ax.plot(t_s, mag[:, 4] * 10, color="#1f77b4", linewidth=1.0, label="5차 고조파 I5 x10 (A)")
    ax.set_ylabel("전류 (A)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("시간 (초)", fontsize=11)
    ax.set_title(
        "② 고조파 지문  (저항 히터는 기본파만 올린다 - 3·5차에는 SMPS 기기 신호가 남는다)",
        fontsize=11, loc="left",
    )

    # ── 패널 3: 전압 계단 (여기만 더 확대) ──────────────────────────────────
    # 앞의 두 패널과 같은 180초로 그리면 0.5초 계단이 선 굵기에 묻혀 보이지 않는다.
    # 계측 해상도를 보여 주는 것이 목적이므로 이 패널만 20초로 좁힌다.
    ax = axes[2]
    zoom_w = int(20.0 * SAMPLING_HZ)
    z0 = max(0, (s0 + s1) // 2 - zoom_w // 2)
    z1 = min(n, z0 + zoom_w)
    zsl = slice(z0, z1)
    t_z = sample.t_rel_s[zsl]
    ax.plot(t_z, sample.v_bus_true[zsl], color="#bbbbbb", linewidth=1.0,
            label="실제 연속 전압 (60Hz)")
    ax.plot(t_z, sample.v_bus[zsl], color="#d62728", linewidth=1.5,
            drawstyle="steps-post", label="계측 전압 (0.5초 계단 - 모델이 보는 값)")
    ax.set_ylabel("전압 (V)", fontsize=11)
    ax.set_xlabel("시간 (초)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        f"③ 전압 계측 해상도 확대 ({t_z[0]:.0f}~{t_z[-1]:.0f}초)  "
        "- 실측 센서는 0.5초에 한 번만 전압을 갱신한다",
        fontsize=11, loc="left",
    )

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def build_report(sample: SyntheticLoadSample) -> Dict:
    """이 데이터가 어떻게 구성되었는지 숫자로 정리한다."""
    n = sample.duration_cycles
    p = sample.power_features[:, 0]
    sustained = _moving_average(p, int(2.0 * SAMPLING_HZ))
    ok, err = sample.verify_power_decomposition(tolerance_w=0.01)

    per_app = {}
    for a in sample.appliance_types:
        on = sample.gt_is_on[a]
        transitions = int(np.count_nonzero(np.diff(on.astype(np.int8)) == 1))
        per_app[a] = {
            "on_minutes": round(float(on.sum()) / SAMPLING_HZ / 60.0, 2),
            "on_percentage": round(100.0 * float(on.mean()), 2),
            "activations": transitions + (1 if on[0] else 0),
            "plugged_percentage": round(100.0 * float(sample.gt_is_plugged[a].mean()), 1),
            "mean_active_w": round(float(sample.gt_target_power_w[a].mean()), 2),
            "peak_active_w": round(float(sample.gt_target_power_w[a].max()), 1),
            "mean_standby_w": round(float(sample.gt_standby_power_w[a].mean()), 3),
        }

    limit = sample.metadata.get("sustained_power_limit_w")
    return {
        "duration_minutes": round(n / SAMPLING_HZ / 60.0, 2),
        "duration_cycles": n,
        "energy_kwh": round(float(np.sum(p)) / SAMPLING_HZ / 3600.0 / 1000.0, 4),
        "power": {
            "mean_w": round(float(p.mean()), 1),
            "median_w": round(float(np.median(p)), 1),
            "peak_instant_w": round(float(p.max()), 1),
            "peak_sustained_2s_w": round(float(sustained.max()), 1),
            "sustained_limit_w": limit,
            "sustained_limit_respected": bool(limit is None or sustained.max() <= limit),
        },
        "voltage": {
            "environment": sample.metadata.get("voltage_environment"),
            "base_v": sample.metadata.get("base_voltage_v"),
            "mean_v": sample.metadata.get("mean_v_bus"),
            "min_v": sample.metadata.get("min_v_bus"),
            "max_sag_v": sample.metadata.get("max_v_sag_v"),
        },
        "scheduling": {
            "episodes_scheduled": sample.metadata.get("episodes_scheduled"),
            "episodes_rejected_overlap": sample.metadata.get("episodes_rejected_overlap"),
            "episodes_rejected_over_budget": sample.metadata.get("episodes_rejected_over_budget"),
        },
        "power_decomposition_ok": bool(ok),
        "power_decomposition_max_error_w": round(err, 6),
        "appliances": per_app,
    }


def run(minutes: float = 120.0, name: str = "", output_dir: str = "synthetic_data/long",
        seed: Optional[int] = None) -> Dict:
    if seed is not None:
        np.random.seed(seed)
    name = name or f"household_{int(minutes)}min"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 84)
    print(f"[NILM AI] 장시간 학습 데이터 생성 - {minutes:.0f}분 ({int(minutes*60*SAMPLING_HZ):,} 사이클)")
    print("=" * 84)

    t0 = time.time()
    pool = SegmentPool(npz_dir="processed_data/npz")
    synthesizer = LoadSynthesizer(
        segment_pool=pool,
        grid_simulator=GridSimulator(),
        augmentor=DataAugmentor(duration_scale_range=(0.6, 2.2), power_scale_std=0.05),
        compute_gt_harmonics=False,   # 전력·상태 학습에는 필요 없다
    )
    print(f"세그먼트 풀 적재: {time.time()-t0:.1f}s | 가전 {len(pool.get_appliance_types())}종 | "
          f"지속부하 한도 {synthesizer.sustained_power_limit_w:.0f}W")

    t1 = time.time()
    gen = ScenarioGenerator(synthesizer=synthesizer)
    sample = gen.create_long_timeline(duration_min=minutes)
    print(f"합성 완료: {time.time()-t1:.1f}s")

    report = build_report(sample)

    npz_path = out_dir / f"{name}.npz"
    ScenarioGenerator.export_synthetic_sample_to_npz(sample, npz_path)

    print("\n차트 생성 중 ...", end="", flush=True)
    overview = plot_overview(
        sample, f"NILM 합성 학습 데이터 구성 개요 - {minutes:.0f}분", out_dir / f"{name}_overview.png")
    detail = plot_detail(
        sample, f"NILM 합성 원리 확대 - 동시 가동이 가장 많은 구간", out_dir / f"{name}_detail.png")
    print(" 완료")

    report["files"] = {"npz": str(npz_path), "overview_png": overview, "detail_png": detail}
    report_path = out_dir / f"{name}_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # ── 콘솔 요약 ───────────────────────────────────────────────────────────
    pw = report["power"]
    print("\n" + "-" * 84)
    print(f"{'가전':18s}{'가동(분)':>10s}{'가동%':>8s}{'횟수':>6s}{'꽂힘%':>8s}{'평균W':>9s}{'피크W':>9s}{'대기W':>8s}")
    print("-" * 84)
    for a, d in sorted(report["appliances"].items(), key=lambda kv: -kv[1]["on_minutes"]):
        print(f"{a:18s}{d['on_minutes']:>10.1f}{d['on_percentage']:>8.1f}{d['activations']:>6d}"
              f"{d['plugged_percentage']:>8.0f}{d['mean_active_w']:>9.1f}{d['peak_active_w']:>9.0f}"
              f"{d['mean_standby_w']:>8.3f}")
    print("-" * 84)
    print(f"총 에너지 {report['energy_kwh']:.4f} kWh | 평균 {pw['mean_w']:.0f}W | "
          f"순간 피크 {pw['peak_instant_w']:.0f}W | 지속 피크 {pw['peak_sustained_2s_w']:.0f}W")
    if pw["sustained_limit_w"] is None:
        print("멀티탭 한도: 설정 없음 (제한하지 않음)")
    else:
        print(f"멀티탭 한도 {pw['sustained_limit_w']:.0f}W 준수: "
              f"{'예' if pw['sustained_limit_respected'] else '아니오'} "
              f"(여유 {pw['sustained_limit_w'] - pw['peak_sustained_2s_w']:.0f}W)")
    v = report["voltage"]
    print(f"전압 환경 {v['environment']} | 기저 {v['base_v']}V | 평균 {v['mean_v']}V | "
          f"최저 {v['min_v']}V | 최대 강하 {v['max_sag_v']}V")
    sc = report["scheduling"]
    print(f"에피소드 {sc['episodes_scheduled']}개 배치 "
          f"(같은 기기 겹침으로 제외 {sc['episodes_rejected_overlap']}, "
          f"용량 초과로 제외 {sc['episodes_rejected_over_budget']})")
    print(f"전력 분해 검산: {'통과' if report['power_decomposition_ok'] else '실패'} "
          f"(최대 오차 {report['power_decomposition_max_error_w']:.6f}W)")
    print("=" * 84)
    print(f"저장 위치: {out_dir.resolve()}")
    print("=" * 84 + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="장시간 NILM 합성 학습 데이터 생성")
    ap.add_argument("--minutes", type=float, default=120.0, help="생성할 길이 (분, 기본 120)")
    ap.add_argument("--name", type=str, default="", help="출력 파일 이름")
    ap.add_argument("--seed", type=int, default=None, help="재현용 난수 시드")
    args = ap.parse_args()
    run(minutes=args.minutes, name=args.name, seed=args.seed)
