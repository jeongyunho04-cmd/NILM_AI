"""
학습용 독립 창 캐시 생성
=========================
60초 창 합성이 550 win/s 라 GPU 가 7~10% 만 일한다 (12.8.2절). 미리 만들어 두면
학습이 GPU 병목으로 바뀌어 2M 창 기준 61분 -> 3.5분이 된다.

python -m src.run_build_traincache                  # 30만창, 약 13.5GB, 9분
python -m src.run_build_traincache --windows 100000 # 작게
"""
import argparse, json, sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
from src.model.traincache import build_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="학습용 독립 창 캐시 생성")
    ap.add_argument("--out", default="cache/train60")
    ap.add_argument("--windows", type=int, default=300_000)
    ap.add_argument("--window-cycles", type=int, default=3600)
    ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-activation-files", default="",
                    help='녹화 단위 홀드아웃 (12.18절). JSON 딕셔너리, {가전: [녹화 stem]}')
    # 차수별 지터 (12.62절). 지정한 값이 **홀수 차수의 중앙값**이 되고 차수에
    # 비례해 커진다. 실측 산포는 A 녹화내부 21.4%/10.3°, B 녹화간 6.6%/1.7°,
    # C 실측복합 53.3%/15.1° 다 (`run_fingerprint_spread_probe`).
    # A 는 모델이 이미 보는 변동이므로 그 아래로 주면 효과가 없다.
    ap.add_argument("--recipe-mix", default="", metavar="NAME|JSON",
                    help="레시피 믹스. run_recipe_mix_probe 의 프리셋 이름(half/full) 이나 "
                         "JSON. 기본은 DEFAULT_RECIPE_MIX (12.67절)")
    ap.add_argument("--power-scale-std", default="", metavar="NAME|JSON",
                    help="기기별 전력 증강 폭 (12.118). 프리셋 measured/resistive "
                         "또는 JSON. 기본은 일괄 0.05")
    ap.add_argument("--dither-even-amp", type=float, default=0.0,
                    help="짝수차 전용 지터 σ (12.69절). 차수 비례를 쓰지 않고 무리 전체에 "
                         "같은 값. 1.4 에서 프로젝터↔충전기 |I2|/|I1| d' 가 5.04 -> 1.06")
    ap.add_argument("--dither-even-phase-deg", type=float, default=0.0)
    ap.add_argument("--sp-curves", action="store_true",
                    help="증강의 전력 스케일을 **부하 의존 서명** `s(p)` 로 옮긴다 "
                         "(12.166). 지금은 `I <- I·a` 로 선형인데, 캡 입력 SMPS 는 "
                         "부하가 바뀌면 도통각이 바뀌어 **모양도 바뀐다**. 실측에서 "
                         "같은 기기의 부하 차이(45.7°)가 다른 기기와의 차이(9.0°)보다 "
                         "3.6~7.3배 크다. 곡선이 없는 기기와 유효 범위 밖은 기존 선형.")
    ap.add_argument("--background", action="store_true",
                    help="**상시 배경 부하**를 넣는다 (12.166). 실측 '모든 기기 OFF' "
                         "창에 2.6~5.3W / k≈3.1 의 강한 용량성 부하가 늘 흐르는데 "
                         "합성에 없었다. 배경 5W 의 |I1| 0.074A 가 미니PC 9.5W 의 "
                         "0.050A 보다 커서, 안 넣으면 모델이 그 전류를 가장 싼 SMPS 로 "
                         "흘린다 (12.159 의 미니PC −77%%). `p_noise` 에 합산된다.")
    ap.add_argument("--dither-amp", type=float, default=0.0,
                    help="고조파 진폭 지터 (로그정규 σ, 홀수차 중앙값). 예: 0.50")
    ap.add_argument("--dither-phase-deg", type=float, default=0.0,
                    help="고조파 위상 지터 (도, 홀수차 중앙값). 예: 15")
    a = ap.parse_args()
    print("=" * 78); print("[NILM AI] 학습용 독립 창 캐시"); print("=" * 78)
    excl = json.loads(a.exclude_activation_files) if a.exclude_activation_files else None
    mix = None
    if a.recipe_mix:
        from src.run_recipe_mix_probe import PRESETS
        mix = PRESETS[a.recipe_mix] if a.recipe_mix in PRESETS else json.loads(a.recipe_mix)
        if abs(sum(mix.values()) - 1.0) > 1e-6:
            raise SystemExit(f'레시피 믹스 합이 1 이 아닙니다: {sum(mix.values()):.4f}')
    if excl:
        print(f"  ** 녹화 단위 홀드아웃: {excl} - 이 녹화의 활성화는 학습에서 뺀다 **")
    if mix:
        print(f"  ** 레시피 믹스 '{a.recipe_mix}': 동시성을 올린다 (12.67절) **")
    pss = None
    if a.power_scale_std:
        from src.synthesis.augmentor import POWER_SCALE_STD_PRESETS
        pss = (POWER_SCALE_STD_PRESETS[a.power_scale_std]
               if a.power_scale_std in POWER_SCALE_STD_PRESETS else json.loads(a.power_scale_std))
        print(f"  ** 기기별 전력 증강 폭 '{a.power_scale_std}' (12.118): "
              + ", ".join(f"{k}={v:g}" for k, v in sorted(pss.items())) + " **")
    if a.dither_even_amp > 0 or a.dither_even_phase_deg > 0:
        print(f"  ** 짝수차 지터: σ={a.dither_even_amp:.2f} / 위상 {a.dither_even_phase_deg:.1f}° "
              f"(무리 전체 동일, 12.69절) **")
    if a.dither_amp > 0 or a.dither_phase_deg > 0:
        print(f"  ** 차수별 지터: 진폭 σ={a.dither_amp:.2f} / 위상 {a.dither_phase_deg:.1f}° "
              f"(홀수차 중앙값, 차수 비례) **")
    build_cache(out_dir=a.out, n_windows=a.windows, window_cycles=a.window_cycles,
                time_split=a.split, seed=a.seed, n_workers=a.workers,
                exclude_activation_files=excl,
                dither_amp=a.dither_amp, dither_phase_deg=a.dither_phase_deg,
                recipe_mix=mix,
                dither_even_amp=a.dither_even_amp,
                dither_even_phase_deg=a.dither_even_phase_deg,
                power_scale_std_map=pss,
                sp_curves=a.sp_curves, background=a.background)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
