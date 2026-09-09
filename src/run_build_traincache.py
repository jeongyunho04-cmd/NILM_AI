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
    ap.add_argument("--level-scramble", default="",
                    help="기기별 전력 **범위** 표집 (13.29/13.30). 프리셋 'smps_operating' 또는 "
                         "JSON. 복합이 쓰는 동작점이 단독 녹화와 어긋난 SMPS 셋을 덮는다 — "
                         "**--sp-curves 와 같이 쓸 것**")
    ap.add_argument("--couple-ext", action="store_true",
                    help="결합 델타의 Σ 에 **비SMPS 전류**를 넣는다 (13.45). 지금은 SMPS 3종만 "
                         "더해서 오븐 5.2A·에어컨 h3 1.44A 가 빠져 있고, D 에서 그 강하가 "
                         "V_h3 자체보다 크다")
    ap.add_argument("--carrier-on", nargs="*", default=None, metavar="APP",
                    help="캐리어 상태를 **세션으로** 본다 (13.40). 인자 없이 주면 오븐. "
                         "오븐은 FAN_LIGHT 를 거쳐 켜지고 통전이 아니면 FAN_LIGHT 이고 "
                         "꺼질 때도 거치는데, 라벨은 HEATING 만 ON 이라 채점(세션 통째)과 "
                         "어긋난다 — 오븐 ON 창의 절반이 자동 오답이었다. "
                         "**핫플은 넣지 말 것** (ARMED_IDLE 0.55W < 계측 바닥)")
    ap.add_argument("--state-mix", default="",
                    help="창을 자를 때 **상태**를 먼저 뽑는다 (13.35). 프리셋 "
                         "'minipc_balanced'/'smps_balanced' 또는 JSON {가전:{상태id:확률}}. "
                         "전력 균등 계층화(12.34.6)가 좁은 상태를 과소 노출한다 — "
                         "미니PC IDLE 은 8.8~12.0W 로 좁아 17.2%% 만 나오는데 "
                         "**실측 복합은 IDLE 로만 돈다**")
    ap.add_argument("--sp-per-texture", action="store_true",
                    help="s(p) 를 **그 녹화의 텍스처**에서 만든 곡선으로 (13.74). 옛 곡선은 "
                         "깨끗한 정현파에서 만들어 자리 차이가 원리적으로 없었다 — 실측 채점에서 "
                         "크기 오차 중앙값 0.123 -> 0.031 (자리 D 는 0.211 -> 0.049). "
                         "--sp-curves 와 같이 써야 하고 processed_data/sp_curves_tex.npz 가 필요하다")
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
    ap.add_argument("--dither-min-order", type=int, default=2, metavar="K",
                    help="차수 비례 지터를 이 차수부터 건다 (12.182). 장소 위상 회전은 h7 "
                         "이상인데 60° 지터가 h3 에 20° 를 줘 φ3 판별자를 지웠다 "
                         "(cnn_ph 미니PC 0.950 -> 0.872). 예: 5")
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
    lvs = None
    if a.level_scramble:
        from src.synthesis.augmentor import LEVEL_SCRAMBLE_PRESETS
        lvs = (LEVEL_SCRAMBLE_PRESETS[a.level_scramble]
               if a.level_scramble in LEVEL_SCRAMBLE_PRESETS else json.loads(a.level_scramble))
        lvs = {k: tuple(v) for k, v in lvs.items()}
        print(f"  ** 기기별 전력 **범위 표집** '{a.level_scramble}' (13.29/13.30): "
              + ", ".join(f"{k}={v[0]:g}~{v[1]:g}x" for k, v in sorted(lvs.items())) + " **")
        if not a.sp_curves:
            print("  ⚠⚠ **--sp-curves 없이 범위 표집을 켰다.** 전력을 크게 옮기면서 전류를 "
                  "선형으로만 곱하게 되어 SMPS 의 고조파 모양이 틀린다 "
                  "(충전기 63->40W 에서 h7 30%, 미니PC h1 23%)")
    car = None
    if a.carrier_on is not None:
        car = tuple(a.carrier_on) if a.carrier_on else ("oven",)
        print(f"  ** 캐리어 상태를 세션으로 (13.40): {', '.join(car)} — "
              f"게이트는 세션, 전력은 그 순간 실제값 **")
    smx = None
    if a.state_mix:
        from src.synthesis.augmentor import STATE_MIX_PRESETS
        smx = (STATE_MIX_PRESETS[a.state_mix]
               if a.state_mix in STATE_MIX_PRESETS else json.loads(a.state_mix))
        smx = {k: {int(s): float(v) for s, v in m.items()} for k, m in smx.items()}
        print(f"  ** 상태 계층 표집 '{a.state_mix}' (13.35): "
              + ", ".join(f"{k}=" + "/".join(f"s{s}:{v:.0%}" for s, v in sorted(m.items()))
                          for k, m in sorted(smx.items())) + " **")
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
              f"(홀수차 중앙값, 차수 비례"
              + (f", h{a.dither_min_order} 부터" if a.dither_min_order > 2 else "") + ") **")
    build_cache(out_dir=a.out, n_windows=a.windows, window_cycles=a.window_cycles,
                time_split=a.split, seed=a.seed, n_workers=a.workers,
                exclude_activation_files=excl,
                dither_amp=a.dither_amp, dither_phase_deg=a.dither_phase_deg,
                recipe_mix=mix,
                dither_even_amp=a.dither_even_amp,
                dither_even_phase_deg=a.dither_even_phase_deg,
                power_scale_std_map=pss, level_scramble=lvs, state_mix=smx,
                carrier_apps=car, couple_ext=a.couple_ext,
                sp_curves=a.sp_curves, sp_per_texture=a.sp_per_texture, background=a.background,
                dither_min_order=a.dither_min_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
