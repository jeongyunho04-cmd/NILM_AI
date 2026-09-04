# 판정 기준 — (나) 1단계 위상 지터: 모델이 h7 이상 위상을 믿지 않게 (12.179.7 의 (나), 12.182)

**결과를 보기 전에 적는다** (2026-09-04, 규칙 33/48).

## 무엇을 바꾸나 — 하나

캐시(와 홀드아웃)의 차수별 지터에 **위상**을 켠다: `--dither-amp 0.5 --dither-phase-deg 60`
(`adapt_ac` 계보는 amp 0.2, phase 0). `σ_k = x·k/9` 라
```
        h3    h5    h7    h9    h11   h13   h15
위상 σ  20°   33°   47°   60°   73°   87°   100°     장소 B ∠T: h9 66° / h11 105° / h13 89° / h15 48°
진폭 σ  0.17  0.28  0.39  0.50  0.61  0.72  0.83     장소 B |T|: 0.7 ~ 3.2 (= e^−0.34 ~ e^1.15)
```
활성화당 1회 뽑으므로 창 안의 기기마다 다른 회전이 들어간다 — 12.181.4 의 "기기별 T" 다.
레시피(smps_hi_fix)·s(p)·1단계 인수·2단계 레시피는 `cnn_ac`/`adapt_ac` 와 같다. `--site-transfer` 없음.

## 예측

12.181 이 보인 것: 유령·시드는 h2~h9 위상만 되돌려도 살아나고, 미니PC 신원은 h11~h15 인데
그 전달비가 기기별이다. 위상 지터가 그 자리를 덮으면 모델은 (i) 저항 배분을 SMPS 위상에
안 기대고, (ii) 미니PC 를 h11~h15 의 절대 위상이 아니라 다른 것(크기 비율, k, 저차 위상)으로
가려야 한다. (ii) 가 가능한지는 모른다 — 12.34 가 φ3 를 넣은 이유가 크기 비율은 SMPS 3종이
겹쳐서였다. 그래서 ④ 를 잰다.

```
① 미니PC IDLE 전력 (test_15/18 중앙, run_minipc_idle_probe)   1.31W -> **4W 이상**
   **그리고** test_16/17(미니PC 없음) 미니PC 유령 < 1W, 장소B 미니PC 정밀도 > 0.6  (흡수가 아니라 검출)
② 388~414s 세 시드 드라 게이트 > 0.9 / AC < 0.1  (run_ghost_window_probe --range 388,414)
   s0 test_15 유령 33.75 -> 5W 미만
③ 지킬 것 (수정된 site_table, 병합 채점): 장소A 전체F1 0.925~0.931 · 포트A 0.92~0.95 · 프로젝터A 0.92
           장소B 드라이기 0.93 · 포트B 0.955~0.985 · 장소B 전체 0.89~0.915
④ 대가: 합성 홀드아웃 SMPS 3종 F1 과 전체 F1 — cnn_ac 의 0.9706 과 **같은 holdout60_ac** 에서 비교.
   (홀드아웃 생성기는 지터를 걸지 않는다 — `run_build_holdout` 에 dither 인수가 없고 `holdout.py` 가
   `DataAugmentor(level_scramble=...)` 만 만든다. 레시피·s(p) 가 같으므로 holdout60_ac 를 그대로 쓴다.
   규칙 63 의 "같은 설정" 이 지터 없는 쪽에서 성립한다.)
```

## 통과/실패의 뜻

```
①②③ 통과            -> 채택. 운영점 adapt_ph_s* (시드는 ③ 로 고른다). 장소 보정 절차 불필요.
② 통과, ① 실패       -> 위상 지터는 저항 배분을 살리지만 SMPS 신원은 못 만든다. 미니PC 는 12.171.4 의
                       차분 항(L_step) 이나 별도 판별자로. 유령·시드만이라도 채택할지는 ③ 으로 판단.
③ 무너짐 (장소A -0.02 이상, 포트B < 0.95)
                    -> 지터가 판별 정보를 지웠다. 위상 지터를 h5 부터(k_min) 걸거나 ph 를 30 으로 내려 재시도.
④ 합성 SMPS F1 이 0.05 이상 내려감
                    -> 같은 뜻. 합성에서부터 못 가르면 실측은 볼 것도 없다.
```

## 실행

```
python -m src.run_build_traincache --out cache/train60_ph --recipe-mix smps_hi_fix \
  --dither-amp 0.5 --dither-phase-deg 60 --sp-curves                       # ~20분
python -m src.run_build_holdout --out processed_data/holdout60_ph (같은 지터·--sp-curves)   # 규칙 63
python -m src.run_train_cnn --cache cache/train60_ph --holdout processed_data/holdout60_ph \
  --epochs 300 --epoch-windows 50000 --batch 512 --w-over 0 --eval-every 10 \
  --harm-odd-only --standby-operating session --fine-channels 52 --tag cnn_ph   # ~27분
python -m src.run_adapt --init results/cnn_ph.pt --cache cache/train60_ph \
  --steps 1000 --batch 256 --lr 3e-5 --lam 0.5 --w-cons 0.1 --w-harm 4.0 --w-hedge 0.2 \
  --harm-weight inv_h2 --harm-offset results/norton_coef.npz \
  --w-res 0.1 --res-apps electiric_kettle,oven,hair_dryer,hotplate \
  --w-swap 2.0 --swap-tol 0.02 --swap-slack 1 --smps-boost 4.0 --real-stride 30 --eval-every 250 \
  --seed {0,1,2} --tag adapt_ph_s{0,1,2} --out results                        # 3시드 9분
python -m src.run_gate_check --ckpt results/adapt_ph_s0.pt results/adapt_ph_s1.pt results/adapt_ph_s2.pt \
  --events processed_data/real_events_refined.json --session-merge --postproc off --out results/_gcsm_ph.json
python -m src.run_site_table results/_gcsm_ac.json results/_gcsm_ph.json --site both --absent
python -m src.run_minipc_idle_probe --ckpt results/adapt_ph_s*.pt
python -m src.run_ghost_window_probe --stem test_15 --range 388,414 --ckpt results/adapt_ph_s*.pt
python -m src.run_score_holdout --ckpt results/cnn_ph.pt results/cnn_ac.pt --holdout processed_data/holdout60_ph processed_data/holdout60_ac   # ④ 2x2
```
