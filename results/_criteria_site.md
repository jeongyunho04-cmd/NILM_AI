# 판정 기준 — 장소 전달비 입력 보정 + 재적응 (12.179.7 의 (가), 12.181)

**결과를 보기 전에 적는다** (2026-09-04, 규칙 33/48).

## 무엇을 바꾸나 — 하나

실측 33채널을 만들 때 Re/Im h2~h15 를 장소 전달비 `T_h` 로 나눈다
(`results/_site_transfer_TB.npy`, 충전기·프로젝터 단일 창의 실측/합성 페이저 비).
`L_harm` 의 `obs_harm` 도 같은 프레임으로 나눈다 (지문 `sig` 가 장소 A 프레임이므로).
장소 B 파일(test_15~18)에만 건다. 장소 A 는 `T_A ~= 1` 이라 안 건다.

레시피·1단계·캐시·시드는 `adapt_ac_*` 와 **완전히 같다.** 바뀐 것은 실측 입력 한 줄이다.

## 예측 (기전에서 나온다 — 12.179.5 / 12.180.2)

`cnn_ac` 만으로 보정 입력에서 미니PC p_raw 10.5W, 드라이기 강풍 게이트 1.000,
겹침 창 AC 0.000 을 이미 낸다. 적응은 거기서 출발하므로:

```
① 미니PC IDLE 전력 (test_15/18 창 중앙, run_minipc_idle_probe)   1.31W -> **4W 이상**   (참 ~10~12)
   ⚠ 보정만으로는 게이트가 0.26~0.37 이다. 재적응이 게이트를 열어야 통과한다. 못 열면
     "p_raw 는 맞고 게이트가 남았다" 로 적고, 그 다음은 미니PC 자신의 전달비다.
② 장소 B 드라이기 F1 0.928~0.939 유지, 포트 0.955~0.985 유지  (병합 채점, 수정된 site_table)
③ s0 의 test_15 유령 33.75W -> **5W 미만**, 그리고 388~414초 창에서 **세 시드가 같아야** 한다
   (run_ghost_window_probe --range 388,414: 드라 게이트 > 0.9, AC < 0.1, 세 시드 전부)
④ 장소 A 전체F1 0.925~0.931 · 포트 0.92~0.95 · 프로젝터 0.92 유지 (보정을 안 거는 파일이라
   변하면 안 된다 — 변하면 2단계가 장소 B 창을 통해 A 를 바꾼 것이다)
```

## 통과/실패의 뜻

```
①③ 통과, ②④ 유지     -> 채택. 운영점을 adapt_site_s* 로 바꾸고 run_live 가 ckpt 의 T 를 쓴다.
                        시연 장소에서는 30초 단일 SMPS 녹화로 T 를 다시 잰다.
③ 통과, ① 실패        -> 유령·시드는 고쳤고 미니PC 게이트는 따로다. 12.179.5 의 "미니PC 자신의 T" 로.
③ 실패                -> 12.180 의 기전이 틀렸다. 1단계가 그 창을 맞히는데도 적응이 틀어지는 것이면
                        적응 항(L_harm 의 AC 처리)을 다시 본다.
④ 무너짐              -> 장소 B 만 보정한 것이 2단계 분포를 갈랐다. 보정을 A 에도 걸어(T_A) 대조.
```

## 실행

```
python -m src.run_adapt --init results/cnn_ac.pt --cache cache/train60_ac \
  --steps 1000 --batch 256 --lr 3e-5 --lam 0.5 \
  --w-cons 0.1 --w-harm 4.0 --w-hedge 0.2 --harm-weight inv_h2 \
  --harm-offset results/norton_coef.npz \
  --w-res 0.1 --res-apps electiric_kettle,oven,hair_dryer,hotplate \
  --w-swap 2.0 --swap-tol 0.02 --swap-slack 1 \
  --smps-boost 4.0 --real-stride 30 --eval-every 250 \
  --site-transfer results/_site_transfer_TB.npy --site-transfer-stems test_15,test_16,test_17,test_18 \
  --seed {0,1,2} --tag adapt_site_s{0,1,2} --out results
python -m src.run_gate_check --ckpt results/adapt_site_s0.pt results/adapt_site_s1.pt results/adapt_site_s2.pt \
  --events processed_data/real_events_refined.json --session-merge --postproc off --out results/_gcsm_site.json
python -m src.run_site_table results/_gcsm_ac.json results/_gcsm_site.json --site both --absent
python -m src.run_minipc_idle_probe --ckpt results/adapt_site_s0.pt results/adapt_site_s1.pt results/adapt_site_s2.pt
python -m src.run_ghost_window_probe --stem test_15 --range 388,414 --ckpt results/adapt_site_s0.pt results/adapt_site_s1.pt results/adapt_site_s2.pt
```
채점기는 체크포인트에 저장된 `site_transfer` 를 읽어 같은 보정을 건다 (체크포인트가
자기 입력 프레임을 안다). `cnn_ac` 는 그 항이 없어 보정 없이 채점된다.
