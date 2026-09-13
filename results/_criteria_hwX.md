# 판정 기준 — 오븐 standby 의 전력과 고조파를 맞춘다 (12.163)

**결과보다 먼저 적는다 (규칙 22). 시드 3개 (규칙 50).**

## 무엇이 어긋나 있나
합성은 오븐 activation 의 휴지 구간을 `gt_standby_p = net_power_features[0]`
= **15.02W (FAN_LIGHT)** 로 정확히 담는다. 그런데 `L_harm` 이 쓰는
`standby_sig` 는 `get_standby_profile` 이 주는 **OFF_STANDBY 의 6.44mA** 다.

```
                 전력            고조파
y_standby      15.0 W  ✓      —
standby_sig      —          **6.44 mA**  ✗   (OFF_STANDBY)
실제 FAN_LIGHT  15.0 W       **67.4 mA**       (activation 의 net 페이저)
```
**10배 어긋난다.** 전력은 15W 를 요구받는데 고조파는 그 1/10 만 설명하니
모델이 타협한다 — 실측에서 오븐 standby 예측이 **5.37W** 다 (참값 15W).

## 처방
`standby_sig[oven]` 을 activation 안 FAN_LIGHT(state 1) 구간의 **net 페이저**로
바꾼다. **합성이 실제로 넣는 값과 같은 자를 쓴다.** 손실 상수만 바꾸므로
2단계 재적응(3분)으로 확인된다.

## 자
```
hwO   대조   standby_sig = OFF_STANDBY   (있음, seed 0/1/2)
hwX   처방   standby_sig = FAN_LIGHT     seed 0/1/2
```

## ① 인과 — 이것이 안 되면 형태가 틀린 것이다
```
오븐 standby 예측 (장소 A, 오븐 세션 중 히터 off 창)   5.37W -> 12~16W
**안 움직이면 나머지가 좋아져도 채택하지 않는다**
```

## ② 사는 것
```
없는 기기의 standby 가 줄어야 한다 — 장소 A 에 에어컨·선풍기는 없는데
  지금 각각 1.24W / 0.53W 를 낸다. 오븐 몫을 나눠 가진 것으로 보인다
잔차 (그 창) 2.36W -> 줄거나 유지
```

## ③ 지켜야 하는 것 (3시드 중앙, 장소 B, 후처리 off)
```
드라이기 0.942  |  포트 0.994  |  미니PC 0.534  |  전체 F1 0.851  |  유령 6.56W
장소 A 전체 F1 0.813  |  장소 A 오븐 F1 0.569
```

## 미리 적어 두는 위험
```
- `standby_sig` 는 **상수 하나**다. 오븐이 완전히 꺼진 창(OFF_STANDBY, 6%)에는
  FAN_LIGHT 값이 과대다. FAN_LIGHT 이 activation 휴지의 95% 라 그쪽을 고른다.
- `idle = σ(plugged)(1−σ(on))` 인데 합성의 `gt_plugged` 는 꽂혀 있으면 항상 1 이다.
  오븐이 안 쓰이는 창에도 67mA 를 예측하게 되어 **유령이 늘 수 있다**.
  ③ 의 유령 6.56W 가 그 자다.
- 1단계는 안 바꾼다. 모델의 `standby` 머리는 이미 15W 로 감독받았으므로
  (`y_standby` 가 맞다) 2단계에서 고조파만 맞추면 될 것으로 본다. 아니면 1단계로.
