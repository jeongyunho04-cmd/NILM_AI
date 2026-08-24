운영 조합 (adapt_ph1 + cnn_ov1) 을 stride 6 = 0.1초 간격으로 그린 것.

  python -m src.run_plot_timeline --ckpt results/adapt_ph1.pt \
      --ckpt-smps results/cnn_ov1.pt --stride 6 \
      --out results/plots_operating_stride6

핫플 릴레이 펄스가 1.0~1.3초이므로 기본 stride 15(0.25초)보다 0.1초가 정직하다
(설계 문서 12.52 — 채점에서도 같은 이유로 앞으로채움 -> 최근접으로 고쳤다).

디렉터리 이름 주의: 이전 이름 plots_op6s / plots_op_s6 은 각각 "6초 룩어헤드" 와
"stride 6" 을 뜻하는 한 글자 차이였다. 혼동해서 이름을 바꿨다.
