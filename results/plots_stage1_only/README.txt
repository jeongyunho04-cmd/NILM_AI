1단계 단독 (cnn_ov1) 을 그린 것. **운영 조합이 아니다.**

  python -m src.run_plot_timeline --ckpt results/cnn_ov1.pt \
      --out results/plots_stage1_only

test_4 를 plots_operating_stride6 의 같은 파일과 나란히 보면
2단계 무라벨 적응이 무엇을 고치는지 보인다 —
여기서는 전기포트(초록)가 얹혀 예측 합계가 관측을 800~1000W 넘어서고,
운영 조합에서는 그 유령이 사라진다 (설계 문서 12.41.3, 인수인계 0절).
