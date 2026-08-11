# 1D-SimCLR (자기지도 대조 학습) NILM AI 사용 가이드

이 디렉토리는 1D 전력/고조파 파형을 기반으로 **자기지도 대조 학습(Self-Supervised Contrastive Learning)**을 수행하고, 라벨 없이도 가전제품별 고유 전기 지문 지도(Latent Cluster Map) 구축 및 미등록 신규 가전 접속 자동 감지(Novelty Detection)를 수행하는 1D-SimCLR 모듈입니다.

---

## 📁 디렉토리 파일 구성

```
D:\stm project\NILM_AI\ssl_simclr\
├── config_simclr.yaml          # 하이퍼파라미터 (온도 tau, 배치 크기, 증강 설정)
├── augmentations_1d.py         # 1D 물리 파형 데이터 증강 (전압 sag/swell, 노이즈, 위상 이동, 크롭)
├── simclr_model.py             # 1D 인코더 f(·) 백본 + MLP 투영 헤드 g(·)
├── loss_infonce.py             # InfoNCE (NT-Xent) 코사인 유사도 대조 손실 연산
├── dataset_simclr.py           # 무작위 물리 증강 쌍 (x_A, x_B) 동적 생성 PyTorch Dataset
├── train_simclr.py             # PyTorch AMP FP16 사전 학습 메인 루프
├── fingerprint_analyzer.py     # 전기 지문 지도 시각화 (t-SNE/UMAP) 및 신규 가전 자율 감지 엔진
├── fine_tune_unet.py           # 대조 학습 인코더 기반 Downstream 분해 성능 검증 스크립트
└── README_SIMCLR.md            # 본 사용 가이드 문서
```

---

## 🚀 실행 방법 (RTX 5070 GPU 권장)

### 1단계: 1D-SimCLR 자기지도 사전 학습 실행
라벨 없는 전력 파형으로 전기 지문 인코더를 학습합니다. (NVIDIA RTX 5070 기준 약 26~30분 소요)

```bash
python train_simclr.py -c config_simclr.yaml -e 50 -b 512
```

- 학습 결과물은 `checkpoint_simclr/best_simclr_encoder.pth`로 자동 저장됩니다.
- 학습 손실 곡선 그래프는 `checkpoint_simclr/simclr_loss_curve.png`로 저장됩니다.

---

### 2단계: 가전별 전기 지문 지도 시각화 및 신규 가전 자동 감지 테스트
학습된 전기 지문 클러스터를 t-SNE / UMAP 2차원 산점도로 시각화하고, 집 안에 완전히 새로운 가전(예: 로봇청소기 80W)이 접속했을 때 0초 만에 미등록 가전임을 자율 감지(Novelty Detection)하는 테스트를 수행합니다.

```bash
python fingerprint_analyzer.py -c config_simclr.yaml -method tsne
```

- **출력 결과**:
  - `checkpoint_simclr/fingerprint_map_tsne.png` (가전별 지문 클러스터 지도)
  - 콘솔 출력: `[NOVELTY DETECTED] Unregistered New Appliance!` 메시지 및 전기적 부하 유형(저항성, SMPS, 인버터 모터) 추정 결과 출력

---

### 3단계: Downstream 분해 성능 검증 (선택 사항)
사전 학습된 SimCLR 인코더를 UNet 디코더에 결합하여 기존 RTX 5070 가중치(`best_unet_nilm.pth`)와 결합하거나 분해 정확도를 측정합니다.

```bash
python fine_tune_unet.py -c config_simclr.yaml
```

---

## 💡 주요 하이퍼파라미터 설정 (`config_simclr.yaml`)

```yaml
simclr:
  window_len: 320             # 윈도우 길이 (320 주기 ~ 5.33초)
  temperature: 0.07           # InfoNCE 손실 함수 온도 파라미터 tau
  embedding_dim: 512          # 인코더 지문 벡터 차원 h
  projection_dim: 128         # 투영 헤드 벡터 차원 z

augmentations:
  voltage_jitter_pct: 0.03    # 계통 전압 흔들림 (+/- 3%)
  noise_snr_db: 30.0          # 선 노이즈 주입 (SNR 30dB)
  phase_shift_deg: 3.0        # 위상 미세 이동 (+/- 3도)
  max_crop_pct: 0.15          # 시간 축 무작위 자르기 (최대 15%)
```
