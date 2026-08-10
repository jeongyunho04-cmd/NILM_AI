# NILM AI 1D-UNet Model Training Guide (RTX 5070 GPU Desktop PC)

이 가이드는 현재 노트북에서 만든 `NILM_AI` 프로젝트 폴더를 **NVIDIA RTX 5070 그래픽카드가 설치된 데스크톱 PC**로 옮겨서 **1D-UNet 딥러닝 모델을 초고속으로 학습(Training)시키는 가이드**입니다.

---

## 💻 데스크톱 PC (RTX 5070) 세팅 3단계

### 1단계: 프로젝트 폴더 이동
노트북의 `NILM_AI` 전체 폴더(또는 `preprocessing` 폴더)를 USB / 구글 드라이브 등을 이용해 **데스크톱 PC의 원하는 위치**(예: `D:\NILM_AI`)로 복사합니다.

### 2단계: Python 환경 및 PyTorch CUDA 전용 라이브러리 설치
데스크톱 PC에서 CMD(명령 프롬프트) 또는 PowerShell을 열고 프로젝트 폴더로 이동한 뒤 아래 명령어를 실행합니다:

```cmd
# 1. 일반 기본 패키지 설치
pip install -r requirements_gpu.txt

# 2. RTX 5070 CUDA GPU 가속 전용 PyTorch 설치 (필수!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3단계: GPU 가속 및 설치 확인
아래 명령어로 데스크톱 PC에서 RTX 5070 GPU가 정상 인식되는지 확인합니다:

```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
> `CUDA Available: True`, `Device Name: NVIDIA GeForce RTX 5070`이 출력되면 완료입니다!

---

## 🚀 1D-UNet AI 모델 학습 실행 방법

데스크톱 PC에서 아래 명령어를 입력하여 학습을 시작합니다:

```cmd
cd /d "D:\NILM_AI\preprocessing"

# 12시간 합성 데이터셋(synthetic_nilm_12h.csv) 기반 1D-UNet 학습 실행
python train_unet.py -data "./output/synthetic_nilm_12h.csv" -b 256 -e 20
```

### ⚙️ 주요 실행 옵션 (Command Line Arguments)
* `-data`: 학습에 사용할 합성 데이터셋 파일 경로 (기본값: `./output/synthetic_nilm_12h.csv`)
* `-b` (`--batch-size`): 배치 사이즈 (RTX 5070 최적 추천: `256`)
* `-e` (`--epochs`): 학습 에폭 수 (기본값: `20`)
* `-w` (`--window-len`): 슬라이딩 윈도우 크기 (기본값: `320` cycles $\approx$ 5.33초)
* `-o` (`--output-dir`): 모델 체크포인트 저장 폴더 (기본값: `./checkpoint_unet`)

---

## 🧪 1D-UNet 모델 성능 평가 & 분해 테스트 실행 방법

학습이 정상적으로 완료된 후, 데스크톱 PC에서 아래 명령어를 실행하여 **모델의 실제 분해 정확도(MAE, RMSE, R², F1-Score)를 평가하고 결과 그래프를 생성**합니다:

```cmd
python test_unet.py -data "./output/synthetic_nilm_12h.csv" -ckpt "./checkpoint_unet"
```

### 📊 출력되는 평가 지표 리포트 예시
* **MAE (Mean Absolute Error, 평균 절대 오차 W)**: 실제 전력과 AI 추정 전력 간의 평균 차이 ($W$)
* **R² Score (결정계수 %)**: AI가 각 가전제품의 전력 변화를 얼마나 정확히 설명했는지 (%)
* **F1-Score (%)**: 가전제품의 **켜짐/꺼짐(ON/OFF) 상태를 얼마나 정확히 판별**했는지 (%)
* **`disaggregation_result.png`**: 실제 전력 파형(파란색) vs 1D-UNet AI 추정 파형(빨간색 점선) 비교 시각화 그래프

