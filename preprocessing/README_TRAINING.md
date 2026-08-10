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

## 📊 학습 결과 및 결과물 안내

학습이 시작되면 진행률 바(`tqdm`)와 함께 RTX 5070 GPU 가속으로 **약 1~3분 내로 20 에폭 학습이 완료**됩니다.

학습 완료 후 `./checkpoint_unet/` 폴더에 아래 파일들이 자동 생성됩니다:
1. `best_unet_nilm.pth`: **최고 정확도를 기록한 1D-UNet 가중치 모델 파일**
2. `scaler.pkl`: **입력 피처 정규화 스케일러** (실전 추론/분해 시 필수)
3. `loss_curve.png`: **학습 및 검증 손실 그래프**
