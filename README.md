# Depth vs Width: ResNet vs Wide Residual Networks 비교 연구

이 프로젝트는 딥러닝에서 네트워크의 **깊이(Depth)**와 **너비(Width)** 중 어느 것이 성능 향상에 더 효율적인지 비교 분석하는 연구입니다. ResNet-110 (Deep & Thin)과 WRN-28-2 (Wide & Shallow) 모델을 CIFAR-10 데이터셋에서 학습하고 평가합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 특징](#주요-특징)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [실험 결과](#실험-결과)
- [참고 문헌](#참고-문헌)

## 🎯 프로젝트 개요

이 프로젝트는 다음 연구 질문에 답하기 위해 설계되었습니다:

> **"CNN 성능 향상을 위해 모델의 깊이(Depth)와 너비(Width) 중 어느 것이 더 효율적인가?"**

### 비교 모델

1. **ResNet-110** (Baseline)
   - Deep & Thin 구조
   - 110 layers, widening factor k=1
   - 약 1.73M 파라미터

2. **WRN-28-2**
   - Wide & Shallow 구조
   - 28 layers, widening factor k=2
   - 약 1.47M 파라미터

3. **WRN-28-2-Dropout**
   - WRN-28-2에 Dropout(0.3) 적용
   - 일반화 성능 향상을 위한 변형 모델

## ✨ 주요 특징

- **공정한 비교**: 유사한 파라미터 예산(약 1.5M~1.7M) 내에서 모델 비교
- **표준화된 실험 설정**: WRN 논문의 표준 하이퍼파라미터 사용
- **완전한 재현성**: 고정된 random seed(42) 사용
- **상세한 분석**: 학습 곡선, 클래스별 성능, 효율성 분석 포함

## 🚀 설치 방법

### 요구사항

- Python 3.7 이상
- CUDA 지원 GPU (권장, CPU도 가능하지만 느림)

### 1. 저장소 클론

```bash
git clone <repository-url>
cd WRN
```

### 2. 가상환경 생성 및 활성화 (권장)

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. GPU 지원 확인 (선택사항)

PyTorch가 GPU를 인식하는지 확인:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## 📖 사용 방법

### Jupyter Notebook 실행

1. Jupyter Notebook 실행:

```bash
jupyter notebook
```

2. `depth_or_width.ipynb` 파일을 열고 순서대로 셀을 실행합니다.

### 노트북 구조

노트북은 다음 섹션으로 구성되어 있습니다:

1. **Problem Definition**: ResNet과 WRN의 이론적 배경
2. **Dataset**: CIFAR-10 데이터셋 로드 및 전처리
3. **Model Design & Implementation**: 
   - ResNet-110 구현
   - WRN-28-2 구현
4. **Training**: 모델 학습 (200 epochs)
5. **Testing**: 테스트 세트에서 평가
6. **Result Analysis**: 
   - 학습 곡선 시각화
   - 성능 비교
   - 클래스별 성능 분석
   - 효율성 분석

### 학습된 모델 사용

학습이 완료되면 `checkpoints/` 디렉토리에 모델이 저장됩니다:

```
checkpoints/
├── resnet110/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
├── wrn28_2/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
└── wrn28_2_dropout/
    ├── best_model.pth
    ├── final_model.pth
    └── training_history.json
```

모델 로드 예시:

```python
import torch
from model import ResNet110  # 또는 WRN28_2

# 모델 인스턴스 생성
model = ResNet110(num_classes=10)

# 체크포인트 로드
checkpoint = torch.load('./checkpoints/resnet110/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📁 프로젝트 구조

```
WRN/
├── depth_or_width.ipynb          # 메인 노트북 파일
├── requirements.txt               # Python 패키지 의존성
├── README.md                      # 프로젝트 설명서
├── data/                          # CIFAR-10 데이터셋 (자동 다운로드)
│   └── cifar-10-batches-py/
├── checkpoints/                   # 학습된 모델 체크포인트
│   ├── resnet110/
│   ├── wrn28_2/
│   └── wrn28_2_dropout/
└── test_results_summary.json      # 테스트 결과 요약
```

## 📊 실험 결과

### 주요 결과 요약

| 모델 | 파라미터 수 | 깊이 | 너비(k) | 테스트 정확도 | 테스트 Loss |
|------|------------|------|---------|--------------|-------------|
| ResNet-110 | 1,730,522 | 110 | 1 | 94.29% | 0.2656 |
| WRN-28-2 | 1,467,610 | 28 | 2 | 94.62% | 0.2151 |
| WRN-28-2-Dropout | 1,467,610 | 28 | 2 | **94.75%** | **0.1989** |

### 주요 발견

1. **너비 확장의 효율성**: WRN-28-2는 ResNet-110보다 약 15% 적은 파라미터로 더 높은 성능을 달성
2. **Dropout의 효과**: WRN-28-2-Dropout이 가장 높은 테스트 정확도와 가장 낮은 테스트 Loss를 기록
3. **효율성**: WRN 시리즈 모델이 ResNet-110보다 약 18% 높은 효율성(정확도/파라미터)을 보임

## 🔧 하이퍼파라미터

### 학습 설정

- **Optimizer**: SGD with Momentum
- **Initial Learning Rate**: 0.1
- **Momentum**: 0.9
- **Weight Decay**: 0.0005
- **Learning Rate Schedule**: Multi-step decay
  - Epoch 60: 0.1 → 0.02
  - Epoch 120: 0.02 → 0.004
  - Epoch 160: 0.004 → 0.0008
- **Batch Size**: 128
- **Total Epochs**: 200
- **Random Seed**: 42

### 데이터 증강

- **Training**: 4-pixel padding + Random horizontal flip + Random crop
- **Validation/Test**: Normalization only

## 🐛 문제 해결

### CUDA out of memory 오류

배치 크기를 줄이거나 GPU 메모리를 확인하세요:

```python
# 배치 크기 조정 (예: 128 → 64)
batch_size = 64
```

### 데이터셋 다운로드 실패

인터넷 연결을 확인하거나 수동으로 CIFAR-10 데이터셋을 다운로드하여 `data/` 디렉토리에 배치하세요.

## 📚 참고 문헌

1. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [DOI](
https://doi.org/10.48550/arXiv.1512.03385)

2. Zagoruyko, Sergey, and Nikos Komodakis. "Wide Residual Networks." Proceedings of the British Machine Vision Conference (BMVC). 2016. [DOI]()
https://doi.org/10.48550/arXiv.1605.07146 [GitHub](https://github.com/szagoruyko/wide-residual-networks)

3. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
