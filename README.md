# shelter-dog-survival-prediction
유기견 생존 예측 멀티모달 AI (순종견 편향 제거)
# 🐕 유기견 생존 예측 AI: 순종견 편향 제거 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **멀티모달 딥러닝**으로 유기견 생존 확률을 예측하고, **순종견 편향을 98.7% 감소**시킨 AI 프로젝트

![Project Overview](docs/images/project_overview.png)

---

## 🎯 프로젝트 목표

1. **높은 정확도**: 유기견 생존 예측 83.3% 달성
2. **순종견 편향 제거**: XGBoost 대비 98.7% 감소
3. **멀티모달 AI**: 이미지 + 표 데이터 결합

---

## 🏆 주요 성과

| 지표 | XGBoost | CNN | **멀티모달** |
|------|---------|-----|-------------|
| **정확도** | 75.1% | 63.2% | **83.3%** ⭐ |
| **is_mixed 순위** | 1위 (25.0%) | - | **65위 (0.31%)** |
| **순종견 편향** | 매우 심각 ❌ | - | **98.7% 감소** ✅ |

---

## 📊 모델 아키텍처
```
┌─────────────────────────────────────────────────┐
│                   입력 데이터                    │
├──────────────────┬──────────────────────────────┤
│   이미지 (224×224) │  표 데이터 (8 features)    │
│                  │                              │
│  EfficientNet-B0 │  Feature Engineering         │
│  (사전학습 모델)   │  - is_mixed                 │
│       ↓          │  - age, weight              │
│  CNN Features    │  - health_score             │
│  (1280차원)       │  - care_encoded             │
└──────────┬───────┴──────────┬───────────────────┘
           │                  │
           └─────────┬─────────┘
                     ↓
              Feature Fusion
                (1288차원)
                     ↓
                 XGBoost
              (Multimodal)
                     ↓
           ┌─────────────────┐
           │  예측 결과       │
           │  0: 생존        │
           │  1: 자연사      │
           │  2: 안락사      │
           └─────────────────┘
```

---

## 🚀 빠른 시작

### 설치
```bash
# 1. Repository 클론
git clone https://github.com/your-username/shelter-dog-survival-prediction.git
cd shelter-dog-survival-prediction

# 2. 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt
```

### 모델 다운로드

학습된 모델 파일이 너무 커서 GitHub에 직접 올릴 수 없는 경우:
```bash
# Google Drive 또는 Dropbox 링크
wget https://drive.google.com/... -O models/best_cnn_model.pth
wget https://drive.google.com/... -O models/multimodal_model.pkl
```

### 추론 (예측)
```python
from src.multimodal_model import MultimodalPredictor

# 모델 로드
predictor = MultimodalPredictor(
    cnn_path='models/best_cnn_model.pth',
    multimodal_path='models/multimodal_model.pkl'
)

# 예측
result = predictor.predict(
    image_path='path/to/dog_image.jpg',
    age=2,
    weight=10,
    is_mixed=1
)

print(f"예측: {result['class']}")  # 생존/자연사/안락사
print(f"확률: {result['probability']:.2%}")
```

---

## 📖 상세 문서

- [📐 모델 아키텍처](docs/architecture.md)
- [🎓 학습 과정](docs/training_process.md)
- [📊 성능 비교](docs/performance_comparison.md)
- [⚖️ 편향 분석](docs/bias_analysis.md)

---

## 📁 프로젝트 구조
```
shelter-dog-survival-prediction/
├── data/                  # 데이터
├── models/                # 학습된 모델
├── notebooks/             # Jupyter 노트북
├── src/                   # 소스 코드
├── docs/                  # 기술 문서
└── results/               # 결과물
```

---

## 🛠️ 기술 스택

- **딥러닝**: PyTorch, torchvision
- **머신러닝**: XGBoost, scikit-learn
- **데이터**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **이미지**: PIL, OpenCV

---

## 📈 실험 결과

### 성능 비교

![Performance Comparison](results/performance_comparison.png)

### 순종견 편향 분석

![Bias Analysis](results/bias_analysis.png)

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

## 💡 주요 발견

1. **이미지가 핵심**: CNN이 99% 기여
2. **순종견 편향 제거**: is_mixed 1위 → 65위
3. **공정한 예측**: 순종견/믹스견 차이 11.7%p (XGBoost 대비 크게 감소)

---

## 📝 인용

이 프로젝트를 사용하시는 경우 아래와 같이 인용해주세요:
```bibtex
@misc{ji2026shelter,
  author = {Sunghyun Ji},
  title = {Shelter Dog Survival Prediction with Bias Reduction},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/shelter-dog-survival-prediction}
}
```

---

## 📜 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

## 🙏 감사의 글

- 데이터 제공: 농림축산식품부 동물보호관리시스템
- 영감: 보호소 봉사 경험

---

## 🔗 관련 링크

- [Jupyter Notebooks](notebooks/)
- [Technical Documentation](docs/)
- [Model Files](models/)

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
