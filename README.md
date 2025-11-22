# BEMS 전력 소비 예측 GRU 모델

[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.5-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

BEMS(Building Energy Management System) 전력 데이터를 활용한 시계열 예측 프로젝트입니다. GRU(Gated Recurrent Unit) 모델을 사용하여 15분 단위의 전력 소비를 예측합니다.

## 📊 프로젝트 개요

### 배경 및 목적
건물 에너지 관리 시스템(BEMS)에서 전력 소비를 정확히 예측
- ⚡ **에너지 비용 절감**: 피크 시간대 전력 사용 최적화
- 🌱 **탄소 배출 감소**: 효율적인 에너지 관리
- 📊 **운영 효율성 향상**: 사전 예방적 설비 관리

### 핵심 성과
- 📈 **R² Score 0.87**: 전력 소비 변동의 87%를 모델이 설명
- ⚡ **단기 예측 우수**: 15~30분 예측에서 MAE 0.02kW 이하 달성
- 🔧 **체계적 전처리**: 581개 결측치를 3단계 복합적인 방식으로 처리

### 기술 스택
- **데이터**: 15분 간격의 전력 소비 데이터
- **모델**: PyTorch 기반 3-layer GRU
- **예측**: 과거 96개 시퀀스(24시간) → 다음 15분 예측

## 🎯 담당 역할

본 프로젝트에서 다여 학습
   - 평균보다 2.3배 이상 높은 시간을 자동으로 피크로 판단

2. **풍부한 특성 엔지니어링**
   - 순환 인코딩으로 시간의 주기성 표현
   - 다양한 지연 특성으로 시간적 의존성 포착
   - 차분 및 통계 특성으로 변화율과 변동성 반영

3. **체계적인 결측치 처리**
   - 결측 구간 길이에 따른 적응적 보간 방법 적용
   - 장기 결측에 패턴 기반 복원으로 데이터 품질 향상

4. **재현 가능한 설정 관리**
   - YAML/JSON 기반 설정 파일로 실험 추적 용이
   - 버전 관리 및 하이퍼파라미터 히스토리 관리

## 🔍 향후 개선 방향

- [ ] Attention 메커니즘 추가로 장기 의존성 강화
- [ ] 실시간 데이터 연동으로 최적 모델 제작 및 지속적 예측 개선


## 👤 Author

**정동인 (Dongin Jung)**
- Role: AI Engineer Intern @ UCUBE
- Email: jde577776@gmail.com
- GitHub: [github.com/DOrigin1202](https://github.com/DOrigin1202)
- Notion: https://www.notion.so/2a986a6d9e1c80b79c16ee766de35773

## 📄 License


© 2025 UCUBE. All rights reserved.

This project is proprietary and confidential.

---

⚡ **Built with PyTorch for Smart Energy Management**
