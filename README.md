해당 README는 AI로 작성한 문서입니다.

# BEMS 전력 예측 모델

건물 에너지 관리를 위한 GRU 기반 전력 소비 예측 시스템

## 프로젝트 개요

15분 간격 전력 데이터를 활용한 시계열 예측 모델로, 
최대 전력 소비량을 예측하여 효율적인 에너지 관리를 지원합니다.

## 주요 기능

- ⚡ 최대 전력 소비량 예측
- 📊 15분 단위 시계열 데이터 처리
- 🔄 실시간 예측 API 제공

## 기술 스택

- **Framework**: PyTorch
- **Model**: GRU (Gated Recurrent Unit)
- **API**: FastAPI
- **Database**: PostgreSQL
- **Deployment**: Docker

## 설치 방법

```bash
# 1. 저장소 클론
git clone [repository-url]

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치# BEMS 전력 예측 모델

건물 에너지 관리를 위한 GRU 기반 전력 소비 예측 시스템

## 프로젝트 개요

15분 간격 전력 데이터를 활용한 시계열 예측 모델로, 
최대 전력 소비량을 예측하여 효율적인 에너지 관리를 지원합니다.

## 주요 기능

- ⚡ 최대 전력 소비량 예측
- 📊 15분 단위 시계열 데이터 처리
- 🔄 실시간 예측 API 제공

## 기술 스택

- **Framework**: PyTorch
- **Model**: GRU 
- **API**: FastAPI
- **Database**: PostgreSQL
- **Deployment**: Docker


## 설치 방법
```bash
# 1. 저장소 클론
git clone [repository-url]

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

```bash
## 사용 방법

### 모델 학습

python train.py --epochs 200 --batch_size 32
```

### 추론 실행
```bash
python inference.py --input data/test.csv
```

### API 서버 실행
```bash
uvicorn main:app --reload
```

## 모델 성능

| 지표 | 값 |
|------|------|
| Test MAPE | 23.98% |
| R² Score | 0.8734 |
| RMSE | 0.6071 |
| MAE | 0.4033 |

## 디렉토리 구조
pip install -r requirements.txt
```
