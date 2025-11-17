# AIEMS API

GRU 모델 학습 및 예측을 위한 FastAPI 서버입니다.

## 기능

- CSV 데이터 업로드
- GRU 모델 학습
- 시계열 예측
- 회귀분석 모델 생성
- 다중 변수 회귀분석
- 회귀분석 결과 예측
- 학습 히스토리 조회
- 모델 상태 모니터링
- **Modbus TCP 전력 데이터 수집**
- **실시간 전력량계 모니터링**
- **전력 데이터 저장 및 조회**
- **시계열 전력 데이터 분석**

## 실행 방법

### Docker Compose 사용 (권장)

```bash
# 서버 시작
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 서버 중지
docker-compose down
```

### Docker 직접 사용

```bash
# 이미지 빌드
docker build -t aidems-api .

# 컨테이너 실행
docker run -p 8000:8000 aidems-api
```

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 기본 엔드포인트

- `GET /` - 루트 엔드포인트
- `GET /health` - 헬스체크

### 데이터 관리

- `POST /api/data/upload` - 훈련 데이터 업로드 (CSV)

### 전력량계 관리

- `GET /api/power-meters` - 전력량계 목록 조회
- `GET /api/power-meters/{meter_id}` - 특정 전력량계 정보 조회

### 전력 데이터 관리

- `POST /api/power-data` - 전력 데이터 저장
- `POST /api/power-data/batch` - 전력 데이터 일괄 저장
- `GET /api/power-data/{meter_id}` - 전력 데이터 조회
- `GET /api/power-data/{meter_id}/latest` - 최신 전력 데이터 조회
- `GET /api/power-data/{meter_id}/stats` - 전력 데이터 통계
- `GET /api/power-data/{meter_id}/timeseries` - 시계열 전력 데이터
- `DELETE /api/power-data/{meter_id}` - 오래된 전력 데이터 삭제
- `GET /api/power-data/{meter_id}/trend` - 전력 소비량 트렌드 분석
- `GET /api/power-data/{meter_id}/quality` - 전력 품질 지표 분석



### GRU 모델 관련

- `POST /api/gru/train` - GRU 모델 학습
- `POST /api/gru/predict` - 예측 실행
- `POST /api/gru/predict-from-db` - 데이터베이스 기반 예측

### 회귀분석 관련

- `POST /api/regression/analyze` - 회귀분석 실행
- `GET /api/regression/results/{user_id}` - 회귀분석 결과 조회
- `GET /api/regression/predict/{user_id}` - 회귀분석 예측
- `GET /api/regression/models/{user_id}` - 사용 가능한 모델 목록
- `GET /api/regression/status/{user_id}` - 회귀분석 상태 조회
- `DELETE /api/regression/results/{user_id}` - 회귀분석 결과 삭제

### 스케줄러 관련

- `POST /api/scheduler/start` - 스케줄러 시작
- `POST /api/scheduler/stop` - 스케줄러 중지
- `GET /api/scheduler/status` - 스케줄러 상태 조회
- `POST /api/scheduler/manual-update` - 수동 주간 업데이트
- `GET /api/scheduler/active-users` - 활성 사용자 목록

### 모델 관리

- `GET /api/model/status` - 모델 상태 조회
- `POST /api/model/save` - 모델 저장
- `POST /api/model/load` - 모델 로드
- `DELETE /api/model/reset` - 모델 상태 초기화

## API 문서

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 사용 예시

#### 1. 데이터 업로드
```bash
curl -X POST "http://localhost:8000/api/data/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"
```

#### 2. 모델 학습
```bash
curl -X POST "http://localhost:8000/api/lstm/train" \
     -H "Content-Type: application/json" \
     -d '{
       "sequence_length": 10,
       "epochs": 100,
       "batch_size": 32,
       "learning_rate": 0.001
     }'
```

#### 3. 예측
```bash
curl -X POST "http://localhost:8000/api/lstm/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input_sequence": [1.0, 2.0, 3.0, 4.0, 5.0],
       "model_name": "lstm_model"
     }'
```

#### 4. 회귀분석 실행
```bash
curl -X POST "http://localhost:8000/api/regression/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "MBER_ID": "user123",
       "energy_sources": ["전력", "가스"]
     }'
```

#### 5. 회귀분석 결과 조회
```bash
curl -X GET "http://localhost:8000/api/regression/results/user123"
```

#### 6. 회귀분석 예측
```bash
curl -X GET "http://localhost:8000/api/regression/predict/user123?y_column=Y1&model_no=1&x_values=X1:25.5,X2:60.2,X3:15.8"
```

#### 7. 전력 소비량 트렌드 분석
```bash
curl -X GET "http://localhost:8000/api/power-data/1/trend?days=7"
```

#### 8. 전력 품질 지표 분석
```bash
curl -X GET "http://localhost:8000/api/power-data/1/quality?hours=24"
```

#### 9. 시계열 전력 데이터 조회
```bash
curl -X GET "http://localhost:8000/api/power-data/1/timeseries?start_time=2024-01-01T00:00:00&end_time=2024-01-02T00:00:00&interval=1h"
```

## 지원하는 기능

- **데이터 업로드**: CSV 파일 형식 지원
- **GRU 모델**: 시계열 데이터 학습 및 예측
- **회귀분석**: 다중 변수 회귀분석 모델 생성
- **회귀분석 예측**: 조건 기반 에너지 사용량 예측
- **전력 데이터 관리**: 실시간 전력 측정 데이터 수집 및 저장
- **전력 데이터 분석**: 시계열 집계, 트렌드 분석, 품질 지표 분석
- **모니터링**: 학습 진행 상황 및 모델 상태 확인

## 환경 변수

- `PYTHONPATH`: Python 경로 설정
- `PORT`: API 서버 포트 (기본값: 8000)

## 개발

### 프로젝트 구조

```
aidems-api-2025/
├── app/                    # 애플리케이션 패키지
│   ├── main.py            # FastAPI 메인 애플리케이션
│   ├── routers/           # API 라우터들
│   │   ├── base.py       # 기본 엔드포인트
│   │   ├── data.py       # 데이터 관리
│   │   ├── gru.py        # GRU 모델 API
│   │   ├── power_data.py # 전력 데이터 관리 API
│   │   ├── power_meters.py # 전력량계 관리 API
│   │   ├── regression.py # 회귀분석 API
│   │   └── model.py      # 모델 관리 API
│   ├── models/            # 모델 정의
│   │   ├── gru_model.py      # PyTorch GRU 모델 클래스
│   │   ├── power_models.py   # 전력 데이터 ORM 모델
│   │   ├── regression_model.py # statsmodels 회귀분석 모델 클래스
│   │   └── schemas.py         # Pydantic 데이터 모델
│   ├── services/          # 서비스 레이어
│   │   ├── gru_service.py      # GRU 서비스 로직
│   │   ├── power_orm_service.py # 전력 데이터 ORM 서비스
│   │   ├── regression_service.py # 회귀분석 서비스
│   │   └── visualization_service.py # 시각화 서비스
│   └── ...                # 기타 모듈들
├── requirements.txt        # Python 의존성
├── Dockerfile             # Docker 이미지 설정
├── docker-compose.yml     # Docker Compose 설정
├── .dockerignore          # Docker 제외 파일
└── README.md             # 프로젝트 문서
```

### 로그

서버는 loguru를 사용하여 로그를 출력합니다. 주요 로그:
- 데이터 업로드 상태
- 모델 학습 진행 상황
- 예측 요청/응답
- 오류 정보

## 주의사항

1. 첫 학습 시 시간이 오래 걸릴 수 있습니다.
2. GPU 사용을 위해서는 CUDA가 설치된 환경이 필요합니다.
3. 메모리 사용량이 많을 수 있으므로 충분한 RAM을 확보하세요.
4. 전력 데이터 분석은 pandas를 사용하여 시계열 집계를 수행합니다.
5. 대용량 데이터 처리 시 메모리 사용량에 주의하세요. 