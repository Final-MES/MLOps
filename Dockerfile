FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs

# 모델, 데이터, 로그 디렉토리 생성
RUN mkdir -p ${MODEL_DIR} ${DATA_DIR} ${LOG_DIR}

# 애플리케이션 코드 복사는 볼륨 마운트를 통해 수행됨

# 기본 명령어
CMD ["python", "-m", "src.app"]