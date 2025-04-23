FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    default-libmysqlclient-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# Python 패키지 설치 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models/trained \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    PYTHONPATH=/app

# 필요한 디렉토리 생성
RUN mkdir -p ${MODEL_DIR} ${DATA_DIR}/raw ${DATA_DIR}/processed ${DATA_DIR}/monitoring ${LOG_DIR}

# 소스 코드 복사
COPY . .

# 기본 명령어 - 메인 애플리케이션 실행
CMD ["python", "-m", "src.app"]