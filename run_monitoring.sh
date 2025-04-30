#!/bin/bash

# 실시간 모니터링 실행 스크립트

# 설정 변수들
DATA_DIR="data/raw"
MODEL_PATH="models/sensor_classifier.pth"
DB_PROFILE="default"

# SQL 서버 연결 정보
DB_TYPE="sqlserver"
DB_HOST="your_sql_server_host"  # 실제 SQL 서버 호스트 주소로 변경
DB_PORT=1433
DB_NAME="your_database_name"  # 실제 데이터베이스 이름으로 변경
DB_USER="your_username"       # 실제 사용자 이름으로 변경
DB_PASSWORD="your_password"   # 실제 비밀번호로 변경

# 실행
python real_time_monitoring.py \
  --data_dir=$DATA_DIR \
  --model_path=$MODEL_PATH \
  --db_profile=$DB_PROFILE \
  --db_type=$DB_TYPE \
  --db_host=$DB_HOST \
  --db_port=$DB_PORT \
  --db_name=$DB_NAME \
  --db_user=$DB_USER \
  --db_password=$DB_PASSWORD