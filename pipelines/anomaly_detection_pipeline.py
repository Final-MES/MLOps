def preprocess_sensor_data(args):
    """센서 데이터 전처리 및 통합"""
    logger.info("센서 데이터 전처리 시작")
    
    # 통합 데이터 파일 찾기
    import glob
    data_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    
    if not data_files:
        logger.error("데이터 디렉토리에 CSV 파일이 없습니다.")
        return None, None
    
    # 가장 최근 파일 사용 (또는 다른 선택 기준 사용 가능)
    latest_file = max(data_files, key=os.path.getctime)
    logger.info(f"처리할 데이터 파일: {latest_file}")
    
    try:
        # 데이터 로드
        df = pd.read_csv(latest_file)
        logger.info(f"데이터 로드 완료: {len(df)} 행, {len(df.columns)} 열")
        
        # 시간 컬럼 확인 및 처리
        time_column = None
        for col in ['timestamp', 'time', 'datetime']:
            if col in df.columns:
                time_column = col
                # 시간 컬럼 형식이 문자열인 경우 datetime으로 변환
                if df[col].dtype == object:
                    df[col] = pd.to_datetime(df[col])
                break
        
        # 시간 컬럼이 없으면 인덱스 기반 시간 생성
        if time_column is None:
            logger.warning("시간 컬럼을 찾을 수 없어 인덱스 기반 시간을 생성합니다.")
            df['time'] = pd.to_datetime(pd.date_range(start='now', periods=len(df), freq='S'))
            time_column = 'time'
        
        # 상태 컬럼 확인
        state_column = None
        for col in ['state', 'status', 'condition', 'label', 'class']:
            if col in df.columns:
                state_column = col
                break
        
        if state_column is None:
            logger.warning("상태 컬럼을 찾을 수 없습니다.")
            return None, None
        
        # 상태 분포 확인
        state_counts = df[state_column].value_counts()
        logger.info(f"상태 분포: {state_counts.to_dict()}")
        
        # 전처리기 초기화
        preprocessor = SensorDataPreprocessor(window_size=15)
        
        # 결측치 처리
        logger.info("결측치 처리 중...")
        df_cleaned = preprocessor.handle_missing_values(df)
        
        # 이상치 처리
        logger.info("이상치 처리 중...")
        # 상태 및 시간 컬럼 제외
        exclude_cols = [col for col in [time_column, state_column] if col is not None]
        df_cleaned = preprocessor.handle_outliers(df_cleaned, exclude_columns=exclude_cols)
        
        # 시간 컬럼 초 단위 변환 (보간을 위해)
        df_cleaned['time_seconds'] = df_cleaned[time_column].astype(np.int64) // 10**9
        
        # 특성 컬럼 (시간과 상태 컬럼 제외)
        feature_cols = [col for col in df_cleaned.columns if col not in [time_column, state_column, 'time_seconds']]
        
        # 각 상태별로 나누지 않고 전체 데이터를 한 번에 보간
        sensor_data = {
            'combined_sensor': df_cleaned[['time_seconds'] + feature_cols].rename(columns={'time_seconds': 'time'})
        }
        
        # 보간 간격 계산 (초 단위)
        times = sorted(df_cleaned['time_seconds'].values)
        intervals = np.diff(times)
        min_interval = np.min(intervals[intervals > 0]) if any(intervals > 0) else 0.001
        logger.info(f"보간 간격: {min_interval}초")
        
        # 전체 데이터 보간
        logger.info("데이터 보간 중...")
        interpolated_data = preprocessor.interpolate_sensor_data(
            sensor_data,
            time_range=None,  # 자동 생성
            step=min_interval,
            kind='linear'  # 선형 보간 사용
        )
        
        # 보간된 데이터 가져오기
        interpolated_df = interpolated_data['combined_sensor']
        
        # 원래 시간 형식으로 변환
        interpolated_df['timestamp'] = pd.to_datetime(interpolated_df['time'], unit='s')
        
        # 상태 정보 추가 (가장 가까운 시간의 상태로 보간)
        state_df = df_cleaned[[time_column, state_column]].copy()
        state_df['time_seconds'] = state_df[time_column].astype(np.int64) // 10**9
        
        # 가장 가까운 시간의 상태 찾기
        from scipy.spatial.distance import cdist
        
        # 두 시간 배열 간의 거리 계산
        distances = cdist(
            interpolated_df['time'].values.reshape(-1, 1),
            state_df['time_seconds'].values.reshape(-1, 1)
        )
        
        # o 행에 대한 최소 거리 인덱스 찾기
        closest_indices = np.argmin(distances, axis=1)
        
        # 보간된 데이터에 상태 추가
        interpolated_df[state_column] = state_df[state_column].iloc[closest_indices].values
        
        # 처리된 데이터 저장
        os.makedirs(args.processed_dir, exist_ok=True)
        output_path = os.path.join(args.processed_dir, f"processed_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        interpolated_df.to_csv(output_path, index=False)
        logger.info(f"처리된 데이터 저장 완료: {output_path} (총 {len(interpolated_df)} 행)")
        
        # 추가 특성 추출
        logger.info("추가 특성 추출 중...")
        feature_df = preprocessor.extract_statistical_moments(interpolated_df, columns=feature_cols)
        feature_df = preprocessor.extract_frequency_features(feature_df, columns=feature_cols)
        
        # 특성이 추가된 데이터 저장
        feature_output_path = os.path.join(args.processed_dir, f"processed_sensor_data_with_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        feature_df.to_csv(feature_output_path, index=False)
        logger.info(f"특성이 추가된 데이터 저장 완료: {feature_output_path} (총 {len(feature_df)} 행, {len(feature_df.columns)} 열)")
        
        return output_path, feature_output_path
        
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None