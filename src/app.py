import os
import logging
import argparse
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    MLOps 애플리케이션 메인 함수
    환경 변수에 따라 다양한 모드로 실행
    """
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Smart Factory MLOps Application')
    parser.add_argument('--mode', type=str, default=os.getenv('RUN_MODE', 'serve'),
                      help='실행 모드: train, serve, evaluate')
    args = parser.parse_args()
    
    logger.info(f"애플리케이션 시작: 모드 = {args.mode}")
    
    # 모드에 따라 실행
    if args.mode == 'train':
        # 학습 모드
        logger.info("학습 모드 시작")
        from src.models.train import prepare_data, train_lstm_model, evaluate_model
        
        try:
            # 데이터 준비
            data_path = os.getenv('DATA_PATH', 'data/processed/sensor_data.csv')
            train_loader, val_loader, test_loader, data_info = prepare_data(
                data_path=data_path,
                sequence_length=int(os.getenv('SEQUENCE_LENGTH', '24')),
                test_size=float(os.getenv('TEST_SIZE', '0.2')),
                val_size=float(os.getenv('VAL_SIZE', '0.2'))
            )
            
            # 모델 학습
            model, history = train_lstm_model(
                train_loader=train_loader,
                val_loader=val_loader,
                data_info=data_info,
                hidden_size=int(os.getenv('HIDDEN_SIZE', '64')),
                num_layers=int(os.getenv('NUM_LAYERS', '2')),
                learning_rate=float(os.getenv('LEARNING_RATE', '0.001')),
                epochs=int(os.getenv('EPOCHS', '100')),
                patience=int(os.getenv('PATIENCE', '10')),
                model_dir=os.getenv('MODEL_DIR', 'models'),
                model_name=os.getenv('MODEL_NAME', 'lstm_model')
            )
            
            # 모델 평가
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                data_info=data_info,
                plot=True,
                save_plot=True,
                plot_dir=os.getenv('PLOT_DIR', 'plots')
            )
            
            # MLflow 통합 (선택적)
            if os.getenv('USE_MLFLOW', 'False').lower() == 'true':
                from mlflow.tracker import MLflowTracker
                
                # MLflow 추적기 초기화
                tracker = MLflowTracker(
                    experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'smart_factory_lstm'),
                    tracking_uri=os.getenv('MLFLOW_TRACKING_URI', None)
                )
                
                # 하이퍼파라미터
                params = {
                    "hidden_size": int(os.getenv('HIDDEN_SIZE', '64')),
                    "num_layers": int(os.getenv('NUM_LAYERS', '2')),
                    "learning_rate": float(os.getenv('LEARNING_RATE', '0.001')),
                    "batch_size": int(os.getenv('BATCH_SIZE', '32')),
                    "sequence_length": data_info['sequence_length'],
                    "input_size": data_info['input_size'],
                    "feature_count": len(data_info['feature_cols'])
                }
                
                # 모델 추적
                run_id = tracker.track_training(
                    model=model,
                    train_history=history,
                    test_metrics=metrics,
                    params=params,
                    model_name=os.getenv('REGISTERED_MODEL_NAME', 'smart_factory_lstm_model'),
                    description=f"LSTM 모델 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    register=os.getenv('REGISTER_MODEL', 'False').lower() == 'true',
                    plots=True
                )
                
                logger.info(f"MLflow 학습 추적 완료: 실행 ID {run_id}")
            
            logger.info("학습 완료")
            
        except Exception as e:
            logger.error(f"학습 중 오류 발생: {str(e)}", exc_info=True)
    
    elif args.mode == 'serve':
        # 서빙 모드
        logger.info("서빙 모드 시작")
        
        try:
            # FastAPI 서버 실행
            from src.serving.api import app
            import uvicorn
            
            port = int(os.getenv("API_PORT", 8000))
            host = os.getenv("API_HOST", "0.0.0.0")
            
            logger.info(f"모델 서빙 API 시작: {host}:{port}")
            uvicorn.run(app, host=host, port=port)
            
        except Exception as e:
            logger.error(f"서빙 중 오류 발생: {str(e)}", exc_info=True)
    
    elif args.mode == 'evaluate':
        # 평가 모드
        logger.info("평가 모드 시작")
        
        try:
            from src.models.train import evaluate_model
            from src.models.lstm_model import LSTMModel
            import torch
            import json
            
            # 모델 로드
            model_dir = os.getenv('MODEL_DIR', 'models')
            model_name = os.getenv('MODEL_NAME', 'lstm_model')
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            model_info_path = os.path.join(model_dir, f"{model_name}_info.json")
            
            # 모델 정보 로드
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            # 모델 초기화
            model = LSTMModel(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size'],
                num_layers=model_info['num_layers'],
                output_size=model_info['output_size']
            )
            
            # 모델 가중치 로드
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            logger.info(f"모델 로드 완료: {model_path}")
            
            # 평가 데이터 준비
            from src.models.train import prepare_data
            
            data_path = os.getenv('EVAL_DATA_PATH', 'data/processed/eval_data.csv')
            _, _, test_loader, data_info = prepare_data(
                data_path=data_path,
                sequence_length=model_info['sequence_length'],
                test_size=1.0,  # 모든 데이터를 테스트 세트로 사용
                val_size=0.0
            )
            
            # 모델 평가
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                data_info=data_info,
                plot=True,
                save_plot=True,
                plot_dir=os.getenv('PLOT_DIR', 'plots')
            )
            
            # 평가 결과 저장
            eval_result_path = os.path.join(model_dir, f"{model_name}_eval_result.json")
            with open(eval_result_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"평가 완료: 결과 저장 경로 = {eval_result_path}")
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}", exc_info=True)
    
    else:
        logger.error(f"알 수 없는 모드: {args.mode}")

if __name__ == "__main__":
    main()