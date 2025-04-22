import os
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

from src.models.lstm_model import LSTMModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'mlflow.log'))
    ]
)
logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    MLflow를 사용하여 모델 학습, 성능 평가, 모델 관리를 수행하는 클래스
    """
    
    def __init__(
        self, 
        experiment_name: str = "smart_factory_lstm",
        tracking_uri: Optional[str] = None,
        model_registry: Optional[str] = None
    ):
        """
        MLflow 추적기 초기화
        
        Args:
            experiment_name (str): MLflow 실험 이름
            tracking_uri (str, optional): MLflow 추적 서버 URI
            model_registry (str, optional): 모델 레지스트리 URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.tracking_uri = mlflow.get_tracking_uri()
        self.experiment_name = experiment_name
        self.model_registry = model_registry
        
        # 실험 설정
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not self.experiment:
            experiment_id = mlflow.create_experiment(experiment_name)
            self.experiment = mlflow.get_experiment(experiment_id)
        
        logger.info(f"MLflow 추적 서버: {self.tracking_uri}")
        logger.info(f"실험 이름: {experiment_name}, ID: {self.experiment.experiment_id}")
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        MLflow 실행 시작
        
        Args:
            run_name (str, optional): 실행 이름
            
        Returns:
            str: 실행 ID
        """
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
        
        if not run_name:
            run_name = f"lstm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=run_name)
        logger.info(f"MLflow 실행 시작: {run_name} (ID: {run.info.run_id})")
        
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        하이퍼파라미터 로깅
        
        Args:
            params (Dict[str, Any]): 로깅할 파라미터
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        메트릭 로깅
        
        Args:
            metrics (Dict[str, float]): 로깅할 메트릭
            step (int, optional): 현재 스텝
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self, 
        model: LSTMModel, 
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        PyTorch 모델 로깅
        
        Args:
            model (LSTMModel): 로깅할 모델
            artifact_path (str): 아티팩트 경로
            registered_model_name (str, optional): 등록할 모델 이름
            
        Returns:
            str: 모델 URI
        """
        return mlflow.pytorch.log_model(
            model, 
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        ).model_uri
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        아티팩트 로깅
        
        Args:
            local_path (str): 로컬 파일 경로
            artifact_path (str, optional): 아티팩트 경로
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """
        딕셔너리 로깅
        
        Args:
            dictionary (Dict[str, Any]): 로깅할 딕셔너리
            artifact_file (str): 아티팩트 파일 이름
        """
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_figure(self, figure: plt.Figure, artifact_file: str) -> None:
        """
        Matplotlib 그림 로깅
        
        Args:
            figure (plt.Figure): 로깅할 그림
            artifact_file (str): 아티팩트 파일 이름
        """
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self) -> None:
        """MLflow 실행 종료"""
        mlflow.end_run()
        logger.info("MLflow 실행 종료")
    
    def get_best_run(
        self, 
        metric_name: str = "val_loss", 
        ascending: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        최고 성능의 실행 검색
        
        Args:
            metric_name (str): 기준 메트릭 이름
            ascending (bool): 오름차순 정렬 여부 (True이면 낮은 값이 더 좋음)
            
        Returns:
            Tuple[str, Dict[str, Any]]: (실행 ID, 실행 데이터)
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"]
        )
        
        if len(runs) == 0:
            logger.warning(f"메트릭 '{metric_name}'을 가진 실행을 찾을 수 없습니다.")
            return None, {}
        
        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        
        # 전체 실행 데이터 검색
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(best_run_id)
        
        logger.info(f"최고 성능 실행: {best_run_id}, {metric_name}={best_run[f'metrics.{metric_name}']}")
        
        return best_run_id, {
            "run_id": best_run_id,
            "metrics": {k.split(".")[-1]: v for k, v in best_run.items() if k.startswith("metrics.")},
            "params": {k.split(".")[-1]: v for k, v in best_run.items() if k.startswith("params.")},
            "start_time": best_run["start_time"],
            "status": run_data.info.status
        }
    
    def load_model(
        self, 
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        stage: str = "Production"
    ) -> LSTMModel:
        """
        MLflow에서 모델 로드
        
        Args:
            run_id (str, optional): 실행 ID
            model_uri (str, optional): 모델 URI
            model_name (str, optional): 레지스트리 모델 이름
            model_version (str, optional): 모델 버전
            stage (str): 모델 스테이지 (Production, Staging, Archived)
            
        Returns:
            LSTMModel: 로드된 모델
        """
        if model_uri:
            model_path = model_uri
        elif run_id:
            model_path = f"runs:/{run_id}/model"
        elif model_name and model_version:
            model_path = f"models:/{model_name}/{model_version}"
        elif model_name:
            model_path = f"models:/{model_name}/{stage}"
        else:
            # 최고 성능 모델 로드
            best_run_id, _ = self.get_best_run()
            if not best_run_id:
                raise ValueError("로드할 모델을 찾을 수 없습니다.")
            model_path = f"runs:/{best_run_id}/model"
        
        logger.info(f"모델 로드 중: {model_path}")
        model = mlflow.pytorch.load_model(model_path)
        
        return model
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """
        모델 레지스트리에 모델 등록
        
        Args:
            model_uri (str): 모델 URI
            name (str): 등록할 모델 이름
            description (str, optional): 모델 설명
            
        Returns:
            str: 등록된 모델 버전
        """
        client = mlflow.tracking.MlflowClient()
        
        try:
            model_details = mlflow.register_model(model_uri=model_uri, name=name)
            version = model_details.version
            
            if description:
                client.update_registered_model(
                    name=name,
                    description=description
                )
                
            logger.info(f"모델 등록 완료: {name}, 버전 {version}")
            return version
        except Exception as e:
            logger.error(f"모델 등록 실패: {str(e)}")
            raise
    
    def set_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        모델 스테이지 설정
        
        Args:
            name (str): 모델 이름
            version (str): 모델 버전
            stage (str): 설정할 스테이지 (Production, Staging, Archived)
            archive_existing_versions (bool): 기존 프로덕션 버전을 아카이브할지 여부
        """
        client = mlflow.tracking.MlflowClient()
        
        if archive_existing_versions and stage == "Production":
            # 현재 프로덕션 버전 검색
            prod_versions = [
                mv for mv in client.search_model_versions(f"name='{name}'")
                if mv.current_stage == "Production"
            ]
            
            # 기존 프로덕션 버전 아카이브
            for mv in prod_versions:
                if mv.version != version:
                    client.transition_model_version_stage(
                        name=name,
                        version=mv.version,
                        stage="Archived"
                    )
                    logger.info(f"기존 프로덕션 버전 {mv.version} 아카이브됨")
        
        # 스테이지 전환
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        
        logger.info(f"모델 {name} 버전 {version}의 스테이지가 '{stage}'로 설정됨")
    
    def delete_model_version(self, name: str, version: str) -> None:
        """
        모델 버전 삭제
        
        Args:
            name (str): 모델 이름
            version (str): 삭제할 모델 버전
        """
        client = mlflow.tracking.MlflowClient()
        client.delete_model_version(name=name, version=version)
        logger.info(f"모델 {name} 버전 {version} 삭제됨")
    
    def compare_runs(
        self, 
        run_ids: List[str], 
        metric_names: List[str]
    ) -> pd.DataFrame:
        """
        여러 실행을 비교
        
        Args:
            run_ids (List[str]): 비교할 실행 ID 목록
            metric_names (List[str]): 비교할 메트릭 이름 목록
            
        Returns:
            pd.DataFrame: 비교 결과 데이터프레임
        """
        results = []
        client = mlflow.tracking.MlflowClient()
        
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0),
                    "status": run.info.status
                }
                
                # 파라미터 추가
                for key, value in run.data.params.items():
                    run_data[f"param.{key}"] = value
                
                # 메트릭 추가
                for metric_name in metric_names:
                    if metric_name in run.data.metrics:
                        run_data[f"metric.{metric_name}"] = run.data.metrics[metric_name]
                    else:
                        run_data[f"metric.{metric_name}"] = None
                
                results.append(run_data)
            except Exception as e:
                logger.warning(f"실행 {run_id} 정보를 가져오는 중 오류 발생: {str(e)}")
        
        if not results:
            logger.warning("비교할 실행이 없습니다.")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def plot_metric_comparison(
        self, 
        runs_df: pd.DataFrame, 
        metric_name: str,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        여러 실행의 메트릭을 비교하는 그래프 생성
        
        Args:
            runs_df (pd.DataFrame): 비교할 실행 데이터프레임
            metric_name (str): 비교할 메트릭 이름
            figsize (Tuple[int, int], optional): 그래프 크기
            
        Returns:
            plt.Figure: 생성된 그래프
        """
        metric_col = f"metric.{metric_name}"
        if metric_col not in runs_df.columns:
            logger.warning(f"메트릭 '{metric_name}'이 데이터프레임에 없습니다.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 날짜 기준으로 정렬
        runs_df_sorted = runs_df.sort_values("start_time")
        
        # 메트릭 값 시각화
        ax.bar(range(len(runs_df_sorted)), runs_df_sorted[metric_col])
        ax.set_xticks(range(len(runs_df_sorted)))
        ax.set_xticklabels([str(idx) for idx in range(len(runs_df_sorted))], rotation=45)
        
        # 런 ID를 툴팁으로 표시
        for i, (_, row) in enumerate(runs_df_sorted.iterrows()):
            ax.annotate(
                row["run_id"][:8] + "...",
                xy=(i, row[metric_col]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )
        
        ax.set_title(f"Comparison of {metric_name} across runs")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Run Index")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        return fig

    def track_training(
        self,
        model: LSTMModel,
        train_history: Dict[str, List[float]],
        test_metrics: Dict[str, float],
        params: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        model_name: Optional[str] = "smart_factory_lstm_model",
        description: Optional[str] = None,
        register: bool = False,
        plots: bool = True
    ) -> str:
        """
        모델 학습 과정과 결과를 MLflow에 로깅
        
        Args:
            model (LSTMModel): 학습된 모델
            train_history (Dict[str, List[float]]): 학습 이력
            test_metrics (Dict[str, float]): 테스트 메트릭
            params (Dict[str, Any]): 하이퍼파라미터
            feature_importance (Dict[str, float], optional): 특성 중요도
            model_name (str, optional): 등록할 모델 이름
            description (str, optional): 모델 설명
            register (bool): 모델 레지스트리에 등록 여부
            plots (bool): 시각화 그래프 생성 여부
            
        Returns:
            str: 실행 ID
        """
        # 실행 시작
        run_id = self.start_run()
        
        try:
            # 하이퍼파라미터 로깅
            self.log_params(params)
            
            # 훈련 이력 로깅
            for epoch, (train_loss, val_loss) in enumerate(zip(
                train_history.get('train_loss', []),
                train_history.get('val_loss', [])
            )):
                self.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)
            
            # 테스트 메트릭 로깅
            self.log_metrics(test_metrics)
            
            # 특성 중요도 로깅 (있는 경우)
            if feature_importance:
                self.log_dict(feature_importance, "feature_importance.json")
                
                if plots:
                    # 특성 중요도 시각화
                    fig, ax = plt.subplots(figsize=(10, 6))
                    features = list(feature_importance.keys())
                    importances = list(feature_importance.values())
                    
                    sorted_idx = np.argsort(importances)
                    ax.barh([features[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
                    ax.set_title("Feature Importance")
                    ax.set_xlabel("Importance")
                    
                    self.log_figure(fig, "feature_importance.png")
                    plt.close(fig)
            
            # 손실 그래프 생성
            if plots and 'train_loss' in train_history and 'val_loss' in train_history:
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = range(1, len(train_history['train_loss']) + 1)
                ax.plot(epochs, train_history['train_loss'], 'b-', label='Training Loss')
                ax.plot(epochs, train_history['val_loss'], 'r-', label='Validation Loss')
                ax.set_title('Training and Validation Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
                
                self.log_figure(fig, "loss_curves.png")
                plt.close(fig)
            
            # 모델 로깅
            model_uri = self.log_model(model, "model")
            
            # 모델 레지스트리에 등록
            if register and model_name:
                version = self.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    description=description
                )
                
                # 프로덕션으로 설정
                self.set_model_stage(
                    name=model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True
                )
            
            return run_id
        finally:
            # 실행 종료
            self.end_run()


def main():
    """MLflow 모듈 사용 예시"""
    from src.models.train import prepare_data, train_lstm_model, evaluate_model
    
    # 데이터 준비
    data_path = "data/processed/sensor_data.csv"
    train_loader, val_loader, test_loader, data_info = prepare_data(
        data_path=data_path,
        sequence_length=24
    )
    
    # 모델 학습
    model, history = train_lstm_model(
        train_loader=train_loader,
        val_loader=val_loader,
        data_info=data_info,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        epochs=10,  # 예시용으로 에폭 수 감소
        patience=3,
        model_dir="models",
        model_name="lstm_model"
    )
    
    # 모델 평가
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        plot=False
    )
    
    # MLflow 추적 (로컬 추적 서버)
    tracker = MLflowTracker(experiment_name="smart_factory_lstm")
    
    # 하이퍼파라미터
    params = {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "sequence_length": data_info['sequence_length'],
        "input_size": data_info['input_size'],
        "feature_count": len(data_info['feature_cols'])
    }
    
    # 훈련 추적
    run_id = tracker.track_training(
        model=model,
        train_history=history,
        test_metrics=metrics,
        params=params,
        model_name="smart_factory_lstm_model",
        description="LSTM 모델 - 센서 데이터 예측",
        register=True,
        plots=True
    )
    
    print(f"MLflow 학습 추적 완료: 실행 ID {run_id}")


if __name__ == "__main__":
    main()