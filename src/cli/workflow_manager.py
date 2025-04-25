"""
워크플로우 관리자 모듈

이 모듈은 다양한 데이터 유형 CLI에서 공통으로 사용되는 
학습, 평가, 배포 등의 워크플로우 관리 기능을 제공합니다.
"""

import os
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

# 로깅 설정
logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    ML 워크플로우 관리자 클래스
    
    데이터 종류별 CLI에서 공통으로 사용되는 워크플로우를 추상화합니다.
    학습, 평가, 배포 등의 일반적인 ML 워크플로우 단계를 정의합니다.
    """
    
    def __init__(
        self, 
        model_type: str,
        base_paths: Dict[str, Path],
        cli_instance: Any  # BaseCLI 인스턴스에 대한 참조
    ):
        """
        워크플로우 관리자 초기화
        
        Args:
            model_type: 모델 유형 식별자 (예: 'sensor', 'image', 'text')
            base_paths: 주요 경로 사전 ('models', 'data', 'logs' 등)
            cli_instance: 이 관리자를 사용하는 CLI 인스턴스 (메시지 출력 등에 사용)
        """
        self.model_type = model_type
        self.paths = base_paths
        self.cli = cli_instance
        logger.info(f"{self.model_type} 모델 워크플로우 관리자 초기화됨")
    
    def train_model(
        self,
        model_instance: Any,
        train_func: Callable,
        train_data: Any,
        valid_data: Any,
        model_params: Dict[str, Any],
        training_params: Dict[str, Any],
        device: Any
    ) -> Tuple[Any, Dict[str, Any], str]:
        """
        모델 학습 워크플로우
        
        Args:
            model_instance: 학습할 모델 인스턴스
            train_func: 실제 학습을 수행하는 함수
            train_data: 학습 데이터
            valid_data: 검증 데이터
            model_params: 모델 하이퍼파라미터
            training_params: 학습 관련 파라미터
            device: 학습에 사용할 장치 (CPU/GPU)
            
        Returns:
            Tuple[Any, Dict[str, Any], str]: (학습된 모델, 학습 이력, 저장된 모델 경로)
        """
        self.cli.print_header("모델 학습")
        
        try:
            # 학습 파라미터 확인 및 표시
            self.cli.show_message("학습에 사용할 설정:")
            self.cli.show_message(f"- 장치: {device}")
            for key, value in model_params.items():
                self.cli.show_message(f"- {key}: {value}")
            for key, value in training_params.items():
                self.cli.show_message(f"- {key}: {value}")
            
            # 학습 시작 확인
            start_training = self.cli.get_yes_no_input("\n위 설정으로 학습을 시작하시겠습니까?")
            if not start_training:
                self.cli.show_message("학습을 취소합니다.")
                return None, None, None
            
            # 모델 학습 수행
            self.cli.show_message("\n[1/3] 모델 학습 중...")
            
            # 모델 학습 (전달된 학습 함수 사용)
            model, history = train_func(
                model=model_instance,
                train_data=train_data,
                valid_data=valid_data,
                device=device,
                **training_params
            )
            
            # 모델 저장
            self.cli.show_message("\n[2/3] 모델 저장 중...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_save_path = self.paths["models"] / f"{self.model_type}_model_{timestamp}.pth"
            self._save_model(model, model_save_path)
            
            # 모델 정보 저장
            self.cli.show_message("\n[3/3] 모델 정보 저장 중...")
            model_info_path = self._save_model_info(model, model_params, training_params)
            
            # 학습 결과 출력
            self.cli.show_success("모델 학습이 완료되었습니다.")
            
            # 검증 성능 출력
            if 'valid_accuracy' in history:
                val_acc = history['valid_accuracy'][-1]
                self.cli.show_message(f"- 검증 정확도: {val_acc:.4f}")
            if 'valid_loss' in history:
                val_loss = history['valid_loss'][-1]
                self.cli.show_message(f"- 검증 손실: {val_loss:.4f}")
            
            self.cli.show_message(f"\n모델이 저장되었습니다: {model_save_path}")
            self.cli.show_message(f"모델 정보가 저장되었습니다: {model_info_path}")
            
            return model, history, model_save_path
            
        except Exception as e:
            self.cli.show_error(f"모델 학습 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 학습 중 예외 발생")
            return None, None, None
    
    def evaluate_model(
        self,
        model_instance: Any,
        eval_func: Callable,
        test_data: Any,
        device: Any,
        viz_funcs: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        모델 평가 워크플로우
        
        Args:
            model_instance: 평가할 모델 인스턴스
            eval_func: 모델 평가를 수행하는 함수
            test_data: 테스트 데이터
            device: 평가에 사용할 장치 (CPU/GPU)
            viz_funcs: 시각화 함수 딕셔너리 {'name': vis_func, ...}
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        self.cli.print_header("모델 평가")
        
        if model_instance is None:
            self.cli.show_error("평가할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            return None
        
        try:
            # 모델 평가 수행
            self.cli.show_message("\n[1/3] 모델 평가 중...")
            
            # 모델 평가 (전달된 평가 함수 사용)
            evaluation_result = eval_func(
                model=model_instance,
                test_data=test_data,
                device=device
            )
            
            # 평가 결과 출력
            self.cli.show_success("모델 평가가 완료되었습니다.")
            
            if 'accuracy' in evaluation_result:
                self.cli.show_message(f"테스트 정확도: {evaluation_result['accuracy']:.4f}")
            elif 'rmse' in evaluation_result:  # 회귀 모델용
                self.cli.show_message(f"테스트 RMSE: {evaluation_result['rmse']:.4f}")
            
            # 평가 결과 저장
            self.cli.show_message("\n[2/3] 평가 결과 저장 중...")
            eval_path = self._save_evaluation_result(evaluation_result)
            self.cli.show_message(f"평가 결과가 저장되었습니다: {eval_path}")
            
            # 시각화 (제공된 경우)
            if viz_funcs:
                self.cli.show_message("\n[3/3] 평가 결과 시각화 중...")
                plot_paths = []
                
                for viz_name, viz_func in viz_funcs.items():
                    try:
                        plot_path = viz_func(
                            model=model_instance,
                            data=test_data,
                            result=evaluation_result,
                            plot_dir=str(self.paths["models"] / "plots")
                        )
                        plot_paths.append((viz_name, plot_path))
                    except Exception as viz_error:
                        logger.error(f"시각화 '{viz_name}' 생성 중 오류: {str(viz_error)}")
                
                # 생성된 시각화 파일 출력
                for viz_name, plot_path in plot_paths:
                    self.cli.show_message(f"- {viz_name} 시각화: {plot_path}")
            
            return evaluation_result
            
        except Exception as e:
            self.cli.show_error(f"모델 평가 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 평가 중 예외 발생")
            return None
    
    def deploy_model(
        self,
        model_instance: Any,
        model_path: str,
        inference_script_template_path: str,
        deploy_dir: Optional[Path] = None,
        evaluation_result: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        additional_files: Optional[Dict[str, str]] = None
    ) -> Optional[Path]:
        """
        모델 배포 워크플로우
        
        Args:
            model_instance: 배포할 모델 인스턴스
            model_path: 모델 파일 경로
            inference_script_template_path: 추론 스크립트 템플릿 경로
            deploy_dir: 배포 디렉토리 (None이면 기본 경로 사용)
            evaluation_result: 모델 평가 결과 (있는 경우)
            model_params: 모델 파라미터 정보
            additional_files: 추가 파일 사전 {'파일명': '내용', ...}
            
        Returns:
            Optional[Path]: 배포 디렉토리 경로 (성공 시)
        """
        self.cli.print_header("모델 배포")
        
        if model_instance is None or not model_path:
            self.cli.show_error("배포할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            return None
        
        try:
            # 배포 디렉토리 설정
            if deploy_dir is None:
                deploy_dir = self.paths["models"] / "deployment"
            
            self.cli.show_message("모델 배포는 학습된 모델을 배포 디렉토리에 복사하고")
            self.cli.show_message("추론을 위한 필요한 파일들을 준비하는 단계입니다.\n")
            
            deploy_dir = Path(self.cli.get_input("배포 디렉토리 경로", deploy_dir))
            
            # 배포 디렉토리 생성
            os.makedirs(deploy_dir, exist_ok=True)
            
            # 타임스탬프로 하위 디렉토리 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            deploy_subdir = deploy_dir / f"deployment_{timestamp}"
            os.makedirs(deploy_subdir, exist_ok=True)
            
            # 모델 파일 복사
            self.cli.show_message("\n[1/4] 모델 파일 복사 중...")
            model_filename = os.path.basename(model_path)
            deploy_model_path = deploy_subdir / model_filename
            shutil.copy2(model_path, deploy_model_path)
            self.cli.show_message(f"모델 파일 복사 완료: {deploy_model_path}")
            
            # 모델 정보 파일 복사
            self.cli.show_message("\n[2/4] 모델 정보 및 평가 결과 복사 중...")
            model_info_src = self.paths["models"] / 'model_info.json'
            model_info_dst = deploy_subdir / 'model_info.json'
            
            if os.path.exists(model_info_src):
                shutil.copy2(model_info_src, model_info_dst)
                self.cli.show_message(f"모델 정보 파일 복사 완료: {model_info_dst}")
            else:
                # 모델 정보 파일이 없으면 새로 생성
                model_info = {}
                if hasattr(model_instance, 'get_model_info'):
                    model_info = model_instance.get_model_info()
                if model_params:
                    model_info.update(model_params)
                
                model_info.update({
                    "model_type": self.model_type,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                with open(model_info_dst, 'w') as f:
                    json.dump(model_info, f, indent=4)
                self.cli.show_message(f"모델 정보 파일 생성 완료: {model_info_dst}")
            
            # 평가 결과 저장 (있는 경우)
            if evaluation_result:
                eval_result_path = deploy_subdir / 'evaluation_result.json'
                # 직렬화 가능한 평가 결과만 포함
                eval_result = {}
                for k, v in evaluation_result.items():
                    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                        eval_result[k] = v
                
                with open(eval_result_path, 'w') as f:
                    json.dump(eval_result, f, indent=4)
                self.cli.show_message(f"평가 결과 저장 완료: {eval_result_path}")
            
            # 추론 스크립트 생성
            self.cli.show_message("\n[3/4] 추론 스크립트 생성 중...")
            
            # 추론 스크립트 템플릿 읽기
            try:
                with open(inference_script_template_path, 'r') as f:
                    inference_script_template = f.read()
                
                # 템플릿에 모델 정보 대입
                inference_script = inference_script_template.format(
                    model_filename=model_filename,
                    model_type=self.model_type,
                    timestamp=timestamp
                )
                
                inference_script_path = deploy_subdir / 'inference.py'
                with open(inference_script_path, 'w') as f:
                    f.write(inference_script)
                
                # 실행 권한 부여
                os.chmod(inference_script_path, 0o755)
                self.cli.show_message(f"추론 스크립트 생성 완료: {inference_script_path}")
                
            except Exception as script_error:
                logger.error(f"추론 스크립트 생성 중 오류: {str(script_error)}")
                self.cli.show_warning(f"추론 스크립트 생성 실패: {str(script_error)}")
            
            # 추가 파일 생성 (있는 경우)
            if additional_files:
                for filename, content in additional_files.items():
                    file_path = deploy_subdir / filename
                    with open(file_path, 'w') as f:
                        f.write(content)
                    self.cli.show_message(f"{filename} 생성 완료: {file_path}")
            
            # README 생성
            self.cli.show_message("\n[4/4] README 생성 중...")
            readme_content = self._generate_readme(
                model_type=self.model_type,
                model_filename=model_filename,
                model_instance=model_instance,
                model_params=model_params
            )
            
            readme_path = deploy_subdir / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            self.cli.show_message(f"README 생성 완료: {readme_path}")
            
            # 배포 완료
            self.cli.show_success(f"모델 배포가 완료되었습니다: {deploy_subdir}")
            
            return deploy_subdir
            
        except Exception as e:
            self.cli.show_error(f"모델 배포 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 배포 중 예외 발생")
            return None
    
    def _save_model(self, model: Any, save_path: Path) -> bool:
        """모델 저장 헬퍼 함수"""
        try:
            # PyTorch 모델
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), save_path)
            # 다른 종류의 모델
            elif hasattr(model, 'save'):
                model.save(save_path)
            else:
                # 직렬화 가능한 일반 모델 (scikit-learn 등)
                import joblib
                joblib.dump(model, save_path)
            
            return True
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
            return False
    
    def _save_model_info(self, model: Any, model_params: Dict[str, Any], training_params: Dict[str, Any]) -> str:
        """모델 정보 저장 헬퍼 함수"""
        model_info = {}
        
        # 모델에서 정보 가져오기
        if hasattr(model, 'get_model_info'):
            model_info.update(model.get_model_info())
        
        # 파라미터 추가
        model_info.update({
            "model_type": self.model_type,
            "model_params": model_params,
            "training_params": training_params,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 정보 저장
        model_info_path = self.paths["models"] / 'model_info.json'
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        return model_info_path
    
    def _save_evaluation_result(self, evaluation_result: Dict[str, Any]) -> str:
        """평가 결과 저장 헬퍼 함수"""
        # 결과를 직렬화 가능한 형태로 변환
        serializable_result = {}
        
        for key, value in evaluation_result.items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable_result[key] = value
            elif hasattr(value, 'tolist'):  # NumPy 배열 등
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = str(value)
        
        # 결과 저장
        eval_result_path = self.paths["data"] / "processed" / f"evaluation_{self.model_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_result_path, 'w') as f:
            json.dump(serializable_result, f, indent=4)
        
        return eval_result_path
    
    def _generate_readme(self, model_type: str, model_filename: str, model_instance: Any, model_params: Optional[Dict[str, Any]]) -> str:
        """README 생성 헬퍼 함수"""
        model_info = {}
        if hasattr(model_instance, 'get_model_info'):
            model_info = model_instance.get_model_info()
        
        # 모델 파라미터 정보
        param_info = ""
        if model_params:
            for key, value in model_params.items():
                param_info += f"- {key}: {value}\n"
        
        # README 내용
        readme = f"""# {model_type.capitalize()} 모델 배포

## 배포 정보
- 배포 날짜: {time.strftime("%Y-%m-%d %H:%M:%S")}
- 모델 파일: {model_filename}

## 사용 방법

### 필요 조건
- Python 3.7 이상
- PyTorch
- NumPy

### 추론 실행
```bash
python inference.py --data 입력_데이터_파일 --model {model_filename} --model-info model_info.json
```

## 모델 정보
- 모델 유형: {model_type}
{param_info}
"""

        # 모델 구조 정보 (있는 경우)
        if model_info:
            readme += "\n## 모델 구조\n"
            for key, value in model_info.items():
                readme += f"- {key}: {value}\n"
        
        readme += """
## 입력 및 출력 형식

### 입력
- CSV 파일 또는 NumPy 배열 (.npy) 형식의 데이터

### 출력
- 예측 클래스 또는 값
- 신뢰도 점수 (분류 모델의 경우)
"""
        
        return readme