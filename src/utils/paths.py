# src/utils/paths.py - 경로 처리 유틸리티
from pathlib import Path

# 프로젝트 루트 디렉토리
def get_project_paths():
    """프로젝트의 주요 경로들을 딕셔너리로 반환"""
    root = Path(__file__).parent.parent.parent.absolute()
    return {
        "root": root,
        "config": root / "config",
        "data": root / "data",
        "models": root / "models",
        "logs": root / "logs",
    }

def get_config_path() -> Path:
    """설정 파일 경로"""
    return get_project_paths()["config"]

def get_data_path(subdir: str = '') -> Path:
    """데이터 디렉토리 경로"""
    path = get_project_paths()["data"]
    if subdir:
        path = path / subdir
    return path

def get_model_path(model_name: str = '') -> Path:
    """모델 파일 경로"""
    path = get_project_paths()["models"]
    if model_name:
        path = path / f"{model_name}.pth"
    return path

def ensure_dir(path: Path) -> Path:
    """디렉토리 존재 확인 및 생성"""
    path.mkdir(parents=True, exist_ok=True)
    return path