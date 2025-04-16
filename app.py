import os
import sys
import subprocess
import shutil
import time
import threading

def sync_notebooks(source_dir, dest_dir, interval=15):
    """
    노트북 파일을 주기적으로 동기화하는 함수
    :param source_dir: 소스 디렉토리 (컨테이너 내부)
    :param dest_dir: 대상 디렉토리 (호스트)
    :param interval: 동기화 간격 (초)
    """
    print(f"Starting notebook synchronization from {source_dir} to {dest_dir}")
    
    while True:
        try:
            # 디렉토리가 존재하는지 확인
            if not os.path.exists(source_dir):
                print(f"Source directory {source_dir} does not exist")
                time.sleep(interval)
                continue
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # 모든 .ipynb 파일 동기화
            for filename in os.listdir(source_dir):
                if filename.endswith('.ipynb'):
                    src_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(dest_dir, filename)
                    
                    # 파일 존재 여부 및 최신성 확인
                    if not os.path.exists(dest_path) or \
                       os.path.getmtime(src_path) > os.path.getmtime(dest_path):
                        try:
                            shutil.copy2(src_path, dest_path)
                            print(f"Synced: {filename}")
                        except Exception as copy_error:
                            print(f"Error syncing {filename}: {copy_error}")
            
            # 지정된 간격만큼 대기
            time.sleep(interval)
        
        except Exception as e:
            print(f"Synchronization error: {e}")
            time.sleep(interval)

def start_jupyter():
    print("Starting Jupyter Notebook...")
    
    # 동기화할 디렉토리 경로 (필요에 따라 수정)
    source_notebook_dir = "/app/notebooks"
    host_notebook_dir = "./notebooks"
    
    # 동기화 스레드 시작
    sync_thread = threading.Thread(
        target=sync_notebooks, 
        args=(source_notebook_dir, host_notebook_dir), 
        daemon=True  # 메인 스레드 종료 시 함께 종료
    )
    sync_thread.start()
    
    # Jupyter 실행 옵션 설정
    jupyter_command = [
        "jupyter", "notebook",
        "--ip=0.0.0.0",     # 모든 IP에서 접근 가능
        "--port=8888",      # 포트 설정
        "--no-browser",     # 브라우저 자동 실행 방지
        "--allow-root",     # root 사용자로 실행 허용 (Docker 내부에서 필요할 수 있음)
        "--NotebookApp.token=''",  # 토큰 인증 비활성화 (선택 사항)
        "--NotebookApp.password=''",  # 비밀번호 인증 비활성화 (선택 사항)
        "--notebook-dir=/app/notebooks"  # 노트북 디렉토리 지정
    ]
    
    # Jupyter 시작
    try:
        jupyter_process = subprocess.Popen(jupyter_command)
        print("Jupyter Notebook is running at http://localhost:8888")
        print("Automatic notebook synchronization is active")
        print("Press Ctrl+C to stop the server")
        jupyter_process.wait()
    except KeyboardInterrupt:
        print("\nStopping Jupyter Notebook...")
        jupyter_process.terminate()
        jupyter_process.wait()
        print("Jupyter Notebook stopped")
    except Exception as e:
        print(f"Error starting Jupyter Notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_jupyter()