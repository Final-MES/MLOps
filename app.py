import os
import sys
import subprocess

def start_jupyter():
    print("Starting Jupyter Notebook...")
    
    # Jupyter 실행 옵션 설정
    jupyter_command = [
        "jupyter", "notebook",
        "--ip=0.0.0.0",     # 모든 IP에서 접근 가능
        "--port=8888",      # 포트 설정
        "--no-browser",     # 브라우저 자동 실행 방지
        "--allow-root",     # root 사용자로 실행 허용 (Docker 내부에서 필요할 수 있음)
        "--NotebookApp.token=''",  # 토큰 인증 비활성화 (선택 사항)
        "--NotebookApp.password=''"  # 비밀번호 인증 비활성화 (선택 사항)
    ]
    
    # Jupyter 시작
    try:
        jupyter_process = subprocess.Popen(jupyter_command)
        print("Jupyter Notebook is running at http://localhost:8888")
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