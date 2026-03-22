import subprocess
import threading
import time
import os


def run_backend():
    backend_path = os.path.join(os.getcwd(), "Backend")

    process = subprocess.Popen(
        "venv\\Scripts\\activate && uvicorn app.main:app --reload",
        shell=True,
        cwd=backend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("Starting Backend...")

    for line in iter(process.stdout.readline, ''):
        if "Application startup complete" in line:
            print("Backend Started")
            break

    return process


def run_frontend():
    frontend_path = os.path.join(os.getcwd(), "Frontend")

    process = subprocess.Popen(
        "npm run dev",
        shell=True,
        cwd=frontend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("Frontend Starting...")
    time.sleep(3)

    print("Frontend:  http://localhost:5173/")
    print("Backend:   http://127.0.0.1:8000/docs")

    return process


def main():
    backend_process = run_backend()
    frontend_process = run_frontend()

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    main()