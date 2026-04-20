import subprocess
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
        print(line.strip())  # optional: shows backend logs

        if "Application startup complete" in line:
            print("Backend Started")
            break

    return process


def run_ollama():
    subprocess.Popen(
        'start cmd /k "ollama start"',
        shell=True
    )

    print("Starting Ollama...")
    time.sleep(3)


def run_frontend():
    frontend_path = os.path.join(os.getcwd(), "Frontend")

    subprocess.Popen(
        'start cmd /k "npm run dev"',
        shell=True,
        cwd=frontend_path
    )

    print("Frontend Starting...")
    time.sleep(3)

    print("Frontend:  http://localhost:5173/")
    print("Backend:   http://127.0.0.1:8000/docs")


def main():
    backend_process = run_backend()   # Wait until backend fully starts
    run_ollama()                      # Then start Ollama
    run_frontend()                    # Then start frontend

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        print("Launcher stopped.")


if __name__ == "__main__":
    main()