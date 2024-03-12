import requests
import subprocess

def check_mlflow_server(host='127.0.0.1', port=8081):
    """Check if the MLflow server is running and start it if not."""
    try:
        # Attempt to connect to the MLflow server
        response = requests.get(f"http://{host}:{port}")
        if response.status_code == 200:
            print("MLflow server is running.")
        else:
            raise Exception("MLflow server not responding as expected.")
    except Exception as e:
        print(f"MLflow server check failed: {e}")
        print("Starting MLflow server...")
        # Start the MLflow server using subprocess in the background
        subprocess.Popen(["mlflow", "server", "--host", host, "--port", str(port)],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        print(f"MLflow server started on {host}:{port}.")

