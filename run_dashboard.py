import os
import socket
import webbrowser
import time
import uvicorn
from server.app import app

def find_available_port(default_port: int = 8000, fallback_ports = (8080, 8001, 8081)) -> int:
    """Finds the first available TCP port to prevent Errno 10048 address collision crashes."""
    for p in [default_port] + list(fallback_ports):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return default_port

def main():
    port = int(os.environ.get("PORT", find_available_port(8000)))
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("  LAUNCHING MICROSERVE-LLM SYSTEMS OBSERVABILITY DASHBOARD")
    print("=" * 65)
    print(f"-> Server running on: {url}")
    print("-> Opening dashboard in your default browser...")
    
    # Automatically open browser after 1 second
    import threading
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
