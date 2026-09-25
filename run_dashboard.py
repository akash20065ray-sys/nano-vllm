import webbrowser
import time
import uvicorn
from server.app import app

def main():
    print("=" * 65)
    print("  LAUNCHING MICROSERVE-LLM SYSTEMS OBSERVABILITY DASHBOARD")
    print("=" * 65)
    print("-> Server running on: http://127.0.0.1:8000")
    print("-> Opening dashboard in your default browser...")
    
    # Automatically open browser after 1 second
    import threading
    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:8000")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
