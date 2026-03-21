"""
main.py
-------
Entry point for the Sales Copilot GPU server.

Run with:
    python server/main.py

Or via uvicorn directly (same thing, more control):
    uvicorn server.api:app --host 0.0.0.0 --port 8000

Why 0.0.0.0?
    Binding to 0.0.0.0 means the server accepts connections on all network
    interfaces — necessary on RunPod so your Mac can reach it over the internet.
    Binding to 127.0.0.1 (localhost) would only accept connections from the
    same machine, which would work locally but not on a remote GPU pod.
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("SERVER_BIND_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", 8000))

if __name__ == "__main__":
    print(f"Starting Sales Copilot server on {HOST}:{PORT}")
    uvicorn.run(
        "api:app",
        host     = HOST,
        port     = PORT,
        reload   = False,    # Never use reload=True in production — wastes memory and
                             # causes models to reload on every file change
        workers  = 1,        # Single worker — models are loaded once and shared.
                             # Multiple workers would each load their own copy,
                             # multiplying GPU memory usage.
        log_level = "info",
    )