"""
Start the backend server.

    python run.py

Host, port and reload are read from the environment / .env (see .env.example).
"""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )

