#!/usr/bin/env python3

import os
import signal
import sys
import warnings
import uvicorn
from dotenv import load_dotenv
from logging_config import setup_logging
from app import app, cleanup_resources
import logging

load_dotenv()

# Setup centralized logging
setup_logging("server")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Parameter strict_text has been deprecated and will be ignored.")


def main():
    # Note: Signal handlers are managed by uvicorn, which will trigger
    # the FastAPI lifespan shutdown properly
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    reload = os.getenv("SERVER_RELOAD", "false").lower() == "true"
    log_level = os.getenv("SERVER_LOG_LEVEL", "info")
    workers = int(os.getenv("SERVER_WORKERS", "1"))
    
    if reload:
        os.environ["START_CELERY_WORKER"] = "false"
    
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            workers=workers if not reload else 1
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        cleanup_resources()
        sys.exit(1)

if __name__ == "__main__":
    main()