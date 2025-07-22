#!/usr/bin/env python3

import os
import signal
import sys
import warnings
import uvicorn
from dotenv import load_dotenv

# Fix for macOS ML library issues - must be set before importing anything else
if os.uname().sysname == 'Darwin':  # macOS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # Prevent MPS initialization in forked processes
    os.environ['PYTORCH_MPS_ALLOCATOR_DISABLE'] = '1'
from core.logging_config import setup_logging
from core.app import app, cleanup_resources
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
            "core.app:app",
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