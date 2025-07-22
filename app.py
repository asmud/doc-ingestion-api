import os
import signal
import threading
import subprocess
import logging
import atexit
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global references for cleanup
celery_worker_process = None
worker_thread = None
shutdown_event = threading.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Application startup initiated...")
    
    await startup_tasks()
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown initiated...")
    cleanup_resources()
    logger.info("Application shutdown completed")

app = FastAPI(
    title=os.getenv("API_TITLE", "Document Ingestion API"),
    version=os.getenv("API_VERSION", "1.0.0"),
    lifespan=lifespan
)

cors_origins = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if cors_origins == "*" else [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=os.getenv("CORS_CREDENTIALS", "true").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_resources():
    """Clean up resources and shutdown workers gracefully"""
    global celery_worker_process, worker_thread, shutdown_event
    
    logger.info("Starting graceful shutdown...")
    
    try:
        # Set shutdown event if it exists
        if shutdown_event:
            shutdown_event.set()
        
        # Signal Celery worker to shutdown gracefully
        if celery_worker_process and celery_worker_process.poll() is None:
            logger.info(f"Shutting down Celery worker (PID: {celery_worker_process.pid})...")
            
            # Try to shutdown process group first (for better cleanup)
            try:
                import os
                if os.name != 'nt':  # Unix-like systems
                    os.killpg(os.getpgid(celery_worker_process.pid), signal.SIGTERM)
                else:
                    celery_worker_process.terminate()
            except (ProcessLookupError, PermissionError, AttributeError):
                celery_worker_process.terminate()
            
            # Wait for graceful shutdown
            try:
                celery_worker_process.wait(timeout=3)
                logger.info("Celery worker shutdown gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Celery worker didn't shutdown gracefully, forcing termination")
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(celery_worker_process.pid), signal.SIGKILL)
                    else:
                        celery_worker_process.kill()
                except (ProcessLookupError, PermissionError, AttributeError):
                    celery_worker_process.kill()
                
                try:
                    celery_worker_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    logger.error("Failed to kill Celery worker completely")
        
        # Clean up pipeline resources
        try:
            from pipeline import DocumentIntelligencePipeline
            if hasattr(DocumentIntelligencePipeline, '_instance'):
                pipeline = DocumentIntelligencePipeline._instance
                if pipeline and hasattr(pipeline, 'cleanup'):
                    logger.info("Cleaning up pipeline resources...")
                    pipeline.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up pipeline: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Graceful shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


async def startup_tasks():
    """Application startup tasks"""
    global celery_worker_process, worker_thread
    
    try:
        from celery_app import celery_app
        from job_manager import init_job_manager
        from logging_config import setup_logging, get_logging_config
        
        # Setup server logging
        setup_logging("server")
        
        # Note: Don't register signal handlers here as uvicorn handles them
        # and will trigger the lifespan shutdown properly
        
        init_job_manager(celery_app)
        start_worker = os.getenv("START_CELERY_WORKER", "false").lower() == "true"
        
        if start_worker:
            def start_celery_worker():
                global celery_worker_process
                try:
                    logging_config = get_logging_config()
                    celery_log_level = logging_config.get_celery_log_level_string()
                    
                    worker_cmd = [
                        "celery", "-A", "celery_app", "worker",
                        f"--loglevel={celery_log_level}", "--concurrency=2",
                        "--queues=documents,batch,crawl"
                    ]
                    
                    logger.info("Starting embedded Celery worker...")
                    celery_worker_process = subprocess.Popen(
                        worker_cmd, 
                        cwd=os.getcwd(),
                        preexec_fn=os.setsid if os.name != 'nt' else None
                    )
                    
                    logger.info(f"Celery worker started with PID: {celery_worker_process.pid}")
                        
                except Exception as e:
                    logger.error(f"Error starting Celery worker: {e}")
            
            worker_thread = threading.Thread(target=start_celery_worker, daemon=True)
            worker_thread.start()
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


# Include API routes
import api
