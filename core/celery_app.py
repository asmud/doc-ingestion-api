from celery import Celery
from celery.signals import worker_shutdown, worker_ready, worker_process_init
from core.config import CeleryConfig
from core.logging_config import setup_logging
from typing import List, Dict, Any
import time
import signal
import logging
import gc

# Setup Celery worker logging
setup_logging("celery")
logger = logging.getLogger(__name__)

celery_config = CeleryConfig()
celery_app = Celery('doc_ingestion')

celery_app.conf.update(
    broker_url=celery_config.broker_url,
    result_backend=celery_config.result_backend,
    task_serializer=celery_config.task_serializer,
    result_serializer=celery_config.result_serializer,
    accept_content=celery_config.accept_content,
    timezone=celery_config.timezone,
    enable_utc=celery_config.enable_utc,
    task_routes=celery_config.task_routes,
    worker_prefetch_multiplier=celery_config.worker_prefetch_multiplier,
    task_acks_late=celery_config.task_acks_late,
    worker_disable_rate_limits=celery_config.worker_disable_rate_limits,
    task_soft_time_limit=celery_config.task_soft_time_limit,
    task_time_limit=celery_config.task_time_limit,
    task_max_retries=celery_config.task_max_retries,
    task_default_retry_delay=celery_config.task_default_retry_delay,
    # Graceful shutdown settings
    worker_hijack_root_logger=False,
    worker_log_color=False,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=1000,  # Restart workers periodically to prevent memory leaks
)

# Global pipeline instance for cleanup
_pipeline_instance = None

def cleanup_pipeline_resources():
    """Clean up pipeline resources to prevent memory leaks"""
    global _pipeline_instance
    try:
        if _pipeline_instance:
            if hasattr(_pipeline_instance, 'cleanup'):
                logger.info("Cleaning up pipeline resources...")
                _pipeline_instance.cleanup()
            _pipeline_instance = None
        
        # Force garbage collection
        gc.collect()
        logger.info("Pipeline resources cleaned up")
        
    except Exception as e:
        logger.warning(f"Error cleaning up pipeline resources: {e}")

@worker_process_init.connect
def worker_process_init_handler(sender=None, **kwargs):
    """Initialize worker process"""
    logger.info("Celery worker process initialized")

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Worker ready signal handler"""
    logger.info("Celery worker is ready to accept tasks")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Worker shutdown signal handler"""
    logger.info("Celery worker shutdown initiated")
    cleanup_pipeline_resources()

def get_or_create_pipeline():
    """Get or create pipeline instance with proper cleanup tracking"""
    global _pipeline_instance
    if _pipeline_instance is None:
        from core.pipeline import DocumentIntelligencePipeline
        _pipeline_instance = DocumentIntelligencePipeline()
        logger.info("Created new pipeline instance")
    return _pipeline_instance

@celery_app.task(bind=True, name='process_document_file_task')
def process_document_file_task(self, file_data: bytes, filename: str, output_format: str = "json", processing_mode: str = "full"):
    try:
        from processor.doc import process_document_file_from_bytes
        result = process_document_file_from_bytes(file_data, filename, output_format, processing_mode)
        return result
    except Exception as exc:
        logger.error(f"Task failed: {exc}")
        self.retry(countdown=celery_config.task_default_retry_delay, exc=exc)
    finally:
        # Periodic cleanup to prevent memory accumulation
        if hasattr(self, 'request') and self.request.retries == 0:
            gc.collect()

@celery_app.task(bind=True, name='process_document_url_task')
def process_document_url_task(self, url: str, output_format: str = "json", processing_mode: str = "full"):
    try:
        from processor.web import process_document_url
        import asyncio
        result = asyncio.run(process_document_url(url, output_format, processing_mode))
        return result
    except Exception as exc:
        logger.error(f"URL task failed: {exc}")
        self.retry(countdown=celery_config.task_default_retry_delay, exc=exc)
    finally:
        if hasattr(self, 'request') and self.request.retries == 0:
            gc.collect()

@celery_app.task(bind=True, name='process_batch_files_task')
def process_batch_files_task(self, files_data: List[Dict[str, Any]], output_format: str = "json", processing_mode: str = "full"):
    try:
        from processor.doc import process_document_files_batch_from_data
        result = process_document_files_batch_from_data(files_data, output_format, processing_mode)
        return result
    except Exception as exc:
        logger.error(f"Batch files task failed: {exc}")
        self.retry(countdown=celery_config.task_default_retry_delay, exc=exc)
    finally:
        if hasattr(self, 'request') and self.request.retries == 0:
            gc.collect()

@celery_app.task(bind=True, name='process_batch_urls_task')
def process_batch_urls_task(self, urls: List[str], output_format: str = "json", processing_mode: str = "full"):
    try:
        from processor.web import process_document_urls_batch
        import asyncio
        result = asyncio.run(process_document_urls_batch(urls, output_format, processing_mode))
        return result
    except Exception as exc:
        logger.error(f"Batch URLs task failed: {exc}")
        self.retry(countdown=celery_config.task_default_retry_delay, exc=exc)
    finally:
        if hasattr(self, 'request') and self.request.retries == 0:
            gc.collect()

@celery_app.task(bind=True, name='crawl_website_task')
def crawl_website_task(self, base_url: str, max_depth: int = 2, same_domain_only: bool = True, output_format: str = "json", processing_mode: str = "full", max_pages: int = 50):
    try:
        from processor.web import crawl_website
        import asyncio
        result = asyncio.run(crawl_website(base_url, max_depth, same_domain_only, output_format, processing_mode, max_pages))
        return result
    except Exception as exc:
        logger.error(f"Crawl task failed: {exc}")
        self.retry(countdown=celery_config.task_default_retry_delay, exc=exc)
    finally:
        if hasattr(self, 'request') and self.request.retries == 0:
            gc.collect()

@celery_app.task(name='health_check_task')
def health_check_task():
    return {
        "status": "healthy",
        "message": "Celery worker is running",
        "timestamp": time.time()
    }

if __name__ == '__main__':
    celery_app.start()