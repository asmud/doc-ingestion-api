from celery.result import AsyncResult
from celery import Celery
from typing import Dict, Any, List, Optional, Union
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"

class JobManager:
    def __init__(self, celery_app: Celery):
        self.celery_app = celery_app
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            
            status_info = {
                "job_id": job_id,
                "status": result.status,
                "current": getattr(result, 'current', 0),
                "total": getattr(result, 'total', 1),
                "timestamp": time.time()
            }
            
            if result.status == JobStatus.PENDING:
                status_info.update({
                    "message": "Job is waiting to be processed",
                    "progress": 0
                })
            elif result.status == JobStatus.STARTED:
                status_info.update({
                    "message": "Job is currently being processed",
                    "progress": (result.current / result.total) * 100 if result.total > 0 else 0
                })
            elif result.status == JobStatus.SUCCESS:
                status_info.update({
                    "message": "Job completed successfully",
                    "progress": 100,
                    "result_available": True
                })
            elif result.status == JobStatus.FAILURE:
                status_info.update({
                    "message": f"Job failed: {str(result.info)}",
                    "progress": 0,
                    "error": str(result.info)
                })
            elif result.status == JobStatus.RETRY:
                status_info.update({
                    "message": "Job is being retried",
                    "progress": 0,
                    "retry_count": getattr(result.info, 'retries', 0) if result.info else 0
                })
            elif result.status == JobStatus.REVOKED:
                status_info.update({
                    "message": "Job was cancelled",
                    "progress": 0
                })
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return {
                "job_id": job_id,
                "status": "ERROR",
                "message": f"Error retrieving job status: {str(e)}",
                "timestamp": time.time()
            }
    
    def get_job_result(self, job_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the result of a completed job.
        
        Args:
            job_id: The job ID to get results for
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing job result
        """
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            
            if result.status == JobStatus.SUCCESS:
                return {
                    "job_id": job_id,
                    "status": result.status,
                    "result": result.result,
                    "completed_at": time.time(),
                    "success": True
                }
            elif result.status == JobStatus.FAILURE:
                return {
                    "job_id": job_id,
                    "status": result.status,
                    "error": str(result.info),
                    "failed_at": time.time(),
                    "success": False
                }
            elif result.status in [JobStatus.PENDING, JobStatus.STARTED, JobStatus.RETRY]:
                # Try to wait for result if timeout is specified
                if timeout:
                    try:
                        job_result = result.get(timeout=timeout)
                        return {
                            "job_id": job_id,
                            "status": JobStatus.SUCCESS,
                            "result": job_result,
                            "completed_at": time.time(),
                            "success": True
                        }
                    except Exception as e:
                        return {
                            "job_id": job_id,
                            "status": result.status,
                            "message": "Job not yet completed",
                            "error": str(e) if "timeout" not in str(e).lower() else "Job timeout",
                            "success": False
                        }
                else:
                    return {
                        "job_id": job_id,
                        "status": result.status,
                        "message": "Job not yet completed",
                        "success": False
                    }
            else:
                return {
                    "job_id": job_id,
                    "status": result.status,
                    "message": f"Job in status: {result.status}",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error getting job result for {job_id}: {e}")
            return {
                "job_id": job_id,
                "status": "ERROR",
                "error": f"Error retrieving job result: {str(e)}",
                "success": False
            }

# Global job manager instance (will be initialized with celery app)
job_manager: Optional[JobManager] = None

def init_job_manager(celery_app: Celery) -> JobManager:
    """
    Initialize the global job manager with a Celery app.
    
    Args:
        celery_app: The Celery application instance
        
    Returns:
        Initialized JobManager instance
    """
    global job_manager
    job_manager = JobManager(celery_app)
    return job_manager

def get_job_manager() -> JobManager:
    """
    Get the global job manager instance.
    
    Returns:
        JobManager instance
        
    Raises:
        RuntimeError: If job manager hasn't been initialized
    """
    if job_manager is None:
        raise RuntimeError("Job manager not initialized. Call init_job_manager() first.")
    return job_manager