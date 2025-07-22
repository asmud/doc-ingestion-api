"""
Utility functions for the Document Ingestion API.
This module provides common utilities for timestamps, error handling, and validation.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Timestamp utilities
def get_current_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()

def get_current_iso_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

# Error handling utilities
def create_error_response(status_code: int, message: str, details: Optional[str] = None) -> HTTPException:
    """Create a standardized HTTPException."""
    error_detail = {"message": message}
    if details:
        error_detail["details"] = details
    error_detail["timestamp"] = get_current_iso_timestamp()
    
    return HTTPException(status_code=status_code, detail=error_detail)

def handle_processing_error(error: Exception, operation: str) -> HTTPException:
    """Handle common processing errors with standardized responses."""
    error_message = str(error)
    logger.error(f"Error during {operation}: {error_message}")
    
    if "not supported" in error_message.lower():
        return create_error_response(400, f"Unsupported format for {operation}", error_message)
    elif "not found" in error_message.lower() or "no such file" in error_message.lower():
        return create_error_response(404, f"Resource not found during {operation}", error_message)
    elif "permission" in error_message.lower() or "access" in error_message.lower():
        return create_error_response(403, f"Access denied during {operation}", error_message)
    else:
        return create_error_response(500, f"Internal error during {operation}", error_message)

# Response formatting utilities
def create_success_response(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    """Create a standardized success response."""
    return {
        "status": "success",
        "message": message,
        "timestamp": get_current_timestamp(),
        "data": data
    }

def create_job_response(job_id: str, status: str, message: str = "") -> Dict[str, Any]:
    """Create a standardized job response."""
    return {
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": get_current_timestamp()
    }

# Validation utilities
def validate_file_size(file_size: int, max_size: int = 50 * 1024 * 1024) -> None:
    """Validate file size (default max 50MB)."""
    if file_size > max_size:
        raise ValueError(f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)")

def validate_batch_size(batch_size: int, max_batch_size: int = 10) -> None:
    """Validate batch processing size."""
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size ({batch_size}) exceeds maximum allowed batch size ({max_batch_size})")
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

def validate_url(url: str) -> None:
    """Basic URL validation."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    if len(url) > 2048:
        raise ValueError("URL length exceeds maximum allowed length (2048 characters)")

# File format utilities
SUPPORTED_DOCUMENT_FORMATS = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', 
    '.md', '.txt', '.rtf', '.odt', '.ods', '.odp'
}

SUPPORTED_AUDIO_FORMATS = {
    '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus'
}

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return filename.lower().split('.')[-1] if '.' in filename else ''

def is_supported_document_format(filename: str) -> bool:
    """Check if the file format is supported for document processing."""
    ext = f".{get_file_extension(filename)}"
    return ext in SUPPORTED_DOCUMENT_FORMATS

def is_supported_audio_format(filename: str) -> bool:
    """Check if the file format is supported for audio processing."""
    ext = f".{get_file_extension(filename)}"
    return ext in SUPPORTED_AUDIO_FORMATS

def validate_file_format(filename: str, require_document: bool = True) -> None:
    """Validate file format."""
    if require_document:
        if not is_supported_document_format(filename):
            supported = ', '.join(sorted(SUPPORTED_DOCUMENT_FORMATS))
            raise ValueError(f"Unsupported document format. Supported formats: {supported}")
    else:
        if not (is_supported_document_format(filename) or is_supported_audio_format(filename)):
            all_supported = ', '.join(sorted(SUPPORTED_DOCUMENT_FORMATS | SUPPORTED_AUDIO_FORMATS))
            raise ValueError(f"Unsupported file format. Supported formats: {all_supported}")