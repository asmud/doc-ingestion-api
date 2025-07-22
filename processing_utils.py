"""
Shared processing utilities for document and web processing.
This module provides common processing patterns and pipeline management.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from logging_config import get_processing_logger
from pipeline import DocumentIntelligencePipeline
from utils import get_current_timestamp, handle_processing_error

logger = get_processing_logger(__name__)

# Singleton pipeline instance
_pipeline_instance: Optional[DocumentIntelligencePipeline] = None

def get_pipeline() -> DocumentIntelligencePipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        try:
            _pipeline_instance = DocumentIntelligencePipeline()
            logger.info("✅ Document Ingestion Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    return _pipeline_instance

def create_processing_result(
    processed_data: Dict[str, Any],
    filename: str,
    processing_mode: str,
    source_type: str = "file"
) -> Dict[str, Any]:
    """Create a standardized processing result."""
    return {
        "filename": filename,
        "source": source_type,
        "processing_mode": processing_mode,
        "processed_content": processed_data,
        "status": "success",
        "timestamp": get_current_timestamp()
    }

def create_error_result(
    filename: str,
    error: Exception,
    processing_mode: str,
    source_type: str = "file"
) -> Dict[str, Any]:
    """Create a standardized error result."""
    return {
        "filename": filename,
        "source": source_type,
        "processing_mode": processing_mode,
        "error": str(error),
        "status": "error",
        "timestamp": get_current_timestamp()
    }

def process_single_item(
    item_path: str,
    processing_mode: str,
    item_name: str,
    source_type: str = "file"
) -> Dict[str, Any]:
    """
    Generic processing function for single items (files or URLs).
    
    Args:
        item_path: Path to file or URL
        processing_mode: Processing mode (full, chunks_only, both)
        item_name: Display name for the item
        source_type: Type of source (file, url)
    
    Returns:
        Processing result dictionary
    """
    try:
        pipeline = get_pipeline()
        
        # Process the item
        if source_type == "file":
            processed_data = pipeline.process_file(item_path, processing_mode)
        else:  # URL
            processed_data = pipeline.process_url(item_path, processing_mode)
        
        return create_processing_result(
            processed_data, 
            item_name, 
            processing_mode, 
            source_type
        )
        
    except Exception as e:
        logger.error(f"Error processing {source_type} '{item_name}': {e}")
        return create_error_result(item_name, e, processing_mode, source_type)

def extract_processing_stats(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract processing statistics from results."""
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = total - successful
    
    return {
        "total_processed": total,
        "successful": successful,
        "failed": failed,
        "success_rate": round((successful / total * 100) if total > 0 else 0, 2)
    }

def cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup a temporary file."""
    try:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")

def validate_processing_mode(mode: str) -> str:
    """Validate and normalize processing mode."""
    valid_modes = {"full", "chunks_only", "both"}
    
    if mode not in valid_modes:
        raise ValueError(f"Invalid processing mode '{mode}'. Must be one of: {', '.join(valid_modes)}")
    
    return mode