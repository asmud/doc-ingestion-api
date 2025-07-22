from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# Enums
class ProcessingMode(str, Enum):
    full = "full"
    chunks_only = "chunks_only" 
    both = "both"

# Response Models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    message: str = Field(..., description="Health status message")
    timestamp: str = Field(..., description="Timestamp of the health check")

class ProcessedContent(BaseModel):
    # Present in "full" and "both" modes
    content: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Processed document content (available in 'full' and 'both' modes)")
    formatted_content: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Formatted document content (available in 'full' and 'both' modes)")
    
    # Present in "chunks_only" and "both" modes  
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="Document chunks (available in 'chunks_only' and 'both' modes)")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks (available when chunks are present)")
    
    # Statistics (available in all modes, but content varies)
    word_count: Optional[int] = Field(None, description="Word count of the document")
    char_count: Optional[int] = Field(None, description="Character count of the document")
    
    # Metadata
    extraction_method: str = Field(..., description="Method used for content extraction")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")

class DocumentResponse(BaseModel):
    filename: Optional[str] = Field(None, description="Original filename")
    file_extension: Optional[str] = Field(None, description="File extension")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    url: Optional[str] = Field(None, description="Source URL if processed from URL")
    content_type: Optional[str] = Field(None, description="MIME content type")
    content_length: Optional[int] = Field(None, description="Content length in bytes")
    output_format: str = Field(..., description="Output format used")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    processed_content: ProcessedContent = Field(..., description="Processed document content and metadata")
    status: str = Field(..., description="Processing status")
    timestamp: Optional[float] = Field(None, description="Processing timestamp")

class WebDocumentResponse(BaseModel):
    url: Optional[str] = Field(None, description="Source URL if processed from URL")
    content_type: Optional[str] = Field(None, description="MIME content type")
    content_length: Optional[int] = Field(None, description="Content length in bytes")
    output_format: str = Field(..., description="Output format used")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    processed_content: ProcessedContent = Field(..., description="Processed document content and metadata")
    status: str = Field(..., description="Processing status")
    timestamp: Optional[float] = Field(None, description="Processing timestamp")

class BatchResponse(BaseModel):
    batch_results: List[DocumentResponse] = Field(..., description="Results for each processed item")
    total_files: Optional[int] = Field(None, description="Total number of files processed")
    total_urls: Optional[int] = Field(None, description="Total number of URLs processed")
    successful_files: Optional[int] = Field(None, description="Number of successfully processed files")
    successful_urls: Optional[int] = Field(None, description="Number of successfully processed URLs")
    output_format: str = Field(..., description="Output format used")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used for all documents")
    processing_method: str = Field(..., description="Processing method used")
    status: str = Field(..., description="Overall processing status")

class CrawlInfo(BaseModel):
    base_url: str = Field(..., description="Starting URL for crawling")
    max_depth: int = Field(..., description="Maximum crawl depth used")
    same_domain_only: bool = Field(..., description="Whether crawling was restricted to same domain")
    max_pages: int = Field(..., description="Maximum pages limit")
    discovered_urls: List[str] = Field(..., description="All discovered URLs during crawling")
    total_discovered: int = Field(..., description="Total number of discovered URLs")

class CrawlResponse(BaseModel):
    crawl_info: CrawlInfo = Field(..., description="Crawling operation details")
    processing_results: BatchResponse = Field(..., description="Processing results for all discovered pages")
    status: str = Field(..., description="Overall crawl status")
    timestamp: float = Field(..., description="Crawl completion timestamp")

class FormatsResponse(BaseModel):
    supported_formats: List[str] = Field(..., description="List of supported file formats")
    total_formats: int = Field(..., description="Total number of supported formats")
    status: str = Field(..., description="Response status")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")
    status_code: int = Field(..., description="HTTP status code")

# Async Job Response Models
class AsyncJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Job submission message")
    task_type: str = Field(..., description="Type of processing task")
    submitted_at: float = Field(..., description="Job submission timestamp")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion time")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    progress: float = Field(..., description="Job progress percentage")
    current: int = Field(0, description="Current progress counter")
    total: int = Field(1, description="Total progress counter")
    result_available: bool = Field(False, description="Whether result is available for retrieval")
    error: Optional[str] = Field(None, description="Error message if job failed")
    retry_count: Optional[int] = Field(None, description="Number of retry attempts")
    timestamp: float = Field(..., description="Status check timestamp")

class JobResultResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    success: bool = Field(..., description="Whether job completed successfully")
    result: Optional[Union[DocumentResponse, WebDocumentResponse, BatchResponse, CrawlResponse]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if job failed")
    completed_at: Optional[float] = Field(None, description="Job completion timestamp")
    failed_at: Optional[float] = Field(None, description="Job failure timestamp")


# Request Models
class DocumentURLRequest(BaseModel):
    url: str = Field(..., description="URL of the document to process")
    output_format: str = Field("json", description="Output format for the processed content")
    processing_mode: ProcessingMode = Field(ProcessingMode.full, description="Processing mode: 'full' (document only), 'chunks_only' (chunks only), 'both' (document + chunks)")

class DocumentURLsBatchRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to process (max 20)")
    output_format: str = Field("json", description="Output format for the processed content")
    processing_mode: ProcessingMode = Field(ProcessingMode.full, description="Processing mode: 'full' (document only), 'chunks_only' (chunks only), 'both' (document + chunks)")

class DocumentCrawlRequest(BaseModel):
    base_url: str = Field(..., description="Starting URL for web crawling")
    max_depth: int = Field(2, description="Maximum crawl depth (1-5 levels)", ge=1, le=5)
    same_domain_only: bool = Field(True, description="Whether to restrict crawling to the same domain")
    output_format: str = Field("json", description="Output format for the processed content")
    processing_mode: ProcessingMode = Field(ProcessingMode.full, description="Processing mode: 'full' (document only), 'chunks_only' (chunks only), 'both' (document + chunks)")
    max_pages: int = Field(50, description="Maximum number of pages to crawl (1-100)", ge=1, le=100)