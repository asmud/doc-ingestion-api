from fastapi import HTTPException, UploadFile, File, Query
from typing import List, Dict, Any, Optional, Union
import time
from utils.utils import get_current_iso_timestamp, handle_processing_error, create_error_response
from processor.doc import process_document_file, get_supported_formats
from processor.web import process_document_url, process_document_urls_batch, crawl_website
from core.models import (
    ProcessingMode,
    HealthResponse,
    DocumentResponse,
    WebDocumentResponse,
    BatchResponse,
    FormatsResponse,
    ErrorResponse,
    AsyncJobResponse,
    JobStatusResponse,
    JobResultResponse,
    DocumentURLRequest,
    DocumentURLsBatchRequest,
    DocumentCrawlRequest
)
from core.app import app

@app.get("/health",
         summary="Health Check",
         description="Check if the API service is running and healthy",
         tags=["Information"],
         response_model=HealthResponse,
         responses={
             200: {
                 "description": "Service is healthy and running",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "healthy",
                             "message": "Document Ingestion API is running",
                             "timestamp": "2024-01-15T10:30:00.123456"
                         }
                     }
                 }
             }
         })
def health_check():
    """
    Health check endpoint to verify the API service is running properly.
    
    Use this endpoint for monitoring and load balancer health checks.
    """
    health_status = {
        "status": "healthy",
        "message": "Document Ingestion API is running",
        "timestamp": get_current_iso_timestamp()
    }
    
    return health_status

@app.get("/formats",
         summary="Get Supported Formats",
         description="Get comprehensive list of document formats supported by the Docling pipeline",
         tags=["Information"],
         response_model=FormatsResponse,
         responses={
             200: {
                 "description": "Supported formats retrieved successfully",
                 "content": {
                     "application/json": {
                         "example": {
                             "supported_formats": ["PDF", "DOCX", "PPTX", "XLSX", "HTML", "TXT", "CSV", "MD"],
                             "total_formats": 8,
                             "status": "success"
                         }
                     }
                 }
             },
             500: {"description": "Error retrieving formats", "model": ErrorResponse}
         })
def get_supported_document_formats():
    """
    Get comprehensive list of document formats supported by the Docling pipeline.
    
    ## Supported Format Categories:
    
    ### Office Documents:
    - **PDF**: Portable Document Format (with OCR support)
    - **DOCX**: Microsoft Word documents
    - **PPTX**: Microsoft PowerPoint presentations
    - **XLSX**: Microsoft Excel spreadsheets
    
    ### Web and Text:
    - **HTML**: Web pages and HTML documents
    - **TXT**: Plain text files
    - **MD**: Markdown documents
    - **CSV**: Comma-separated values
    
    ### Additional Formats:
    The pipeline supports many additional formats through the Docling library.
    Check the returned list for the complete set of supported formats.
    
    ## Usage:
    Use this endpoint to:
    - Validate file formats before upload
    - Display supported formats in your UI
    - Check pipeline capabilities
    
    Returns the complete list of file extensions that can be processed by the system.
    """
    try:
        formats = get_supported_formats()
        return {
            "supported_formats": formats,
            "total_formats": len(formats),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving formats: {str(e)}")

@app.post("/process/url",
          summary="Process Document URL",
          description="Process a document directly from a URL using the Docling pipeline",
          tags=["Document Processing"],
          response_model=WebDocumentResponse,
          responses={
              200: {
                  "description": "Document from URL processed successfully",
                  "content": {
                      "application/json": {
                          "example": {
                              "url": "https://example.com/document.pdf",
                              "content_type": "application/pdf",
                              "content_length": 1048576,
                              "output_format": "json",
                              "chunked": False,
                              "processed_content": {
                                  "content": {"text": "Document content from URL...", "metadata": {}},
                                  "word_count": 1500,
                                  "char_count": 9876,
                                  "extraction_method": "docling_url"
                              },
                              "status": "success",
                              "timestamp": 1705312200.123
                          }
                      }
                  }
              },
              400: {"description": "Invalid URL or inaccessible document", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def process_document_from_url(request: DocumentURLRequest):
    """
    Process a document directly from a URL using the Docling pipeline.
    
    ## Supported URL Types:
    - **PDF Documents**: Direct PDF links from any accessible server
    - **HTML Pages**: Web pages with extractable content
    - **Office Documents**: Word, Excel, PowerPoint files accessible via URL
    - **Text Files**: Plain text documents
    - **Any Docling-supported format** accessible via HTTP/HTTPS
    
    ## Features:
    - Direct URL processing without download
    - Automatic content type detection
    - Smart content extraction based on document type
    - Optional intelligent chunking
    - Multiple output formats
    
    ## Usage Examples:
    ```json
    {
        "url": "https://example.com/report.pdf",
        "output_format": "markdown",
        "chunk_document": true
    }
    ```
    
    Returns comprehensive processing results including extracted content, metadata, and processing statistics.
    """
    try:
        result = await process_document_url(request.url, request.output_format, request.processing_mode)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/process/upload",
          summary="Process Document Files",
          description="Upload and process one or multiple document files using the Docling pipeline",
          tags=["Document Processing"],
          response_model=Union[DocumentResponse, BatchResponse],
          responses={
              200: {
                  "description": "Documents processed successfully",
                  "content": {
                      "application/json": {
                          "examples": {
                              "single_file": {
                                  "summary": "Single file processing result",
                                  "value": {
                                      "filename": "example.pdf",
                                      "file_extension": "PDF",
                                      "file_size": 1048576,
                                      "output_format": "json",
                                      "chunked": False,
                                      "processed_content": {
                                          "content": {"text": "Document content...", "metadata": {}},
                                          "word_count": 1250,
                                          "char_count": 8456,
                                          "extraction_method": "docling"
                                      },
                                      "status": "success",
                                      "timestamp": 1705312200.123
                                  }
                              },
                              "batch_files": {
                                  "summary": "Multiple files processing result",
                                  "value": {
                                      "batch_results": [
                                          {
                                              "filename": "doc1.pdf",
                                              "processed_content": {"content": "...", "extraction_method": "docling_batch"},
                                              "status": "success"
                                          }
                                      ],
                                      "total_files": 3,
                                      "successful_files": 3,
                                      "processing_method": "batch",
                                      "status": "success"
                                  }
                              }
                          }
                      }
                  }
              },
              400: {"description": "Invalid file format or request", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def process_document_upload(
    files: List[UploadFile] = File(..., description="Document file(s) to process. Supports multiple file upload."),
    output_format: str = Query("json", description="Output format for processed content", enum=["json", "markdown", "text", "html"]),
    processing_mode: str = Query("full", description="Processing mode", enum=["text_only", "chunks_only", "embedding", "full"])
):
    """
    Process uploaded document files using the advanced Docling pipeline.
    
    ## Features:
    - **Multi-format Support**: PDF, DOCX, PPTX, XLSX, HTML, and more
    - **Batch Processing**: Upload multiple files simultaneously for efficient processing
    - **Smart Chunking**: Optional intelligent document chunking with metadata preservation
    - **Multiple Output Formats**: JSON, Markdown, Text, or HTML
    - **OCR Capability**: Advanced OCR for scanned documents with multi-language support
    
    ## Upload Limits:
    - File size: No specific limit (depends on server configuration)
    - Number of files: No limit for batch processing
    - Supported formats: All Docling-compatible formats
    
    ## Processing Modes:
    - **text_only**: Document conversion and formatting only (text extraction)
    - **chunks_only**: Only chunk the document, skip text formatting (optimized for RAG)
    - **embedding**: Generate document-level and chunk-level embeddings using nomic-embed-text
    - **full** (default): All features - text conversion, chunking, and embedding generation
    
    ## Processing Options:
    - **output_format**: Choose how you want the extracted content formatted
    - **processing_mode**: Control what type of processing to perform
    
    Returns detailed processing results including extracted content, metadata, and statistics.
    """
    try:
        if len(files) == 1:
            # Single file processing
            result = await process_document_file(files[0], output_format, processing_mode)
            return result
        else:
            # Multiple files processing - use batch
            from processor.doc import process_document_files_batch
            result = await process_document_files_batch(files, output_format, processing_mode)
            return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/process/url/async",
          summary="Process Document URL (Async)",
          description="Process a document from URL using background jobs",
          tags=["Document Processing"],
          response_model=AsyncJobResponse,
          responses={
              202: {"description": "Job submitted successfully"},
              400: {"description": "Invalid URL or inaccessible document", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def process_document_from_url_async(request: DocumentURLRequest):
    """
    Submit a URL processing job to run in the background.
    
    This endpoint immediately returns a job ID for tracking progress.
    Use the job management endpoints to check status and retrieve results.
    """
    try:
        # Initialize Celery app and job manager if not done
        from core.celery_app import celery_app
        from utils.job_manager import init_job_manager, get_job_manager
        
        try:
            job_manager = get_job_manager()
        except RuntimeError:
            job_manager = init_job_manager(celery_app)
        
        # Submit URL processing job
        task = celery_app.send_task(
            'process_document_url_task',
            args=[request.url, request.output_format, request.processing_mode]
        )
        
        return {
            "job_id": task.id,
            "status": "PENDING",
            "message": "URL processing job submitted successfully",
            "task_type": "document_url",
            "submitted_at": time.time(),
            "estimated_completion": time.time() + 45  # Estimate 45 seconds
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

@app.post("/process/upload/async",
          summary="Process Document Files (Async)",
          description="Upload and process document files using background jobs",
          tags=["Document Processing"],
          response_model=AsyncJobResponse,
          responses={
              202: {"description": "Job submitted successfully"},
              400: {"description": "Invalid file format or request", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def process_document_upload_async(
    files: List[UploadFile] = File(..., description="Document file(s) to process. Supports multiple file upload."),
    output_format: str = Query("json", description="Output format for processed content", enum=["json", "markdown", "text", "html"]),
    processing_mode: str = Query("full", description="Processing mode", enum=["text_only", "chunks_only", "embedding", "full"])
):
    """
    Submit document processing jobs to run in the background.
    
    This endpoint immediately returns a job ID for tracking progress.
    Use the job management endpoints to check status and retrieve results.
    
    ## Benefits of Async Processing:
    - Non-blocking operations for large files
    - Better resource management
    - Progress tracking
    - Ability to cancel long-running jobs
    - Queue management for high throughput
    """
    try:
        # Initialize Celery app and job manager if not done
        from core.celery_app import celery_app
        from utils.job_manager import init_job_manager, get_job_manager
        
        try:
            job_manager = get_job_manager()
        except RuntimeError:
            job_manager = init_job_manager(celery_app)
        
        if len(files) == 1:
            # Single file processing
            file = files[0]
            file_content = await file.read()
            
            # Submit single file job
            task = celery_app.send_task(
                'process_document_file_task',
                args=[file_content, file.filename, output_format, processing_mode]
            )
            
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": "Document file processing job submitted successfully",
                "task_type": "document_file",
                "submitted_at": time.time(),
                "estimated_completion": time.time() + 60  # Estimate 1 minute
            }
        else:
            # Multiple files processing
            files_data = []
            for file in files:
                content = await file.read()
                files_data.append({
                    "data": content,
                    "filename": file.filename
                })
            
            # Submit batch job
            task = celery_app.send_task(
                'process_batch_files_task',
                args=[files_data, output_format, processing_mode]
            )
            
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": f"Batch file processing job submitted successfully ({len(files)} files)",
                "task_type": "batch_files",
                "submitted_at": time.time(),
                "estimated_completion": time.time() + (len(files) * 30)  # Estimate 30s per file
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

@app.post("/process/urls/batch/async",
          summary="Process Multiple Document URLs (Async)",
          description="Process multiple URLs using background jobs",
          tags=["Document Processing"],
          response_model=AsyncJobResponse,
          responses={
              202: {"description": "Job submitted successfully"},
              400: {"description": "Invalid URLs or batch size exceeded", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def process_document_urls_batch_async(request: DocumentURLsBatchRequest):
    """
    Submit a batch URL processing job to run in the background.
    
    This endpoint immediately returns a job ID for tracking progress.
    Use the job management endpoints to check status and retrieve results.
    """
    try:
        if len(request.urls) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 URLs allowed per batch request")
        
        # Initialize Celery app and job manager if not done
        from core.celery_app import celery_app
        from utils.job_manager import init_job_manager, get_job_manager
        
        try:
            job_manager = get_job_manager()
        except RuntimeError:
            job_manager = init_job_manager(celery_app)
        
        # Submit batch URL processing job
        task = celery_app.send_task(
            'process_batch_urls_task',
            args=[request.urls, request.output_format, request.processing_mode]
        )
        
        return {
            "job_id": task.id,
            "status": "PENDING",
            "message": f"Batch URL processing job submitted successfully ({len(request.urls)} URLs)",
            "task_type": "batch_urls",
            "submitted_at": time.time(),
            "estimated_completion": time.time() + (len(request.urls) * 20)  # Estimate 20s per URL
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

@app.post("/process/urls/crawl/async",
          summary="Crawl and Process Website (Async)",
          description="Crawl a website and process pages using background jobs",
          tags=["Document Processing"],
          response_model=AsyncJobResponse,
          responses={
              202: {"description": "Job submitted successfully"},
              400: {"description": "Invalid crawl parameters", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          })
async def crawl_and_process_website_async(request: DocumentCrawlRequest):
    """
    Submit a website crawling job to run in the background.
    
    This endpoint immediately returns a job ID for tracking progress.
    Use the job management endpoints to check status and retrieve results.
    
    Note: Crawling jobs can take significantly longer than other operations.
    """
    try:
        # Initialize Celery app and job manager if not done
        from core.celery_app import celery_app
        from utils.job_manager import init_job_manager, get_job_manager
        
        try:
            job_manager = get_job_manager()
        except RuntimeError:
            job_manager = init_job_manager(celery_app)
        
        # Submit crawling job
        task = celery_app.send_task(
            'crawl_website_task',
            args=[
                request.base_url,
                request.max_depth,
                request.same_domain_only,
                request.output_format,
                request.processing_mode,
                request.max_pages
            ]
        )
        
        # Estimate time based on pages and depth
        estimated_pages = min(request.max_pages, (2 ** request.max_depth) * 5)
        estimated_time = estimated_pages * 15  # 15 seconds per page
        
        return {
            "job_id": task.id,
            "status": "PENDING",
            "message": f"Website crawling job submitted successfully (depth: {request.max_depth}, max pages: {request.max_pages})",
            "task_type": "website_crawl",
            "submitted_at": time.time(),
            "estimated_completion": time.time() + estimated_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

@app.get("/jobs/{job_id}/status",
         summary="Get Job Status",
         description="Get the current status of a background job",
         tags=["Job Management"],
         response_model=JobStatusResponse,
         responses={
             200: {"description": "Job status retrieved successfully"},
             404: {"description": "Job not found", "model": ErrorResponse},
             500: {"description": "Internal server error", "model": ErrorResponse}
         })
def get_job_status(job_id: str):
    """
    Get the current status and progress of a background job.
    
    Use this endpoint to check:
    - Job completion status
    - Processing progress
    - Error information if job failed
    - Result availability
    
    ## Job Statuses:
    - **PENDING**: Job is waiting to be processed
    - **STARTED**: Job is currently being processed  
    - **SUCCESS**: Job completed successfully
    - **FAILURE**: Job failed with an error
    - **RETRY**: Job is being retried after failure
    - **REVOKED**: Job was cancelled
    """
    try:
        from utils.job_manager import get_job_manager
        job_manager = get_job_manager()
        return job_manager.get_job_status(job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")

@app.get("/jobs/{job_id}/result",
         summary="Get Job Result",
         description="Get the result of a completed background job",
         tags=["Job Management"],
         response_model=JobResultResponse,
         responses={
             200: {"description": "Job result retrieved successfully"},
             202: {"description": "Job not yet completed"},
             404: {"description": "Job not found", "model": ErrorResponse},
             500: {"description": "Internal server error", "model": ErrorResponse}
         })
def get_job_result(job_id: str, timeout: Optional[int] = Query(None, description="Optional timeout in seconds to wait for completion")):
    """
    Get the result of a background job.
    
    ## Usage:
    - For completed jobs: Returns the full processing result
    - For pending/running jobs: Returns status information
    - With timeout: Waits up to specified seconds for completion
    
    ## Response Codes:
    - **200**: Job completed, result available
    - **202**: Job still processing (when no timeout specified)
    - **404**: Job ID not found
    - **500**: Internal error
    """
    try:
        from utils.job_manager import get_job_manager
        job_manager = get_job_manager()
        return job_manager.get_job_result(job_id, timeout)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job result: {str(e)}")
