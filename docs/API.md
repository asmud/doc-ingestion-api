# API Reference

The Document Ingestion API provides endpoints for processing documents from files and URLs using advanced OCR and document understanding capabilities.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. The API is designed for internal use within trusted networks.

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API health status |
| GET | `/formats` | Get supported document formats |
| POST | `/process/upload` | Process uploaded files (sync) |
| POST | `/process/upload/async` | Process uploaded files (async) |
| POST | `/process/url` | Process document from URL (sync) |
| POST | `/process/url/async` | Process document from URL (async) |
| POST | `/process/urls/batch/async` | Process multiple URLs (async) |
| POST | `/process/urls/crawl/async` | Crawl and process website (async) |
| GET | `/jobs/{job_id}/status` | Get job status |
| GET | `/jobs/{job_id}/result` | Get job result |

## Health Check

### GET /health

Check if the API service is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "message": "Document Ingestion API is running",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

## Supported Formats

### GET /formats

Get a list of supported document formats.

**Response:**
```json
{
  "supported_formats": ["PDF", "DOCX", "PPTX", "XLSX", "HTML", "TXT", "CSV", "MD"],
  "total_formats": 8,
  "status": "success"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/formats"
```

## Document Processing

### Processing Modes

- **`text_only`**: Document conversion and formatting only (text extraction)
- **`chunks_only`**: Document chunking without full formatting (faster, optimized for RAG)
- **`embedding`**: Generate document-level and chunk-level embeddings using nomic-embed-text-v1.5
- **`full`**: All features - text conversion, chunking, and embedding generation (comprehensive, default)

### Output Formats

- **`json`**: Structured JSON output (default)
- **`markdown`**: Markdown formatted text
- **`text`**: Plain text output  
- **`html`**: HTML formatted output

### Content Fields Explanation

The API returns two distinct content fields:

- **`content`**: Raw extracted text from the document (always plain text, regardless of output_format)
- **`formatted_content`**: Content formatted according to the requested `output_format` (JSON/Markdown/HTML/Text)

This allows you to:
- Use `content` for further processing, search, or analysis (consistent format)
- Use `formatted_content` for display or consumption in the desired format

### Embedding Fields (in `embedding` and `full` modes)

- **`document_embedding`**: 768-dimensional vector representing the entire document
- **`embedding_dimension`**: Size of embedding vectors (768)
- **`embedding_model`**: Model used for embeddings ("nomic-embed-text-v1.5")
- **`chunks[].embedding`**: 768-dimensional vector for each individual chunk

### POST /process/upload

Process one or more uploaded document files synchronously.

**Parameters:**
- `files` (required): Document file(s) to process
- `output_format` (optional): Output format (`json`, `markdown`, `text`, `html`)
- `processing_mode` (optional): Processing mode (`text_only`, `chunks_only`, `embedding`, `full`)

**Example:**
```bash
curl -X POST "http://localhost:8000/process/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "output_format=json" \
  -F "processing_mode=full"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "file_extension": "PDF",
  "file_size": 1048576,
  "output_format": "json",
  "processing_mode": "full",
  "processed_content": {
    "content": "Raw extracted text content from document...",
    "formatted_content": {"text": "Document content...", "metadata": {}},
    "chunks": [
      {
        "chunk_id": 0,
        "text": "First chunk of document content...",
        "num_tokens": 150,
        "embedding": [0.1, -0.2, 0.3, ...],  // 768-dimensional vector
        "metadata": {
          "page_numbers": [1],
          "chapter": "Introduction",
          "section": null,
          "chunk_type": "text",
          "char_count": 485
        }
      }
    ],
    "total_chunks": 3,
    "document_embedding": [0.05, -0.12, 0.08, ...],  // 768-dimensional vector
    "embedding_dimension": 768,
    "embedding_model": "nomic-embed-text-v1.5",
    "word_count": 1250,
    "char_count": 8456,
    "extraction_method": "docling",
    "processing_mode": "full"
  },
  "status": "success",
  "timestamp": 1705312200.123
}
```

### POST /process/upload/async

Process uploaded files asynchronously using background jobs.

**Parameters:** Same as `/process/upload`

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "PENDING",
  "message": "Document file processing job submitted successfully",
  "task_type": "document_file",
  "submitted_at": 1705312200.123,
  "estimated_completion": 1705312260.123
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/process/upload/async" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "output_format=json" \
  -F "processing_mode=full"
```

### POST /process/url

Process a document from a URL synchronously.

**Request Body:**
```json
{
  "url": "https://example.com/document.pdf",
  "output_format": "json",
  "processing_mode": "full"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/process/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "output_format": "json",
    "processing_mode": "full"
  }'
```

### POST /process/url/async

Process a document from URL asynchronously.

**Request Body:** Same as `/process/url`

**Response:** Same as other async endpoints (job submission response)

### POST /process/urls/batch/async

Process multiple documents from URLs simultaneously.

**Request Body:**
```json
{
  "urls": [
    "https://example.com/doc1.pdf",
    "https://example.com/doc2.pdf"
  ],
  "output_format": "json",
  "processing_mode": "full"
}
```

**Limits:**
- Maximum 20 URLs per batch request
- 30 second timeout per URL

**Example:**
```bash
curl -X POST "http://localhost:8000/process/urls/batch/async" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/doc1.pdf", "https://example.com/doc2.pdf"],
    "output_format": "json",
    "processing_mode": "full"
  }'
```

### POST /process/urls/crawl/async

Crawl a website and process all discovered pages.

**Request Body:**
```json
{
  "base_url": "https://docs.example.com",
  "max_depth": 2,
  "same_domain_only": true,
  "output_format": "json",
  "processing_mode": "full",
  "max_pages": 25
}
```

**Parameters:**
- `base_url`: Starting URL for crawling
- `max_depth`: Maximum crawl depth (1-5 levels)
- `same_domain_only`: Restrict to same domain
- `max_pages`: Maximum pages to process (1-100)

**Limits:**
- Maximum depth: 5 levels
- Maximum pages: 100 per crawl
- Rate limiting: 1 second delay between requests

**Example:**
```bash
curl -X POST "http://localhost:8000/process/urls/crawl/async" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://docs.example.com",
    "max_depth": 2,
    "same_domain_only": true,
    "output_format": "json",
    "processing_mode": "full",
    "max_pages": 25
  }'
```

## Job Management

All async operations return a job ID that can be used to track progress and retrieve results.

### GET /jobs/{job_id}/status

Get the current status of a background job.

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "SUCCESS",
  "message": "Job completed successfully",
  "progress": 100.0,
  "current": 0,
  "total": 1,
  "result_available": true,
  "error": null,
  "retry_count": null,
  "timestamp": 1705312260.123
}
```

**Job Statuses:**
- `PENDING`: Job is waiting to be processed
- `STARTED`: Job is currently being processed
- `SUCCESS`: Job completed successfully
- `FAILURE`: Job failed with an error
- `RETRY`: Job is being retried after failure

**Example:**
```bash
curl -X GET "http://localhost:8000/jobs/abc123-def456-ghi789/status"
```

### GET /jobs/{job_id}/result

Get the result of a completed background job.

**Parameters:**
- `timeout` (optional): Seconds to wait for completion

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "SUCCESS",
  "success": true,
  "result": {
    // Same structure as synchronous processing responses
  },
  "completed_at": 1705312260.123
}
```

**Example:**
```bash
# Get result immediately
curl -X GET "http://localhost:8000/jobs/abc123-def456-ghi789/result"

# Wait up to 30 seconds for completion
curl -X GET "http://localhost:8000/jobs/abc123-def456-ghi789/result?timeout=30"
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

**400 Bad Request:**
```json
{
  "detail": "Invalid file format or request parameters"
}
```

**404 Not Found:**
```json
{
  "detail": "Job not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error: detailed error message"
}
```

## Rate Limits

- No explicit rate limits currently implemented
- Website crawling respects robots.txt and implements 1-second delays
- Batch operations have size limits (20 URLs max)

## Examples

### Complete Workflow Example

1. **Submit async job:**
```bash
JOB_ID=$(curl -s -X POST "http://localhost:8000/process/url/async" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf", "output_format": "json"}' \
  | jq -r '.job_id')
```

2. **Check status:**
```bash
curl -X GET "http://localhost:8000/jobs/$JOB_ID/status"
```

3. **Get result:**
```bash
curl -X GET "http://localhost:8000/jobs/$JOB_ID/result" | jq '.'
```

### Batch Processing Example

```bash
curl -X POST "http://localhost:8000/process/urls/batch/async" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/report1.pdf",
      "https://example.com/report2.pdf",
      "https://example.com/webpage.html"
    ],
    "output_format": "markdown",
    "processing_mode": "full"
  }'
```

For more examples and advanced usage, see the [Development Guide](DEVELOPMENT.md).