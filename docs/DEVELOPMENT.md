# Development Guide

This guide covers the development workflow, debugging, and testing for the Document Ingestion API.

## Development Environment Setup

### Prerequisites

- Python 3.8+ (3.11 recommended)
- Redis server running locally
- Git for version control
- IDE/Editor with Python support

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd doc-ingestion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start Redis (if not running)
redis-server  # or brew services start redis

# Start development server
python main.py
```

## Development Commands

### Server Management

```bash
# Development server with auto-reload
SERVER_RELOAD=true python main.py

# Start API server only (no embedded worker)
START_CELERY_WORKER=false python main.py

# Start with debug logging
SERVER_LOG_LEVEL=debug python main.py

# Custom host/port
SERVER_HOST=127.0.0.1 SERVER_PORT=8001 python main.py
```

### Celery Worker Management

```bash
# Start worker manually with verbose logging
celery -A celery_app worker --loglevel=debug

# Worker with specific concurrency
celery -A celery_app worker --concurrency=2

# Monitor worker activity
celery -A celery_app inspect active
celery -A celery_app inspect stats
celery -A celery_app inspect registered

# Purge all tasks (development only)
celery -A celery_app purge
```

## Project Structure

```
doc-ingestion/
├── main.py                 # Application entry point
├── app.py                  # FastAPI application setup
├── api.py                  # API endpoint definitions
├── pipeline.py             # Core document processing pipeline
├── celery_app.py          # Celery configuration and tasks
├── models.py              # Pydantic models and schemas
├── config.py              # Configuration management
├── job_manager.py         # Background job utilities
├── doc.py                 # File processing logic
├── web.py                 # URL/web processing logic
├── processing_utils.py    # Shared processing utilities
├── utils.py               # General utility functions
├── logging_config.py      # Logging configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── .env.example          # Environment template
├── CLAUDE.md             # Claude Code instructions
├── docs/                 # Documentation
│   ├── API.md           # API reference
│   ├── SETUP.md         # Installation guide
│   ├── MODELS.md        # Model configuration
│   ├── DEVELOPMENT.md   # This file
│   ├── DEPLOYMENT.md    # Production deployment
│   └── ARCHITECTURE.md  # System architecture
└── models/              # Local ML models storage
```

## Code Architecture

### Core Components

#### Pipeline (`pipeline.py`)
- **DocumentIntelligencePipeline**: Main processing engine
- Wraps IBM Docling for document understanding
- Handles OCR, layout detection, and content extraction
- Manages model loading and device configuration

#### API Layer (`api.py`)
- FastAPI endpoints for synchronous and asynchronous processing
- Request/response validation using Pydantic models
- Error handling and HTTP status management

#### Background Processing (`celery_app.py`)
- Celery tasks for async document processing
- Redis-backed job queue and result storage
- Progress tracking and error handling

#### Job Management (`job_manager.py`)
- Utilities for job status tracking
- Result retrieval and timeout handling

### Processing Flow

1. **Request Reception**: API endpoints receive files or URLs
2. **Validation**: Input validation using Pydantic models
3. **Job Submission**: Async requests create Celery tasks
4. **Processing**: Pipeline processes documents using local models
5. **Result Storage**: Results stored in Redis with job ID
6. **Response**: Client polls for status and retrieves results

## Debugging

### Logging Configuration

The application uses structured logging with different levels:

```python
# Enable debug logging
SERVER_LOG_LEVEL=debug python main.py

# View real-time logs
tail -f logs/app.log
tail -f logs/celery.log

# Filter logs by level
grep "ERROR" logs/app.log
grep "WARNING" logs/app.log
```

### Common Debug Scenarios

#### Model Loading Issues

```bash
# Check model directory
ls -la models/

# Test model loading directly
python -c "
from pipeline import DocumentIntelligencePipeline
pipeline = DocumentIntelligencePipeline()
print('✅ Pipeline initialized successfully')
"

# Clear model cache
rm -rf models/
rm -rf ~/.cache/huggingface/
```

#### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
redis-cli monitor

# View active connections
redis-cli CLIENT LIST
```

#### Worker Process Issues

```bash
# Debug worker startup
celery -A celery_app worker --loglevel=debug

# Check worker status
celery -A celery_app inspect ping

# Monitor task execution
celery -A celery_app events
```

### Memory Debugging

```bash
# Monitor memory usage
ps aux | grep python
htop  # or top

# Force CPU processing to reduce memory
echo "DEVICE=cpu" >> .env

# Reduce chunk size
echo "CHUNK_SIZE=1000" >> .env
```

## Testing

### Manual API Testing

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Synchronous Processing
```bash
# Upload file
curl -X POST "http://localhost:8000/process/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test.pdf" \
  -F "output_format=json"

# Process URL
curl -X POST "http://localhost:8000/process/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/doc.pdf"}'
```

#### Asynchronous Processing
```bash
# Submit async job
JOB_ID=$(curl -s -X POST "http://localhost:8000/process/upload/async" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test.pdf" \
  | jq -r '.job_id')

# Check status
curl "http://localhost:8000/jobs/$JOB_ID/status"

# Get result
curl "http://localhost:8000/jobs/$JOB_ID/result"
```

### Load Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils  # Ubuntu
brew install apache-bench       # macOS

# Basic load test
ab -n 100 -c 10 http://localhost:8000/health

# Upload load test (requires multipart support)
# Use tools like wrk or hey for complex POST requests
```

### Integration Testing

Create test scripts for different scenarios:

```python
# test_integration.py
import requests
import time
import json

def test_async_processing():
    # Submit job
    response = requests.post(
        "http://localhost:8000/process/url/async",
        json={"url": "https://example.com/test.pdf"}
    )
    assert response.status_code == 200
    
    job_id = response.json()["job_id"]
    
    # Poll for completion
    max_attempts = 30
    for _ in range(max_attempts):
        status_response = requests.get(f"http://localhost:8000/jobs/{job_id}/status")
        status = status_response.json()["status"]
        
        if status == "SUCCESS":
            break
        elif status == "FAILURE":
            raise Exception("Job failed")
        
        time.sleep(2)
    
    # Get result
    result_response = requests.get(f"http://localhost:8000/jobs/{job_id}/result")
    assert result_response.status_code == 200

if __name__ == "__main__":
    test_async_processing()
    print("✅ Integration test passed")
```

## Performance Optimization

### CPU vs GPU Processing

```bash
# CPU processing (stable, lower memory)
echo "DEVICE=cpu" >> .env

# GPU processing (faster, requires more memory)
echo "DEVICE=cuda" >> .env  # NVIDIA
echo "DEVICE=mps" >> .env   # Apple Silicon
```

### Memory Optimization

```bash
# Reduce memory usage
echo "CHUNK_SIZE=1000" >> .env
echo "CHUNK_OVERLAP=100" >> .env

# Single worker for memory-constrained systems
celery -A celery_app worker --concurrency=1
```

### Processing Speed

```bash
# Faster OCR engine
echo "DEFAULT_OCR_ENGINE=tesseract" >> .env

# Skip OCR when possible
echo "FORCE_FULL_PAGE_OCR=False" >> .env

# Larger chunks for batch processing
echo "CHUNK_SIZE=2000" >> .env
```

## Code Style and Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Maintain consistent import ordering
- Use descriptive variable and function names

### Error Handling

```python
# Good: Specific error handling
try:
    result = pipeline.process_file(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    return {"error": "File not found"}
except Exception as e:
    logger.error(f"Processing failed: {e}")
    return {"error": "Processing failed"}

# Good: Use utility functions
from utils import handle_processing_error

try:
    result = process_document()
except Exception as e:
    raise handle_processing_error(e, "document processing")
```

### Logging Best Practices

```python
# Good: Structured logging with context
logger.info(f"Processing document: {filename} (size: {file_size} bytes)")
logger.error(f"Failed to process {filename}: {error_message}")

# Good: Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning condition")
logger.error("Error occurred")
```

## Environment Management

### Development vs Production

```bash
# Development environment
SERVER_RELOAD=true
SERVER_LOG_LEVEL=debug
DEVICE=cpu  # For stability

# Production environment
SERVER_RELOAD=false
SERVER_LOG_LEVEL=info
SERVER_WORKERS=4
DEVICE=cuda  # If GPU available
```

### Model Management

```bash
# Download models for offline development
TRANSFORMERS_OFFLINE=False python -c "
from pipeline import DocumentIntelligencePipeline
DocumentIntelligencePipeline()
"

# Switch to offline mode
echo "TRANSFORMERS_OFFLINE=True" >> .env
echo "HF_DATASETS_OFFLINE=True" >> .env
```

## Troubleshooting Development Issues

### Common Problems

#### "Redis connection refused"
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
redis-server
# or
brew services start redis
sudo systemctl start redis
```

#### "ModuleNotFoundError"
```bash
# Install in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### "CUDA out of memory"
```bash
# Switch to CPU processing
echo "DEVICE=cpu" >> .env

# Or reduce batch size
echo "CHUNK_SIZE=500" >> .env
```

#### "Worker not receiving tasks"
```bash
# Check Celery broker URL
echo $CELERY_BROKER_URL

# Restart worker
celery -A celery_app worker --purge
```

### Debug Environment

Create a debug configuration file:

```bash
# .env.debug
API_TITLE=Document Ingestion API (Debug)
SERVER_RELOAD=true
SERVER_LOG_LEVEL=debug
DEVICE=cpu
CHUNK_SIZE=1000
START_CELERY_WORKER=true
CELERY_TASK_ALWAYS_EAGER=false
```

## Contributing

When contributing to the project:

1. **Create feature branches**: `git checkout -b feature/new-feature`
2. **Write tests**: Add tests for new functionality
3. **Update documentation**: Keep docs in sync with code changes
4. **Follow code style**: Maintain consistent formatting and conventions
5. **Test thoroughly**: Ensure all existing functionality still works

For more information on deployment and architecture, see:
- [Setup Guide](SETUP.md) - Installation and basic configuration
- [Architecture Guide](ARCHITECTURE.md) - System design and components
- [Deployment Guide](DEPLOYMENT.md) - Production deployment strategies