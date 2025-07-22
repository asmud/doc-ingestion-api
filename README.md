# Document Ingestion API

A high-performance FastAPI-based document processing service that extracts and structures content from various document formats using IBM's Docling pipeline with advanced OCR capabilities.

## Features

- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, HTML, TXT, CSV, MD, AUDIO
- **Advanced OCR**: Indonesian + English support via EasyOCR and Tesseract
- **Intelligent Processing**: Layout detection, figure classification, formula recognition
- **Flexible Output**: JSON, Markdown, Text, HTML formats
- **Async Processing**: Background jobs with Celery and Redis
- **Web Crawling**: Recursive website processing with depth control
- **Smart Chunking**: Semantic text segmentation for RAG applications

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Run the API
python main.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Upload and Process Files
```bash
curl -X POST "http://localhost:8000/process/upload" \
  -F "files=@document.pdf" \
  -F "output_format=json"
```

### Process URL Asynchronously
```bash
# Submit job
JOB_ID=$(curl -s -X POST "http://localhost:8000/process/url/async" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/doc.pdf"}' | jq -r '.job_id')

# Check status
curl "http://localhost:8000/jobs/$JOB_ID/status"

# Get result
curl "http://localhost:8000/jobs/$JOB_ID/result"
```

### Website Crawling
```bash
curl -X POST "http://localhost:8000/process/urls/crawl/async" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://docs.example.com",
    "max_depth": 2,
    "max_pages": 25
  }'
```

## Processing Modes

- **`full`**: Complete document processing with formatted output
- **`chunks_only`**: Fast chunking for RAG applications (optimized for vector databases)
- **`both`**: Full processing + chunking for comprehensive analysis

## Configuration

Key environment variables in `.env`:

```env
# Device selection
DEVICE=cpu                    # cpu, cuda, or mps

# OCR engine
DEFAULT_OCR_ENGINE=easyocr    # easyocr or tesseract

# Processing parameters  
CHUNK_SIZE=6500              # Tokens per chunk
PROCESSING_MODE=full         # full, chunks_only, both

# Redis connection
CELERY_BROKER_URL=redis://localhost:6379/9
```

## Architecture

- **FastAPI**: REST API with async support
- **Celery**: Background task processing
- **Redis**: Job queue and result storage
- **Docling**: IBM's document AI pipeline
- **Local Models**: No external API dependencies

## Documentation

- **[API Reference](docs/API.md)** - Complete endpoint documentation
- **[Setup Guide](docs/SETUP.md)** - Detailed installation instructions
- **[Models Guide](docs/MODELS.md)** - ML model configuration and optimization
- **[Development Guide](docs/DEVELOPMENT.md)** - Development workflow and debugging
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment strategies
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components

## System Requirements

- **Python**: 3.8+ (3.11 recommended)
- **Memory**: 8GB+ RAM (models require ~3GB)
- **Storage**: 10GB+ for models and processing
- **Redis**: 6.0+ for background jobs

## License

[License information here]