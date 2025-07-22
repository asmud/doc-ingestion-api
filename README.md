# Document Ingestion API

A high-performance FastAPI-based document processing service that extracts and structures content from various document formats using IBM's Docling pipeline with advanced OCR capabilities.

## Features

- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, HTML, TXT, CSV, MD, AUDIO
- **Advanced OCR**: Indonesian + English support via EasyOCR and Tesseract
- **Intelligent Processing**: Layout detection, figure classification, formula recognition
- **Semantic Embeddings**: Document and chunk-level embeddings using nomic-embed-text-v1.5
- **Flexible Output**: JSON, Markdown, Text, HTML formats
- **Async Processing**: Background jobs with Celery and Redis
- **Web Crawling**: Recursive website processing with depth control
- **Smart Chunking**: Semantic text segmentation for RAG applications
- **Modular Architecture**: Clean separation of core, processor, and utility modules

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

- **`text_only`**: Document conversion and formatting only (text extraction)
- **`chunks_only`**: Fast chunking for RAG applications (optimized for vector databases)
- **`embedding`**: Generate document-level and chunk-level embeddings using nomic-embed-text
- **`full`**: All features - text conversion, chunking, and embedding generation (comprehensive)

## Configuration

Key environment variables in `.env`:

```env
# Device selection
DEVICE=cpu                    # cpu, cuda, or mps

# OCR engine
DEFAULT_OCR_ENGINE=easyocr    # easyocr or tesseract

# Processing parameters  
CHUNK_SIZE=200               # Tokens per chunk
PROCESSING_MODE=full         # text_only, chunks_only, embedding, full
EMBEDDING_DIMENSION=768      # Embedding vector size
EMBEDDING_BATCH_SIZE=32      # Batch size for embeddings

# Redis connection
CELERY_BROKER_URL=redis://localhost:6379/9
```

## Architecture

```
doc-intelligence/
├── main.py              # Application entry point
├── core/                # Core application logic
│   ├── app.py          # FastAPI configuration
│   ├── api.py          # REST API endpoints
│   ├── pipeline.py     # Document processing pipeline
│   ├── models.py       # Pydantic models
│   └── config.py       # Configuration management
├── processor/           # Document processing modules
│   ├── doc.py          # File upload processing
│   ├── web.py          # URL/web crawling
│   ├── custom_asr.py   # Audio processing
│   └── embedding.py    # Semantic embeddings
├── utils/               # Shared utilities
│   ├── utils.py        # Common functions
│   ├── processing_utils.py  # Processing helpers
│   └── job_manager.py  # Background job management
└── docs/               # Documentation
```

**Technology Stack:**
- **FastAPI**: REST API with async support
- **Celery**: Background task processing
- **Redis**: Job queue and result storage
- **Docling**: IBM's document AI pipeline
- **nomic-embed-text**: Semantic embeddings
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