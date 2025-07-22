# Setup Guide

This guide provides detailed instructions for installing and configuring the Document Ingestion API.

## Requirements

### System Requirements

- **Python**: 3.8 or higher (3.11 recommended)
- **Redis**: 6.0 or higher (required for background job processing)
- **Memory**: 8GB+ RAM recommended (4GB minimum)
- **Storage**: 10GB+ free space for models and processing

### Platform Support

- **macOS**: Full support (Intel and Apple Silicon)
- **Linux**: Full support (Ubuntu 20.04+, CentOS 8+)
- **Windows**: Limited support (WSL2 recommended)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd doc-ingestion
```

### 2. Python Environment Setup

**Option A: Using pyenv (Recommended)**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install and use Python 3.11
pyenv install 3.11.13
pyenv local 3.11.13

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n doc-ingestion python=3.11
conda activate doc-ingestion
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install as development package (optional)
pip install -e .
```

### 4. Redis Installation

**macOS:**
```bash
# Using Homebrew
brew install redis
brew services start redis

# Or using MacPorts
sudo port install redis
sudo port load redis
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**CentOS/RHEL:**
```bash
sudo yum install epel-release
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis
```

**Verify Redis Installation:**
```bash
redis-cli ping
# Should return: PONG
```

## Configuration

### 1. Environment Variables

Copy the example environment file and customize:

```bash
# Copy environment file
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Key Configuration Options

#### API Configuration
```env
API_TITLE=Document Ingestion API
API_VERSION=1.0.0
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_RELOAD=false
SERVER_LOG_LEVEL=info
SERVER_WORKERS=1
```

#### Processing Configuration
```env
DEVICE=cpu                          # cpu, cuda, or mps (macOS)
DEFAULT_OCR_ENGINE=easyocr         # easyocr or tesseract
FORCE_FULL_PAGE_OCR=False
CHUNK_SIZE=6500                    # Token count per chunk
CHUNK_OVERLAP=1640                 # Token overlap between chunks
```

#### Celery Configuration
```env
CELERY_BROKER_URL=redis://localhost:6379/9
CELERY_RESULT_BACKEND=redis://localhost:6379/10
START_CELERY_WORKER=true           # Auto-start embedded worker
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_SOFT_TIME_LIMIT=900    # 15 minutes
CELERY_TASK_TIME_LIMIT=1200        # 20 minutes
```

#### Model Configuration
```env
MODELS_DIR=models
TRANSFORMERS_OFFLINE=True
HF_DATASETS_OFFLINE=True
HF_HOME=models
HF_HUB_CACHE=models
```

#### CORS Configuration
```env
CORS_ORIGINS=*                     # * for all, or comma-separated URLs
CORS_CREDENTIALS=true
```

### 3. Model Setup

The application uses several local models that need to be downloaded:

#### Automatic Model Download
Models will be automatically downloaded on first use if they don't exist locally:

```bash
# Start the application - models will download automatically
python main.py
```

#### Manual Model Setup
For offline environments or faster startup, download models manually:

```bash
# Create models directory
mkdir -p models

# Download required models (this may take time)
python -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = 'False'
from pipeline import DocumentIntelligencePipeline
pipeline = DocumentIntelligencePipeline()
print('Models downloaded successfully')
"
```

#### Model Storage Locations

Models are stored in the `models/` directory:

```
models/
├── EasyOcr/                    # OCR models (Indonesian + English)
├── ds4sd--docling-models/      # Layout detection models
├── ds4sd--DocumentFigureClassifier/  # Figure classification
├── ds4sd--CodeFormula/         # Formula detection
├── nomic-embed-text-v1.5/     # Text embedding (for chunking)
└── cahya--whisper-medium-id/   # Indonesian ASR model
```

## Starting the Application

### Development Mode

**Single Process (with embedded Celery worker):**
```bash
python main.py
```

**With auto-reload (development):**
```bash
SERVER_RELOAD=true python main.py
```

**Manual worker management:**
```bash
# Terminal 1: Start API server without embedded worker
START_CELERY_WORKER=false python main.py

# Terminal 2: Start Celery worker manually
celery -A celery_app worker --loglevel=info --concurrency=2
```

### Production Mode

**Using Gunicorn:**
```bash
# Install gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

**Using Docker (see DEPLOYMENT.md for details):**
```bash
docker build -t doc-ingestion .
docker run -p 8000:8000 doc-ingestion
```

## Verification

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Document Ingestion API is running",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 2. Process Test Document
```bash
# Test with a sample PDF
curl -X POST "http://localhost:8000/process/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample.pdf" \
  -F "output_format=json"
```

### 3. Test Async Processing
```bash
# Submit async job
JOB_ID=$(curl -s -X POST "http://localhost:8000/process/url/async" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}' | jq -r '.job_id')

# Check status
curl http://localhost:8000/jobs/$JOB_ID/status

# Get result
curl http://localhost:8000/jobs/$JOB_ID/result
```

## Troubleshooting

### Common Issues

#### Redis Connection Error
```
Error: Redis connection failed
```
**Solution:**
```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis-server  # Linux
brew services restart redis          # macOS
```

#### Model Download Issues
```
Error: Model not found or download failed
```
**Solutions:**
```bash
# Clear model cache
rm -rf models/
rm -rf ~/.cache/huggingface/

# Retry with internet connection
TRANSFORMERS_OFFLINE=False python main.py

# Or download specific model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ds4sd/docling-models')"
```

#### Memory Issues
```
Error: CUDA out of memory / Killed (OOM)
```
**Solutions:**
```bash
# Force CPU usage
echo "DEVICE=cpu" >> .env

# Reduce chunk size
echo "CHUNK_SIZE=1000" >> .env

# Reduce worker concurrency
celery -A celery_app worker --concurrency=1
```

#### Permission Issues (Linux)
```bash
# Fix file permissions
chmod +x main.py
chown -R $USER:$USER models/

# Fix Redis permissions
sudo chown redis:redis /var/lib/redis/
```

### Log Analysis

**View application logs:**
```bash
# Real-time logs
tail -f logs/app.log

# Celery worker logs
tail -f logs/celery.log

# Filter by log level
grep "ERROR" logs/app.log
```

**Enable debug logging:**
```bash
# Temporary debug mode
SERVER_LOG_LEVEL=debug python main.py

# Permanent debug mode
echo "SERVER_LOG_LEVEL=debug" >> .env
```

### Performance Tuning

#### For High Throughput
```env
# Increase worker concurrency
CELERY_WORKER_CONCURRENCY=4

# Use multiple server workers (production)
SERVER_WORKERS=4

# Optimize chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

#### For Low Memory Systems
```env
# Force CPU processing
DEVICE=cpu

# Reduce chunk size
CHUNK_SIZE=500

# Single worker
CELERY_WORKER_CONCURRENCY=1
```

## Next Steps

- [API Reference](API.md) - Learn about available endpoints
- [Development Guide](DEVELOPMENT.md) - Set up development environment
- [Architecture Guide](ARCHITECTURE.md) - Understand system design
- [Deployment Guide](DEPLOYMENT.md) - Deploy to production