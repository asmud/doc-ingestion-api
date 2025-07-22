# Models Guide

This guide explains the machine learning models used by the Document Ingestion API and how to configure them.

## Overview

The Document Ingestion API uses several specialized models for different document processing tasks:

- **Layout Detection**: Identifies document structure (headers, paragraphs, tables, figures)
- **OCR (Optical Character Recognition)**: Extracts text from images and scanned documents
- **Text Embedding**: Creates vector representations for intelligent chunking
- **Figure Classification**: Identifies and classifies figures and images
- **Formula Detection**: Recognizes mathematical formulas and code blocks
- **Speech Recognition**: Processes audio content (when available)

## Model Storage

All models are stored locally in the `models/` directory to ensure:
- **Offline Operation**: No internet dependency after initial setup
- **Performance**: Faster model loading and inference
- **Privacy**: No data sent to external services
- **Reliability**: Consistent behavior across environments

```
models/
├── EasyOcr/                          # OCR models
│   ├── craft_mlt_25k.pth             # Text detection
│   ├── english_g2.pth                # English text recognition
│   └── latin_g2.pth                  # Latin script recognition
├── ds4sd--docling-models/            # Layout analysis models
│   ├── model_artifacts/
│   │   ├── layout/                   # Page layout detection
│   │   └── tableformer/             # Table structure detection
├── ds4sd--DocumentFigureClassifier/ # Figure classification
├── ds4sd--CodeFormula/              # Formula and code detection
├── nomic-embed-text-v1.5/           # Text embeddings for chunking
└── cahya--whisper-medium-id/        # Indonesian speech recognition
```

## Model Details

### 1. Layout Detection Models

**Model**: `ds4sd/docling-models`
**Purpose**: Detect page layout elements (text blocks, tables, figures, headers)
**Size**: ~500MB
**Languages**: Language-agnostic (works with any script)

**Configuration:**
```env
LAYOUT_MODEL_DIR=models/ds4sd--docling-models
```

**Features:**
- Page segmentation and reading order
- Table detection and structure analysis
- Figure and image region identification
- Header/footer recognition

### 2. OCR Models

#### EasyOCR (Default)

**Model**: Custom EasyOCR models
**Purpose**: Text extraction from images and scanned documents
**Size**: ~100MB
**Languages**: Indonesian (id) + English (en)

**Configuration:**
```env
DEFAULT_OCR_ENGINE=easyocr
EASYOCR_MODELS_DIR=models/EasyOcr
FORCE_FULL_PAGE_OCR=False
```

**Features:**
- Multi-language support (Indonesian + English)
- High accuracy on printed text
- GPU acceleration support
- Handles rotated and skewed text

#### Tesseract (Alternative)

**Purpose**: Alternative OCR engine
**Languages**: Indonesian + English (auto-detection available)

**Configuration:**
```env
DEFAULT_OCR_ENGINE=tesseract
```

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng

# macOS
brew install tesseract tesseract-lang

# CentOS/RHEL
sudo yum install tesseract tesseract-langpack-ind tesseract-langpack-eng
```

### 3. Text Embedding Model

**Model**: `nomic-ai/nomic-embed-text-v1.5`
**Purpose**: Create vector representations for intelligent document chunking
**Size**: ~500MB
**Context Length**: 8192 tokens

**Configuration:**
```env
CHUNKER_TOKENIZER=models/nomic-embed-text-v1.5
TOKENIZER_MODEL_DIR=models/nomic-embed-text-v1.5
```

**Features:**
- High-quality sentence embeddings
- Used for semantic chunking
- Optimized for retrieval tasks
- Supports long documents

### 4. Figure Classification Model

**Model**: `ds4sd/DocumentFigureClassifier`
**Purpose**: Classify figures and images in documents
**Size**: ~100MB

**Configuration:**
```env
FIGURE_CLASSIFIER_MODEL_DIR=models/ds4sd--DocumentFigureClassifier
```

**Supported Figure Types:**
- Charts and graphs
- Diagrams
- Photographs
- Technical drawings
- Tables (visual)

### 5. Formula Detection Model

**Model**: `ds4sd/CodeFormula`
**Purpose**: Detect and extract mathematical formulas and code blocks
**Size**: ~500MB

**Configuration:**
```env
CODE_FORMULA_MODEL_DIR=models/ds4sd--CodeFormula
```

**Features:**
- Mathematical equation detection
- Code block identification
- LaTeX formula extraction
- Programming language recognition

### 6. Speech Recognition Model

**Model**: `cahya/whisper-medium-id`
**Purpose**: Indonesian speech recognition for audio content
**Size**: ~1.5GB
**Language**: Indonesian (Bahasa Indonesia)

**Configuration:**
```env
WHISPER_MODEL_DIR=models/cahya--whisper-medium-id
WHISPER_MODEL_NAME=cahya/whisper-medium-id
ASR_LANGUAGE=id
WHISPER_MAX_NEW_TOKENS=420
WHISPER_CHUNK_DURATION=30.0
WHISPER_CHUNK_OVERLAP=2.0
```

## Device Configuration

### CPU Processing (Default)

**Configuration:**
```env
DEVICE=cpu
```

**Pros:**
- Universal compatibility
- Stable and reliable
- Lower memory usage
- No special hardware requirements

**Cons:**
- Slower processing
- Higher CPU usage

**Recommended for:**
- Production deployments
- Limited GPU memory
- Maximum stability

### GPU Processing

#### CUDA (NVIDIA GPUs)

**Configuration:**
```env
DEVICE=cuda
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support

**Installation:**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Pros:**
- 2-5x faster processing
- Better for large documents
- Parallel processing

#### MPS (Apple Silicon Macs)

**Configuration:**
```env
DEVICE=mps
```

**Requirements:**
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3+
- PyTorch with MPS support

**Pros:**
- Optimized for Apple Silicon
- Good performance/power ratio
- Unified memory architecture

**Note:** MPS may have compatibility issues with some models in multiprocessing environments.

## Model Download and Setup

### Prerequisites

Before downloading models, ensure you have:

```bash
# Install required dependencies
pip install -r requirements.txt

# Create models directory
mkdir -p models
```

### Automatic Download (Recommended)

Models are automatically downloaded on first use:

```bash
# Start application - models download automatically
python main.py
```

The first startup will download all required models (~3GB total) to the `models/` directory.

### Manual Download Methods

#### Method 1: Download All Models

For offline environments or controlled updates:

```bash
# Download all models at once
python -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = 'False'
from pipeline import DocumentIntelligencePipeline
pipeline = DocumentIntelligencePipeline()
print('All models downloaded successfully')
"
```

#### Method 2: Download Individual Models

```python
# Download specific models using transformers
from transformers import AutoModel, AutoTokenizer
import os

# Set cache directory
cache_dir = 'models'
os.makedirs(cache_dir, exist_ok=True)

# Layout detection model (~500MB)
AutoModel.from_pretrained('ds4sd/docling-models', cache_dir=cache_dir)

# Figure classification model (~100MB)
AutoModel.from_pretrained('ds4sd/DocumentFigureClassifier', cache_dir=cache_dir)

# Formula detection model (~500MB)
AutoModel.from_pretrained('ds4sd/CodeFormula', cache_dir=cache_dir)

# Text embedding tokenizer (~500MB)
AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5', cache_dir=cache_dir)

# Speech recognition model (~1.5GB)
AutoModel.from_pretrained('cahya/whisper-medium-id', cache_dir=cache_dir)

print("All models downloaded successfully!")
```

#### Method 3: Download Using HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download models individually
huggingface-cli download ds4sd/docling-models --local-dir models/ds4sd--docling-models
huggingface-cli download ds4sd/DocumentFigureClassifier --local-dir models/ds4sd--DocumentFigureClassifier
huggingface-cli download ds4sd/CodeFormula --local-dir models/ds4sd--CodeFormula
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir models/nomic-embed-text-v1.5
huggingface-cli download cahya/whisper-medium-id --local-dir models/cahya--whisper-medium-id
```

#### Method 4: Download EasyOCR Models

EasyOCR models are downloaded automatically, but you can pre-download them:

```python
import easyocr
import os

# Set custom model directory
os.environ['EASYOCR_MODULE_PATH'] = 'models/EasyOcr'

# Initialize EasyOCR with Indonesian and English
reader = easyocr.Reader(['id', 'en'], model_storage_directory='models/EasyOcr')
print("EasyOCR models downloaded successfully!")
```

### Download Script

Create a dedicated download script:

```bash
# Create download_models.py
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
"""
Download all required models for Document Ingestion API
"""
import os
import sys
from pathlib import Path

def download_models():
    """Download all required models"""
    print("Downloading Document Inges models...")
    
    # Set offline mode to False
    os.environ['TRANSFORMERS_OFFLINE'] = 'False'
    os.environ['HF_HUB_OFFLINE'] = 'False'
    
    try:
        # Import after setting environment
        from transformers import AutoModel, AutoTokenizer
        import easyocr
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Download transformers models
        models_to_download = [
            ('ds4sd/docling-models', 'Layout detection'),
            ('ds4sd/DocumentFigureClassifier', 'Figure classification'),
            ('ds4sd/CodeFormula', 'Formula detection'),
            ('nomic-ai/nomic-embed-text-v1.5', 'Text embeddings'),
            ('cahya/whisper-medium-id', 'Speech recognition')
        ]
        
        for model_name, description in models_to_download:
            print(f"Downloading {description} model: {model_name}")
            try:
                if 'nomic-embed' in model_name:
                    AutoTokenizer.from_pretrained(model_name, cache_dir='models')
                else:
                    AutoModel.from_pretrained(model_name, cache_dir='models')
                print(f"✓ {description} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        
        # Download EasyOCR models
        print("Downloading EasyOCR models...")
        try:
            os.environ['EASYOCR_MODULE_PATH'] = str(models_dir / 'EasyOcr')
            reader = easyocr.Reader(['id', 'en'], 
                                  model_storage_directory=str(models_dir / 'EasyOcr'),
                                  verbose=False)
            print("✓ EasyOCR models downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download EasyOCR models: {e}")
        
        print("\n🎉 All models downloaded successfully!")
        print(f"Models stored in: {models_dir.absolute()}")
        
        # Show disk usage
        import subprocess
        try:
            result = subprocess.run(['du', '-sh', str(models_dir)], 
                                  capture_output=True, text=True)
            print(f"Total size: {result.stdout.strip().split()[0]}")
        except:
            pass
            
    except ImportError as e:
        print(f"✗ Missing dependencies. Please run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
EOF

# Make executable and run
chmod +x download_models.py
python download_models.py
```

### Verify Download

Check that all models were downloaded correctly:

```bash
# Check models directory structure
find models/ -name "*.pth" -o -name "*.safetensors" -o -name "config.json" | head -10

# Check total size
du -sh models/

# Verify with pipeline
python -c "
from pipeline import DocumentIntelligencePipeline
pipeline = DocumentIntelligencePipeline()
print('✓ All models loaded successfully!')
"
```

## Performance Optimization

### Memory Management

**For systems with limited memory:**
```env
DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
FORCE_FULL_PAGE_OCR=False
```

**For high-memory systems:**
```env
DEVICE=cuda  # or mps
CHUNK_SIZE=6500
CHUNK_OVERLAP=1640
FORCE_FULL_PAGE_OCR=True
```

### Processing Speed

**Optimize for speed:**
```env
DEFAULT_OCR_ENGINE=tesseract  # Faster than EasyOCR
FORCE_FULL_PAGE_OCR=False     # Skip OCR when not needed
CHUNK_SIZE=2000               # Larger chunks = fewer operations
```

**Optimize for accuracy:**
```env
DEFAULT_OCR_ENGINE=easyocr    # More accurate
FORCE_FULL_PAGE_OCR=True      # Always run OCR
CHUNK_SIZE=500                # Smaller chunks = better precision
```

## Troubleshooting

### Model Loading Issues

**Problem**: Models fail to load or download
```
Error: Could not load model from models/ds4sd--docling-models
```

**Solutions:**
```bash
# Clear cache and retry
rm -rf models/
rm -rf ~/.cache/huggingface/
TRANSFORMERS_OFFLINE=False python main.py

# Check internet connectivity
ping huggingface.co

# Manual download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ds4sd/docling-models')"
```

### Memory Issues

**Problem**: Out of memory errors
```
CUDA out of memory
```

**Solutions:**
```bash
# Switch to CPU
echo "DEVICE=cpu" >> .env

# Reduce batch size/chunk size
echo "CHUNK_SIZE=500" >> .env

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### OCR Issues

**Problem**: Poor text extraction quality
```
OCR producing garbled text
```

**Solutions:**
```bash
# Try different OCR engine
echo "DEFAULT_OCR_ENGINE=tesseract" >> .env

# Force full page OCR
echo "FORCE_FULL_PAGE_OCR=True" >> .env

# Check image quality and resolution
```

### Performance Issues

**Problem**: Slow processing
```
Documents taking too long to process
```

**Solutions:**
```bash
# Enable GPU acceleration
echo "DEVICE=cuda" >> .env  # or mps for Mac

# Increase worker concurrency
celery -A celery_app worker --concurrency=4

# Use faster OCR
echo "DEFAULT_OCR_ENGINE=tesseract" >> .env
```

## Model Customization

### Adding Custom Models

To add custom models for specific languages or domains:

1. **Create model directory:**
```bash
mkdir models/custom-model
```

2. **Update configuration:**
```python
# In pipeline.py
class DocumentIntelligencePipeline:
    def __init__(self):
        # Load custom model
        self.custom_model = AutoModel.from_pretrained('models/custom-model')
```

3. **Update environment:**
```env
CUSTOM_MODEL_DIR=models/custom-model
```

### Model Versioning

Track model versions for reproducibility:

```bash
# Create version file
echo "v1.0.0" > models/VERSION

# Log model checksums
find models/ -name "*.pth" -o -name "*.safetensors" | xargs sha256sum > models/CHECKSUMS
```

For more advanced configuration and deployment options, see:
- [Setup Guide](SETUP.md) - Installation and basic configuration
- [Architecture Guide](ARCHITECTURE.md) - System design and model integration
- [Deployment Guide](DEPLOYMENT.md) - Production deployment strategies