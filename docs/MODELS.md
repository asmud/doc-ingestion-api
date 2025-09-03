# Models Guide

This guide explains the ONNX-optimized machine learning models used by the Document Ingestion API and how to configure them.

## Overview

The Document Ingestion API uses several specialized ONNX models for different document processing tasks, providing faster inference and reduced memory usage compared to traditional PyTorch models:

- **Layout Detection**: Identifies document structure (headers, paragraphs, tables, figures)
- **OCR (Optical Character Recognition)**: Extracts text from images and scanned documents
- **Text Embedding**: Creates vector representations for intelligent chunking
- **Figure Classification**: Identifies and classifies figures and images
- **Formula Detection**: Recognizes mathematical formulas and code blocks
- **Speech Recognition**: Processes audio content (when available)

## Model Storage

All models are stored locally as ONNX variants in the `models/` directory to ensure:
- **Offline Operation**: No internet dependency after initial setup
- **Performance**: Faster model loading and inference with ONNX Runtime
- **Privacy**: No data sent to external services
- **Reliability**: Consistent behavior across environments
- **Efficiency**: Reduced memory footprint and optimized execution

```
models/
├── EasyOcr-onnx/                           # ONNX OCR models
│   ├── craft_mlt_25k.onnx                 # Text detection
│   ├── english_g2.onnx                    # English text recognition
│   └── latin_g2.onnx                      # Latin script recognition
├── asmud--ds4sd-docling-models-onnx/      # ONNX Layout analysis models
│   ├── model_artifacts/
│   │   ├── layout/                        # Page layout detection
│   │   └── tableformer/                   # Table structure detection
├── asmud--ds4sd-DocumentClassifier-onnx/  # ONNX Figure classification
├── asmud--ds4sd-CodeFormula-onnx/          # ONNX Formula and code detection
├── asmud--LazarusNLP-indobert-onnx/        # Indonesian BERT embeddings
│   ├── model.onnx                         # ONNX model file
│   ├── tokenizer.json                     # Tokenizer configuration
│   └── config.json                        # Model configuration
└── asmud--cahya-whisper-medium-onnx/       # ONNX Indonesian speech recognition
```

## Model Details

### 1. Layout Detection Models

**Model**: `asmud/ds4sd-docling-models-onnx`
**Purpose**: Detect page layout elements (text blocks, tables, figures, headers)
**Size**: ~500MB (ONNX optimized)
**Languages**: Language-agnostic (works with any script)

**Configuration:**
```env
LAYOUT_MODEL_DIR=models/asmud--ds4sd-docling-models-onnx
```

**Features:**
- Page segmentation and reading order
- Table detection and structure analysis
- Figure and image region identification
- Header/footer recognition

### 2. OCR Models

#### EasyOCR (Default)

**Model**: Custom EasyOCR ONNX models
**Purpose**: Text extraction from images and scanned documents
**Size**: ~100MB (ONNX optimized)
**Languages**: Indonesian (id) + English (en)

**Configuration:**
```env
DEFAULT_OCR_ENGINE=easyocr
EASYOCR_MODELS_DIR=models/EasyOcr-onnx
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

**Model**: `asmud/LazarusNLP-indobert-onnx`
**Purpose**: Create vector representations for intelligent document chunking with Indonesian/English support
**Size**: ~300MB (ONNX optimized, reduced from PyTorch)
**Context Length**: 512 tokens
**Embedding Dimension**: 768

**Configuration:**
```env
CHUNKER_TOKENIZER=models/asmud--LazarusNLP-indobert-onnx
TOKENIZER_MODEL_DIR=models/asmud--LazarusNLP-indobert-onnx
EMBEDDING_MODEL_PATH=models/asmud--LazarusNLP-indobert-onnx
```

**Features:**
- High-quality Indonesian/English embeddings
- ONNX Runtime optimization for faster inference
- Used for semantic chunking
- Optimized for retrieval tasks
- Bilingual support (Indonesian/English)

### 4. Figure Classification Model

**Model**: `asmud/ds4sd-DocumentClassifier-onnx`
**Purpose**: Classify figures and images in documents
**Size**: ~100MB (ONNX optimized)

**Configuration:**
```env
FIGURE_CLASSIFIER_MODEL_DIR=models/asmud--ds4sd-DocumentClassifier-onnx
```

**Supported Figure Types:**
- Charts and graphs
- Diagrams
- Photographs
- Technical drawings
- Tables (visual)

### 5. Formula Detection Model

**Model**: `asmud/ds4sd-CodeFormula-onnx`
**Purpose**: Detect and extract mathematical formulas and code blocks
**Size**: ~500MB (ONNX optimized)

**Configuration:**
```env
CODE_FORMULA_MODEL_DIR=models/asmud--ds4sd-CodeFormula-onnx
```

**Features:**
- Mathematical equation detection
- Code block identification
- LaTeX formula extraction
- Programming language recognition

### 6. Speech Recognition Model

**Model**: `asmud/cahya-whisper-medium-onnx`
**Purpose**: Indonesian speech recognition for audio content
**Size**: ~800MB (ONNX optimized, reduced from PyTorch)
**Language**: Indonesian (Bahasa Indonesia)

**Configuration:**
```env
WHISPER_MODEL_DIR=models/asmud--cahya-whisper-medium-onnx
WHISPER_MODEL_NAME=asmud/cahya-whisper-medium-onnx
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
# Install required dependencies (includes ONNX Runtime)
pip install -r requirements.txt

# Create models directory
mkdir -p models
```

### ONNX Runtime Dependencies

The system now uses ONNX Runtime for optimized inference. Required packages are included in `requirements.txt`:

```bash
# Core ONNX dependencies
onnxruntime>=1.16.0  # CPU inference
onnxruntime-gpu>=1.16.0  # GPU inference (optional)
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

# Download ONNX models individually
huggingface-cli download asmud/ds4sd-docling-models-onnx --local-dir models/asmud--ds4sd-docling-models-onnx
huggingface-cli download asmud/ds4sd-DocumentClassifier-onnx --local-dir models/asmud--ds4sd-DocumentClassifier-onnx
huggingface-cli download asmud/ds4sd-CodeFormula-onnx --local-dir models/asmud--ds4sd-CodeFormula-onnx
huggingface-cli download asmud/LazarusNLP-indobert-onnx --local-dir models/asmud--LazarusNLP-indobert-onnx
huggingface-cli download asmud/cahya-whisper-medium-onnx --local-dir models/asmud--cahya-whisper-medium-onnx
```

#### Method 4: Download EasyOCR Models

EasyOCR models are downloaded automatically, but you can pre-download them:

```python
import easyocr
import os

# Set custom model directory
os.environ['EASYOCR_MODULE_PATH'] = 'models/EasyOcr-onnx'

# Initialize EasyOCR with Indonesian and English
reader = easyocr.Reader(['id', 'en'], model_storage_directory='models/EasyOcr-onnx')
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
            ('asmud/ds4sd-docling-models-onnx', 'Layout detection (ONNX)'),
            ('asmud/ds4sd-DocumentClassifier-onnx', 'Figure classification (ONNX)'),
            ('asmud/ds4sd-CodeFormula-onnx', 'Formula detection (ONNX)'),
            ('asmud/LazarusNLP-indobert-onnx', 'Indonesian BERT embeddings (ONNX)'),
            ('asmud/cahya-whisper-medium-onnx', 'Speech recognition (ONNX)')
        ]
        
        for model_name, description in models_to_download:
            print(f"Downloading {description} model: {model_name}")
            try:
                if 'indobert' in model_name:
                    AutoTokenizer.from_pretrained(model_name, cache_dir='models')
                else:
                    AutoModel.from_pretrained(model_name, cache_dir='models')
                print(f"✓ {description} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        
        # Download EasyOCR models
        print("Downloading EasyOCR models...")
        try:
            os.environ['EASYOCR_MODULE_PATH'] = str(models_dir / 'EasyOcr-onnx')
            reader = easyocr.Reader(['id', 'en'], 
                                  model_storage_directory=str(models_dir / 'EasyOcr-onnx'),
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

# Verify with ONNX pipeline
python -c "
from core.pipeline import DocumentIntelligencePipeline
pipeline = DocumentIntelligencePipeline()
print('✓ All ONNX models loaded successfully!')
"
```

## Performance Optimization

### Memory Management

**For systems with limited memory (ONNX optimized):**
```env
DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
FORCE_FULL_PAGE_OCR=False
# ONNX models use ~40% less memory than PyTorch
```

**For high-memory systems (ONNX optimized):**
```env
DEVICE=cuda  # or mps
CHUNK_SIZE=6500
CHUNK_OVERLAP=1640
FORCE_FULL_PAGE_OCR=True
# ONNX GPU acceleration provides additional performance
```

### Processing Speed

**Optimize for speed (ONNX benefits):**
```env
DEFAULT_OCR_ENGINE=tesseract  # Faster than EasyOCR
FORCE_FULL_PAGE_OCR=False     # Skip OCR when not needed
CHUNK_SIZE=2000               # Larger chunks = fewer operations
# ONNX provides 2-3x faster inference on CPU
```

**Optimize for accuracy (ONNX benefits):**
```env
DEFAULT_OCR_ENGINE=easyocr    # More accurate
FORCE_FULL_PAGE_OCR=True      # Always run OCR
CHUNK_SIZE=500                # Smaller chunks = better precision
# ONNX maintains accuracy while being faster
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