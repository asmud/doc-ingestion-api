import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    device: str = os.getenv("DEVICE", "cpu")
    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))
    easyocr_models_dir: Path = Path(os.getenv("EASYOCR_MODELS_DIR", "models/EasyOcr"))
    code_formula_model_dir: Path = Path(os.getenv("CODE_FORMULA_MODEL_DIR", "models/ds4sd--CodeFormula"))
    figure_classifier_model_dir: Path = Path(os.getenv("FIGURE_CLASSIFIER_MODEL_DIR", "models/ds4sd--DocumentFigureClassifier"))
    layout_model_dir: Path = Path(os.getenv("LAYOUT_MODEL_DIR", "models/ds4sd--docling-models"))
    default_ocr_engine: str = os.getenv("DEFAULT_OCR_ENGINE", "easyocr")
    force_full_page_ocr: bool = os.getenv("FORCE_FULL_PAGE_OCR", "").lower() in ("true", "1", "yes")
    asr_language: str = os.getenv("ASR_LANGUAGE", "auto")  # Default to auto-detection
    whisper_max_new_tokens: int = int(os.getenv("WHISPER_MAX_NEW_TOKENS", "420"))
    whisper_chunk_duration: float = float(os.getenv("WHISPER_CHUNK_DURATION", "20.0"))
    whisper_chunk_overlap: float = float(os.getenv("WHISPER_CHUNK_OVERLAP", "3.0"))
    transformers_offline: bool = os.getenv("TRANSFORMERS_OFFLINE", "False").lower() in ("true", "1", "yes")
    hf_datasets_offline: bool = os.getenv("HF_DATASETS_OFFLINE", "False").lower() in ("true", "1", "yes")
    
    # Chunking configuration - use Field with default_factory to read env at runtime
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "200")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "20")))
    chunker_tokenizer: str = Field(default_factory=lambda: os.getenv("CHUNKER_TOKENIZER", "models/nomic-embed-text-v1.5"))
    merge_peers: bool = Field(default_factory=lambda: os.getenv("MERGE_PEERS", "True").lower() in ("true", "1", "yes"))
    
    # Tokenizer model directory
    tokenizer_model_dir: Path = Path(os.getenv("TOKENIZER_MODEL_DIR", "models/nomic-embed-text-v1.5"))
    
    # Embedding configuration
    embedding_model_path: Path = Field(default_factory=lambda: Path(os.getenv("EMBEDDING_MODEL_PATH", "models/nomic-embed-text-v1.5")))
    embedding_dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")))
    embedding_batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    
    # Whisper ASR model directory
    whisper_model_dir: Path = Path(os.getenv("WHISPER_MODEL_DIR", "models/cahya--whisper-medium-id"))
    whisper_model_name: str = os.getenv("WHISPER_MODEL_NAME", "cahya/whisper-medium-id")
    
    @classmethod
    def from_project_root(cls, project_root: Optional[Path] = None) -> "ModelConfig":
        if project_root is None:
            project_root = Path.cwd()
        
        return cls(
            models_dir=project_root / "models",
            easyocr_models_dir=project_root / "models" / "EasyOcr",
            code_formula_model_dir=project_root / "models" / "ds4sd--CodeFormula",
            figure_classifier_model_dir=project_root / "models" / "ds4sd--DocumentFigureClassifier",
            layout_model_dir=project_root / "models" / "ds4sd--docling-models",
            tokenizer_model_dir=project_root / "models" / "nomic-embed-text-v1.5",
            embedding_model_path=project_root / "models" / "nomic-embed-text-v1.5",
            whisper_model_dir=project_root / "models" / "cahya--whisper-medium-id",
        )
    
    def validate_models_exist(self) -> bool:
        required_paths = [
            self.easyocr_models_dir,
            self.code_formula_model_dir,
            self.figure_classifier_model_dir,
            self.layout_model_dir,
            self.tokenizer_model_dir,
            # whisper_model_dir is optional since it uses HF cache structure
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Model directory not found: {path}")
        
        return True

class CeleryConfig(BaseModel):
    broker_url: str = Field(default_factory=lambda: os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
    result_backend: str = Field(default_factory=lambda: os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"))
    task_serializer: str = Field(default_factory=lambda: os.getenv("CELERY_TASK_SERIALIZER", "json"))
    result_serializer: str = Field(default_factory=lambda: os.getenv("CELERY_RESULT_SERIALIZER", "json"))
    accept_content: list = Field(default_factory=lambda: ["json"])
    timezone: str = Field(default_factory=lambda: os.getenv("CELERY_TIMEZONE", "UTC"))
    enable_utc: bool = Field(default_factory=lambda: os.getenv("CELERY_ENABLE_UTC", "True").lower() in ("true", "1", "yes"))
    task_routes: dict = Field(default_factory=lambda: {
        'process_document_file_task': {'queue': 'documents'},
        'process_document_url_task': {'queue': 'documents'},
        'process_batch_files_task': {'queue': 'batch'},
        'process_batch_urls_task': {'queue': 'batch'},
        'crawl_website_task': {'queue': 'crawl'}
    })
    
    # Worker configuration
    worker_prefetch_multiplier: int = Field(default_factory=lambda: int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1")))
    task_acks_late: bool = Field(default_factory=lambda: os.getenv("CELERY_TASK_ACKS_LATE", "True").lower() in ("true", "1", "yes"))
    worker_disable_rate_limits: bool = Field(default_factory=lambda: os.getenv("CELERY_WORKER_DISABLE_RATE_LIMITS", "False").lower() in ("true", "1", "yes"))
    
    # Task configuration
    task_soft_time_limit: int = Field(default_factory=lambda: int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "300")))  # 5 minutes
    task_time_limit: int = Field(default_factory=lambda: int(os.getenv("CELERY_TASK_TIME_LIMIT", "600")))  # 10 minutes
    task_max_retries: int = Field(default_factory=lambda: int(os.getenv("CELERY_TASK_MAX_RETRIES", "3")))
    task_default_retry_delay: int = Field(default_factory=lambda: int(os.getenv("CELERY_TASK_DEFAULT_RETRY_DELAY", "60")))  # 1 minute