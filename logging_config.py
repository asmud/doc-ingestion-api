"""
Centralized logging configuration for the Document Ingestion API.

This module provides unified logging setup for both FastAPI server and Celery workers,
ensuring consistent logging behavior between sync and async requests.
"""

import os
import logging
import sys
from typing import Dict, Optional
from pathlib import Path


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    def __init__(self):
        self.log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # Load log levels from environment
        self.server_log_level = os.getenv("SERVER_LOG_LEVEL", "info").upper()
        self.celery_log_level = os.getenv("CELERY_WORKER_LOG_LEVEL", "info").upper()
        self.asr_log_level = os.getenv("ASR_LOG_LEVEL", "info").upper()
        self.pipeline_log_level = os.getenv("PIPELINE_LOG_LEVEL", "info").upper()
        self.processing_log_level = os.getenv("PROCESSING_LOG_LEVEL", "info").upper()
        
        # Validate log levels
        self._validate_log_levels()
    
    def _validate_log_levels(self):
        """Validate that all log levels are valid."""
        levels_to_check = [
            self.server_log_level, self.celery_log_level, 
            self.asr_log_level, self.pipeline_log_level, self.processing_log_level
        ]
        
        for level in levels_to_check:
            if level not in self.log_levels:
                print(f"Warning: Invalid log level '{level}', falling back to INFO")
    
    def get_log_level(self, component: str = "server") -> int:
        """Get log level for a specific component."""
        level_map = {
            "server": self.server_log_level,
            "celery": self.celery_log_level,
            "asr": self.asr_log_level,
            "pipeline": self.pipeline_log_level,
            "processing": self.processing_log_level
        }
        
        level_name = level_map.get(component, self.server_log_level)
        return self.log_levels.get(level_name, logging.INFO)
    
    def create_formatter(self, include_process: bool = False) -> logging.Formatter:
        """Create a consistent log formatter."""
        if include_process:
            format_string = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        return logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def setup_logger(self, name: str, component: str = "server", 
                    include_process: bool = False) -> logging.Logger:
        """Setup a logger with consistent configuration."""
        logger = logging.getLogger(name)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set log level
        log_level = self.get_log_level(component)
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(self.create_formatter(include_process))
        
        logger.addHandler(handler)
        logger.propagate = False  # Prevent double logging
        
        return logger
    
    def setup_root_logging(self, component: str = "server"):
        """Setup root logging configuration."""
        log_level = self.get_log_level(component)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        # Reduce verbosity of noisy libraries
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Set specific component log levels
        if component == "celery":
            # For Celery workers, also configure component-specific loggers
            logging.getLogger("custom_asr").setLevel(self.get_log_level("asr"))
            logging.getLogger("pipeline").setLevel(self.get_log_level("pipeline"))
            logging.getLogger("processing_utils").setLevel(self.get_log_level("processing"))
    
    def get_celery_log_level_string(self) -> str:
        """Get Celery log level as string for command line."""
        return self.celery_log_level.lower()


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """Get the global logging configuration instance."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def setup_logging(component: str = "server") -> LoggingConfig:
    """
    Setup logging for the application.
    
    Args:
        component: The component type ("server", "celery", "asr", "pipeline", "processing")
    
    Returns:
        LoggingConfig instance
    """
    config = get_logging_config()
    config.setup_root_logging(component)
    return config


def get_logger(name: str, component: str = "server") -> logging.Logger:
    """
    Get a logger with centralized configuration.
    
    Args:
        name: Logger name (usually __name__)
        component: Component type for log level determination
    
    Returns:
        Configured logger instance
    """
    config = get_logging_config()
    return config.setup_logger(name, component, include_process=(component == "celery"))


# Convenience functions for different components
def get_server_logger(name: str) -> logging.Logger:
    """Get a logger for server components."""
    return get_logger(name, "server")


def get_celery_logger(name: str) -> logging.Logger:
    """Get a logger for Celery worker components."""
    return get_logger(name, "celery")


def get_asr_logger(name: str) -> logging.Logger:
    """Get a logger for ASR components."""
    return get_logger(name, "asr")


def get_pipeline_logger(name: str) -> logging.Logger:
    """Get a logger for pipeline components."""
    return get_logger(name, "pipeline")


def get_processing_logger(name: str) -> logging.Logger:
    """Get a logger for processing components."""
    return get_logger(name, "processing")