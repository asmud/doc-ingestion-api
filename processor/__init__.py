# Processor module - avoid circular imports by not importing functions that depend on core.pipeline
# Functions that use DocumentIntelligencePipeline will import it locally when needed

from .embedding import EmbeddingService
# Import processor modules without executing them to avoid circular imports

__all__ = ["EmbeddingService"]