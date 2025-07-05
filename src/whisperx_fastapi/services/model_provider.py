# src/whisperx_fastapi/services/model_provider.py
import torch
import whisperx
from contextlib import contextmanager, nullcontext
from loguru import logger
from whisperx_fastapi.core.config import settings
from whisperx_fastapi.services.model_manager import model_manager

# --- On-Demand Loading Logic (The "On-Demand" Strategy) ---
@contextmanager
def _load_on_demand(model_loader_func, *args, **kwargs):
    """Generic context manager for loading and offloading a model."""
    model_tuple = None
    model_name = kwargs.pop("model_key", "model") # For logging
    logger.info(f"On-Demand Mode: Loading {model_name} to device '{settings.DEVICE}'...")
    
    model_tuple = model_loader_func(*args, **kwargs)
    # Ensure we always have the model itself as the first item for offloading
    model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
    
    try:
        yield model_tuple
    finally:
        if model is not None:
            logger.info(f"On-Demand Mode: Offloading {model_name} to '{settings.OFFLOAD_DEVICE}'...")
            if hasattr(model, 'to'):
                model.to(settings.OFFLOAD_DEVICE)
            # Handle FasterWhisperPipeline case
            elif hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to(settings.OFFLOAD_DEVICE)

            del model
            del model_tuple
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"{model_name} offloaded and VRAM cache cleared.")

# --- Provider Abstraction ---
class AbstractModelProvider:
    def get_asr_model(self, model_size: str):
        raise NotImplementedError

    def get_align_model(self, language_code: str):
        raise NotImplementedError

    def get_diarize_model(self):
        raise NotImplementedError

class CachedModelProvider(AbstractModelProvider):
    """Provider that uses a permanent cache (default behavior)."""
    @contextmanager
    def get_asr_model(self, model_size: str):
        yield model_manager.get_whisper_model(model_size)

    @contextmanager
    def get_align_model(self, language_code: str):
        yield model_manager.get_alignment_model(language_code)

    @contextmanager
    def get_diarize_model(self):
        yield model_manager.get_diarization_model()

class OnDemandModelProvider(AbstractModelProvider):
    """Provider that loads and offloads models for each request."""
    def get_asr_model(self, model_size: str):
        loader_func = lambda: whisperx.load_model(
            model_size, device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE,
            download_root=settings.CACHE_DIR, vad_method=settings.VAD_METHOD,
            threads=settings.CPU_THREADS
        )
        return _load_on_demand(loader_func, model_key=f"ASR-{model_size}")

    def get_align_model(self, language_code: str):
        loader_func = lambda: whisperx.load_align_model(
            language_code=language_code, device=settings.DEVICE, model_dir=settings.CACHE_DIR
        )
        return _load_on_demand(loader_func, model_key=f"Align-{language_code}")

    def get_diarize_model(self):
        loader_func = lambda: whisperx.DiarizationPipeline(
            use_auth_token=settings.HF_TOKEN, device=settings.DEVICE
        )
        return _load_on_demand(loader_func, model_key="Diarize")


# --- Factory and Global Instance ---
def create_model_provider() -> AbstractModelProvider:
    if settings.ON_DEMAND_LOADING:
        logger.info("Initializing On-Demand Model Provider.")
        return OnDemandModelProvider()
    else:
        logger.info("Initializing Cached Model Provider.")
        # Pre-load the default ASR model at startup in cached mode
        model_manager.get_whisper_model(settings.ASR_MODEL_SIZE)
        return CachedModelProvider()

# Global singleton instance to be used by the API
model_provider = create_model_provider()