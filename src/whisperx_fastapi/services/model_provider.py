import torch
import whisperx
from contextlib import contextmanager
from loguru import logger
from whisperx_fastapi.core.config import settings

class AbstractModelProvider:
    def get_asr_model(self, model_size: str):
        raise NotImplementedError

    def get_align_model(self, language_code: str):
        raise NotImplementedError

    def get_diarize_model(self):
        raise NotImplementedError

class CachedModelProvider(AbstractModelProvider):
    """
    Provider that loads models into VRAM and caches them for the lifetime of the application.
    This class now manages its own cache.
    """
    def __init__(self):
        """Initializes the cache dictionary."""
        self._loaded_models = {
            "asr": {},
            "align": {},
            "diarize": {}
        }
        logger.info("CachedModelProvider initialized.")

    def preload(self):
        """Pre-loads the default models into the cache at startup."""
        logger.info(f"Pre-loading default ASR model ({settings.ASR_MODEL_SIZE}) into cache...")
        # Use the context manager to populate the cache
        with self.get_asr_model(settings.ASR_MODEL_SIZE):
            pass
        logger.info("Default ASR model pre-loaded successfully.")

    @contextmanager
    def get_asr_model(self, model_size: str):
        """Loads and returns the ASR pipeline object from cache, loading if not present."""
        if model_size not in self._loaded_models["asr"]:
            logger.info(f"Loading Whisper ASR model to cache: {model_size}...")
            model = whisperx.load_model(
                model_size,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                download_root=settings.CACHE_DIR,
                vad_method=settings.VAD_METHOD,
                threads=settings.CPU_THREADS,
            )
            self._loaded_models["asr"][model_size] = model
            logger.info(f"Whisper ASR model {model_size} cached.")
        yield self._loaded_models["asr"][model_size]

    @contextmanager
    def get_align_model(self, language_code: str):
        """Loads and returns the alignment model from cache."""
        if language_code not in self._loaded_models["align"]:
            logger.info(f"Loading alignment model to cache for language: {language_code}...")
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=settings.DEVICE,
                model_dir=settings.CACHE_DIR
            )
            self._loaded_models["align"][language_code] = (model, metadata)
            logger.info(f"Alignment model for {language_code} cached.")
        yield self._loaded_models["align"][language_code]

    @contextmanager
    def get_diarize_model(self):
        """Loads and returns the diarization model from cache."""
        if "diarize_model" not in self._loaded_models["diarize"]:
            if not settings.HF_TOKEN:
                raise ValueError("Cannot load diarization model: HF_TOKEN is not set in the server environment.")
            
            logger.info("Loading diarization model to cache...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=settings.HF_TOKEN, device=settings.DEVICE)
            self._loaded_models["diarize"]["diarize_model"] = diarize_model
            logger.info("Diarization model cached.")
        yield self._loaded_models["diarize"]["diarize_model"]

class OnDemandModelProvider(AbstractModelProvider):
    """
    Provider that implements a hybrid on-demand strategy:
    - ASR Models (faster-whisper): Are loaded to GPU and destroyed after use due to library limitations.
    - Alignment/Diarization Models (PyTorch): Are loaded to CPU once, then moved to/from GPU for each request (true offloading).
    """
    def __init__(self):
        self._cpu_model_cache = {
            "align": {},
            "diarize": None
        }
        logger.info("OnDemandModelProvider initialized with hybrid strategy.")

    @contextmanager
    def get_asr_model(self, model_size: str):
        """
        Implements the 'Load-and-Destroy' pattern for the ASR model.
        This is necessary because the faster-whisper/ctranslate2 backend does not support
        moving the model between devices after it has been loaded.
        """
        model_key = f"ASR-{model_size}"
        logger.info(f"On-Demand: Loading {model_key} directly to '{settings.DEVICE}'...")
        asr_pipeline = whisperx.load_model(
            model_size, device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE,
            download_root=settings.CACHE_DIR, vad_method=settings.VAD_METHOD,
            threads=settings.CPU_THREADS
        )
        
        try:
            yield asr_pipeline
        finally:
            logger.info(f"On-Demand: Destroying {model_key} to free VRAM...")
            del asr_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"{model_key} destroyed and VRAM cache cleared.")

    @contextmanager
    def get_align_model(self, language_code: str):
        """
        Implements the 'True Offloading' pattern for alignment models.
        It loads the model to CPU RAM once, then moves it to the GPU for inference
        and back to the CPU afterward.
        """
        model_key = f"Align-{language_code}"
        
        # Step 1: Load model to CPU cache if not already present
        if language_code not in self._cpu_model_cache["align"]:
            logger.info(f"On-Demand: First-time load of {model_key} to '{settings.OFFLOAD_DEVICE}' cache.")
            model, metadata = whisperx.load_align_model(
                language_code=language_code, device=settings.OFFLOAD_DEVICE, model_dir=settings.CACHE_DIR
            )
            self._cpu_model_cache["align"][language_code] = (model, metadata)

        # Step 2: Move the cached model to GPU for the request
        model_tuple = self._cpu_model_cache["align"][language_code]
        model, metadata = model_tuple
        
        logger.info(f"On-Demand: Moving {model_key} from '{settings.OFFLOAD_DEVICE}' to '{settings.DEVICE}'.")
        model.to(settings.DEVICE)

        try:
            yield model, metadata
        finally:
            logger.info(f"On-Demand: Offloading {model_key} back to '{settings.OFFLOAD_DEVICE}'.")
            model.to(settings.OFFLOAD_DEVICE)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @contextmanager
    def get_diarize_model(self):
        """Implements 'True Offloading' for the diarization model."""
        if self._cpu_model_cache["diarize"] is None:
            logger.info(f"On-Demand: First-time load of Diarize model to '{settings.OFFLOAD_DEVICE}' cache.")
            self._cpu_model_cache["diarize"] = whisperx.DiarizationPipeline(
                use_auth_token=settings.HF_TOKEN, device=torch.device(settings.OFFLOAD_DEVICE)
            )

        diarize_model = self._cpu_model_cache["diarize"]
        logger.info(f"On-Demand: Moving Diarize model from '{settings.OFFLOAD_DEVICE}' to '{settings.DEVICE}'.")
        diarize_model.to(torch.device(settings.DEVICE))
        
        try:
            yield diarize_model
        finally:
            logger.info(f"On-Demand: Offloading Diarize model back to '{settings.OFFLOAD_DEVICE}'.")
            diarize_model.to(torch.device(settings.OFFLOAD_DEVICE))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Factory and Global Instance ---
def create_model_provider() -> AbstractModelProvider:
    if settings.ON_DEMAND_LOADING:
        logger.info("Initializing On-Demand Model Provider.")
        return OnDemandModelProvider()
    else:
        logger.info("Initializing Cached Model Provider.")
        # Instantiate the provider and call its preload method
        provider = CachedModelProvider()
        provider.preload()
        return provider

# Global singleton instance to be used by the API
model_provider = create_model_provider()