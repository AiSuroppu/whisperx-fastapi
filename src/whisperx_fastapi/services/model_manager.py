import whisperx
import torch
from loguru import logger
from whisperx_fastapi.core.config import settings

class ModelManager:
    _instance = None
    _loaded_models = {
        "asr": {},
        "align": {},
        "diarize": {}
    }

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, device: str = settings.DEVICE, compute_type: str = settings.COMPUTE_TYPE):
        self.device = device
        self.compute_type = compute_type

    def get_whisper_model(self, model_size: str):
        """Loads and returns the ASR pipeline object."""
        if model_size not in self._loaded_models["asr"]:
            logger.info(f"Loading Whisper ASR model: {model_size}...")
            model = whisperx.load_model(
                model_size,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                download_root=settings.CACHE_DIR,
                vad_method=settings.VAD_METHOD,
                threads=settings.CPU_THREADS,
            )
            self._loaded_models["asr"][model_size] = model
            logger.info(f"Whisper ASR model {model_size} loaded.")
        return self._loaded_models["asr"][model_size]

    def get_alignment_model(self, language: str):
        cache_key = f"align_{language}"
        if cache_key not in self._loaded_models["align"]:
            logger.info(f"Loading alignment model for language: {language}...")
            model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_dir=settings.CACHE_DIR
            )
            self._loaded_models["align"][cache_key] = (model, metadata)
            logger.info(f"Alignment model for {language} loaded.")
        return self._loaded_models["align"][cache_key]

    def get_diarization_model(self):
        if "diarize_model" not in self._loaded_models["diarize"]:
            if not settings.HF_TOKEN:
                raise ValueError("Cannot load diarization model: HF_TOKEN is not set in the server environment.")
            
            logger.info("Loading diarization model...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=settings.HF_TOKEN, device=self.device)
            self._loaded_models["diarize"]["diarize_model"] = diarize_model
            logger.info("Diarization model loaded.")
        return self._loaded_models["diarize"]["diarize_model"]

# Global singleton instance
model_manager = ModelManager()