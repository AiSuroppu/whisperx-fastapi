from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Service settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Model settings
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16" # e.g., "int8", "float16", "float32"
    ASR_MODEL_SIZE: str = "large-v3"
    CACHE_DIR: Optional[str] = None
    CPU_THREADS: int = 4 # Number of threads for CPU execution
    WHISPER_BATCH_SIZE: int = 1 # Batch size for Whisper model

    # VAD Configuration
    VAD_METHOD: Literal["silero", "pyannote"] = "pyannote"

    # Diarization settings
    HF_TOKEN: Optional[str] = None # Server-side HuggingFace token

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()