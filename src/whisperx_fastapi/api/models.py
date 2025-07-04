from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"

class ResponseFormat(str, Enum):
    SRT = "srt"
    JSON = "json"

# --- Request Models ---

class TranscriptionRequest(BaseModel):
    """Parameters for a transcription request."""
    # Core Parameters
    language: Optional[str] = Field(None, description="ISO 639-1 language code. Auto-detects if None.")
    task: TaskType = Field(TaskType.TRANSCRIBE, description="Task to perform: 'transcribe' or 'translate'.")
    
    # VAD & Chunking Parameters
    chunk_size: int = Field(30, description="Size of audio chunks to process in seconds.")
    vad_onset: float = Field(0.5, description="VAD onset threshold for speech detection.")
    vad_offset: float = Field(0.363, description="VAD offset threshold for speech detection.")

    # ASR Decoding Options
    temperatures: Union[float, List[float]] = Field([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], description="Temperature for sampling. Can be a single float or a list for fallback.")
    beam_size: int = Field(5, description="Primary control for speed vs. accuracy. Higher values are more accurate but slower.")
    initial_prompt: Optional[str] = Field(None, description="Optional text to provide as context to the model.")
    suppress_numerals: bool = Field(False, description="If True, suppress numeric and symbol tokens. A high-level alternative to 'suppress_tokens'.")
    hotwords: Optional[str] = Field(None, description="Hotwords to provide the model.")
    
    # Alignment Options
    interpolate_method: str = Field("nearest", description="Method for interpolating timestamps: 'nearest', 'linear', 'pad'.")
    return_char_alignments: bool = Field(False, description="Whether to return character-level timestamps.")
    
    class Config:
        extra = 'forbid' # Forbid any extra parameters to prevent user confusion

class ForcedAlignmentRequest(BaseModel):
    """Parameters for a forced alignment request."""
    text_content: str = Field(..., min_length=1, description="The ground-truth text to align against the audio. Must not be empty.")
    language: str = Field(..., description="ISO 639-1 language code of the text.")
    
    # Alignment Options
    interpolate_method: str = Field("nearest", description="Method for interpolating timestamps: 'nearest', 'linear', 'pad'.")
    return_char_alignments: bool = Field(False, description="Whether to return character-level timestamps.")
    
    class Config:
        extra = 'forbid'

# --- Response Models ---

class WordSegment(BaseModel):
    word: str
    start: Optional[float]
    end: Optional[float]
    score: Optional[float]
    speaker: Optional[str] = None

class CharSegment(BaseModel):
    char: str
    start: Optional[float]
    end: Optional[float]
    score: Optional[float]

class AlignedSegment(BaseModel):
    text: str
    start: float
    end: float
    words: List[WordSegment]
    # This field will be populated only if return_char_alignments is True
    chars: Optional[List[CharSegment]] = None
    speaker: Optional[str] = None

class TranscriptionResponse(BaseModel):
    language: str
    segments: List[AlignedSegment]
    word_segments: List[WordSegment]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None