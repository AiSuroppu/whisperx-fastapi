from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field, conlist
from enum import Enum

# --- Enums ---

class TaskType(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"

class InterpolateMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    PAD = "pad"

class JobStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class ResponseFormat(str, Enum):
    SRT = "srt"
    JSON = "json"

# --- Core Configuration Objects ---

class ASROptions(BaseModel):
    """Fine-tuning options for the ASR (Whisper) model."""
    task: TaskType = Field(TaskType.TRANSCRIBE, description="Task to perform: 'transcribe' or 'translate'.")
    language: Optional[str] = Field(None, description="ISO 639-1 language code. Auto-detects if None, but specifying is recommended for batch jobs.")
    # VAD & Chunking Parameters
    chunk_size: int = Field(30, description="Size of audio chunks to process in seconds.")
    vad_onset: float = Field(0.5, description="VAD onset threshold for speech detection.")
    vad_offset: float = Field(0.363, description="VAD offset threshold for speech detection.")
    # ASR Decoding Options
    temperatures: Union[float, List[float]] = Field([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], description="Temperature for sampling. Can be a single float or a list for fallback.")
    beam_size: int = Field(5, description="Number of beams in beam search. Higher values are more accurate but slower.")
    initial_prompt: Optional[str] = Field(None, description="Optional text to provide as context to the model.")
    suppress_numerals: bool = Field(False, description="If True, suppress numeric and symbol tokens.")
    hotwords: Optional[str] = Field(None, description="Hotwords to provide the model.")

class AlignmentOptions(BaseModel):
    """Options for the alignment process."""
    interpolate_method: InterpolateMethod = Field(InterpolateMethod.NEAREST, description="Method for interpolating timestamps.")
    return_char_alignments: bool = Field(False, description="Whether to return character-level timestamps.")


# --- Request Models ---

class TranscriptionJob(BaseModel):
    """A single transcription job, containing the audio source."""
    filename: str = Field(..., description="The exact filename of the audio file uploaded in the multipart request.")
    # Per-job overrides could be added here in the future if needed
    # language: Optional[str] = None 

class TranscriptionRequest(BaseModel):
    """
    A request to transcribe one or more audio files.
    The endpoint is unified: send a list with one item for a single job, or multiple for a batch.
    """
    jobs: conlist(TranscriptionJob, min_length=1)
    asr_options: ASROptions = Field(default_factory=ASROptions)
    alignment_options: AlignmentOptions = Field(default_factory=AlignmentOptions)

class AlignmentJob(BaseModel):
    """A single forced alignment job, containing audio and its ground-truth text."""
    filename: str = Field(..., description="The exact filename of the audio file uploaded in the multipart request.")
    text_content: str = Field(..., min_length=1, description="The ground-truth text to align against the audio.")

class ForcedAlignmentRequest(BaseModel):
    """
    A request to perform forced alignment on one or more audio files.
    """
    jobs: conlist(AlignmentJob, min_length=1)
    language: str = Field(..., description="ISO 639-1 language code of the text. Must be specified for all jobs.")
    alignment_options: AlignmentOptions = Field(default_factory=AlignmentOptions)


# --- Response Models ---

class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None

class CharSegment(BaseModel):
    char: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None

class AlignedSegment(BaseModel):
    text: str
    start: float
    end: float
    words: List[WordSegment]
    # This field will be populated only if return_char_alignments is True
    chars: Optional[List[CharSegment]] = None
    speaker: Optional[str] = None

class SingleAlignmentResult(BaseModel):
    """The result of processing a single alignment job."""
    language_code: str
    segments: List[AlignedSegment]
    word_segments: List[WordSegment]

class JobError(BaseModel):
    """Model for a failed job."""
    status: JobStatus = Field(JobStatus.ERROR, description="Indicates that the job failed.")
    message: str = Field(..., description="A description of the error that occurred.")

class SuccessfulJobResult(BaseModel):
    """Wrapper for a successful job result to include status."""
    status: JobStatus = Field(JobStatus.SUCCESS, description="Indicates the job was successful.")
    data: SingleAlignmentResult

# A job can either succeed or fail.
SingleJobResponse = Union[SuccessfulJobResult, JobError]

class TranscriptionResponse(BaseModel):
    """The final response for a transcription request, containing results for all jobs keyed by filename."""
    results: Dict[str, SingleJobResponse]

class ForcedAlignmentResponse(BaseModel):
    """The final response for a forced alignment request, keyed by filename."""
    results: Dict[str, SingleJobResponse]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None