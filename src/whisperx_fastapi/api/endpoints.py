import os
import tempfile
from dataclasses import replace
from typing import Optional

import whisperx
from fastapi import (APIRouter, UploadFile, File, Form, HTTPException)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from whisperx_fastapi.api.models import (
    TranscriptionRequest, ForcedAlignmentRequest, TranscriptionResponse,
    ResponseFormat
)
from whisperx_fastapi.core.config import settings
from whisperx_fastapi.services.model_manager import model_manager
from whisperx_fastapi.services.srt_utils import generate_srt_from_segments, generate_srt_from_words
from whisperx.audio import load_audio

router = APIRouter()


# --- New Generation Endpoints ---

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe Audio and Align Timestamps"
)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    req_json: str = Form(alias="params", default=TranscriptionRequest().model_dump_json()),
):
    """
    Transcribes audio, performs alignment, and returns detailed segment and word-level
    timestamps. This endpoint offers key controls over the transcription and alignment process.
    """
    try:
        params = TranscriptionRequest.model_validate_json(req_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in 'params' field: {e}")

    audio_bytes = await file.read()

    def blocking_task():
        # Get the pre-loaded, shared pipeline from the model manager
        pipeline = model_manager.get_whisper_model(settings.ASR_MODEL_SIZE)
        
        # Store original settings to restore them after the request
        original_options = pipeline.options
        original_vad_params = pipeline._vad_params.copy()
        original_suppress_numerals = pipeline.suppress_numerals

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name

            # 1. Prepare and update pipeline options for this specific request
            temperatures = params.temperatures
            # The library expects a tuple or list, so wrap a single float
            if isinstance(temperatures, float):
                temperatures = [temperatures]

            request_asr_options = {
                "beam_size": params.beam_size,
                "best_of": params.beam_size, # Keep best_of same as beam_size
                "temperatures": tuple(temperatures),
                "initial_prompt": params.initial_prompt,
                "hotwords": params.hotwords
            }
            pipeline.options = replace(original_options, **request_asr_options)
            pipeline._vad_params["vad_onset"] = params.vad_onset
            pipeline._vad_params["vad_offset"] = params.vad_offset
            pipeline.suppress_numerals = params.suppress_numerals

            # 2. Transcribe with the request-specific settings
            result = pipeline.transcribe(
                temp_audio_path,
                batch_size=settings.WHISPER_BATCH_SIZE,
                language=params.language,
                task=params.task.value,
                chunk_size=params.chunk_size,
            )

            # 3. Align the transcribed result
            lang_code = result["language"]
            align_model, metadata = model_manager.get_alignment_model(lang_code)
            aligned_result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                temp_audio_path,
                device=settings.DEVICE,
                interpolate_method=params.interpolate_method,
                return_char_alignments=params.return_char_alignments,
            )
            aligned_result["language"] = lang_code
            return aligned_result
        finally:
            # 4. CRITICAL: Always restore the original pipeline settings for the next request
            pipeline.options = original_options
            pipeline._vad_params = original_vad_params
            pipeline.suppress_numerals = original_suppress_numerals
            # AND: Clean up the temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    result = await run_in_threadpool(blocking_task)
    return TranscriptionResponse(**result)


@router.post(
    "/align",
    response_model=TranscriptionResponse,
    summary="Forced Alignment of Text to Audio"
)
async def align_endpoint(
    file: UploadFile = File(...),
    req_json: str = Form(alias="params", default=ForcedAlignmentRequest(text_content="", language="en").model_dump_json()),
):
    """
    Performs forced alignment of a provided text transcript (ground truth) against
    an audio file to get precise word-level timestamps.
    """
    try:
        params = ForcedAlignmentRequest.model_validate_json(req_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in 'params' field: {e}")
        
    audio_bytes = await file.read()

    def blocking_task():
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name

            audio = load_audio(temp_audio_path)
            duration = audio.shape[0] / whisperx.audio.SAMPLE_RATE
            
            # Create a single segment from the user-provided text for alignment
            segments = [{"text": params.text_content, "start": 0.0, "end": duration}]

            # Get alignment model for the specified language
            align_model, metadata = model_manager.get_alignment_model(params.language)
            
            # Perform forced alignment
            aligned_result = whisperx.align(
                segments,
                align_model,
                metadata,
                temp_audio_path,
                device=settings.DEVICE,
                interpolate_method=params.interpolate_method,
                return_char_alignments=params.return_char_alignments,
            )
            aligned_result["language"] = params.language
            return aligned_result
        finally:
            # Clean up the temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    result = await run_in_threadpool(blocking_task)
    return TranscriptionResponse(**result)


# --- Legacy Endpoint ---

@router.post("/transcribe-audio", summary="Legacy Transcription Endpoint (Maintained for backward compatibility)")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    diarize: bool = Form(False, description="Enable speaker diarization."),
    word_timings: int = Form(0, ge=0, le=1, description="0 for segment-level SRT, 1 for word-level SRT."),
    translate: int = Form(0, ge=0, le=1, description="1 to translate to English, 0 otherwise."),
    response_format: ResponseFormat = Form(ResponseFormat.SRT, description="Desired output format ('srt' or 'json').")
):
    """
    A simplified endpoint for transcription, alignment, and optional diarization.
    This endpoint is maintained for backward compatibility. For more control,
    please use the `/transcribe` endpoint.
    """
    audio_bytes = await file.read()

    def blocking_pipeline():
        temp_audio_path = None
        try:
            # Write audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name
            
            # 1. Transcribe (using default server settings)
            task = "translate" if bool(translate) else "transcribe"
            asr_model = model_manager.get_whisper_model(settings.ASR_MODEL_SIZE)
            result = asr_model.transcribe(temp_audio_path, batch_size=settings.WHISPER_BATCH_SIZE, task=task)
            
            # 2. Align (required for both word timings and diarization)
            lang = result["language"]
            align_model, metadata = model_manager.get_alignment_model(lang)
            aligned_result = whisperx.align(result["segments"], align_model, metadata, temp_audio_path, device=settings.DEVICE)
            
            # 3. Diarize (if requested)
            if diarize:
                diarize_model = model_manager.get_diarization_model()
                diarize_segments = diarize_model(temp_audio_path)
                final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            else:
                final_result = aligned_result
            
            return final_result
        finally:
            # Clean up the temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    result = await run_in_threadpool(blocking_pipeline)

    if response_format == ResponseFormat.JSON:
        return JSONResponse(content=result)
    
    if response_format == ResponseFormat.SRT:
        if bool(word_timings):
            srt_content = generate_srt_from_words(result)
        else:
            srt_content = generate_srt_from_segments(result)
        
        return Response(
            content=srt_content,
            media_type="application/x-subrip",
            headers={"Content-Disposition": "attachment; filename=subtitles.srt"}
        )
    
    raise HTTPException(status_code=400, detail="Invalid response format specified.")


# --- Management Endpoint ---

@router.get("/health", summary="Check API Health", tags=["Management"])
async def health_check():
    """Checks if the service is running and the default ASR model is loaded."""
    try:
        # A more robust check might involve checking all required models
        is_ready = model_manager.is_model_loaded("asr", settings.ASR_MODEL_SIZE)
        status = "ready" if is_ready else "initializing"
        return {"status": status, "model_loaded": is_ready}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})