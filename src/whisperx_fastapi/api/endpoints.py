import os
import tempfile
import json
from dataclasses import replace
from typing import Dict, List, Union

import whisperx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from whisperx_fastapi.api.models import (
    TranscriptionRequest, TranscriptionResponse,
    ForcedAlignmentRequest, ForcedAlignmentResponse, SingleAlignmentResult,
    ASROptions, AlignmentOptions, SuccessfulJobResult, JobError,
    ResponseFormat # For legacy endpoint
)
from whisperx_fastapi.core.config import settings
from whisperx_fastapi.services.model_provider import model_provider, CachedModelProvider
from whisperx_fastapi.services.srt_utils import generate_srt_from_segments, generate_srt_from_words
from whisperx.audio import load_audio

router = APIRouter()

def _perform_transcription(audio_path: str, asr_options: ASROptions, pipeline) -> Dict:
    """Phase 1: Performs transcription on a single audio file."""
    original_options = pipeline.options
    original_vad_params = pipeline._vad_params.copy()
    original_suppress_numerals = pipeline.suppress_numerals
    try:
        temperatures = asr_options.temperatures
        if isinstance(temperatures, float): temperatures = [temperatures]
        request_asr_options = {
            "beam_size": asr_options.beam_size, "best_of": asr_options.beam_size,
            "temperatures": tuple(temperatures), "initial_prompt": asr_options.initial_prompt,
            "hotwords": asr_options.hotwords,
        }
        pipeline.options = replace(original_options, **request_asr_options)
        pipeline._vad_params["vad_onset"] = asr_options.vad_onset
        pipeline._vad_params["vad_offset"] = asr_options.vad_offset
        pipeline.suppress_numerals = asr_options.suppress_numerals
        result = pipeline.transcribe(
            audio_path, batch_size=settings.WHISPER_BATCH_SIZE,
            language=asr_options.language, task=asr_options.task.value,
            chunk_size=asr_options.chunk_size,
        )
        return result
    finally:
        pipeline.options = original_options
        pipeline._vad_params = original_vad_params
        pipeline.suppress_numerals = original_suppress_numerals

def _apply_alignment(transcription_result: Dict, audio_path: str, align_model, metadata, alignment_options: AlignmentOptions) -> Dict:
    """Phase 2: Applies forced alignment to a transcription result."""
    aligned_result = whisperx.align(
        transcription_result["segments"], align_model, metadata,
        audio_path, device=settings.DEVICE,
        interpolate_method=alignment_options.interpolate_method.value,
        return_char_alignments=alignment_options.return_char_alignments,
    )
    aligned_result["language_code"] = transcription_result["language"]
    return aligned_result

def _perform_forced_alignment(audio_path: str, text_content: str, language: str, alignment_options: AlignmentOptions, align_model, metadata) -> Dict:
    """Core logic for forced alignment."""
    audio = load_audio(audio_path)
    duration = audio.shape[0] / whisperx.audio.SAMPLE_RATE
    segments = [{"text": text_content, "start": 0.0, "end": duration}]
    aligned_result = whisperx.align(
        segments, align_model, metadata, audio_path, device=settings.DEVICE,
        interpolate_method=alignment_options.interpolate_method.value,
        return_char_alignments=alignment_options.return_char_alignments,
    )
    aligned_result["language_code"] = language
    return aligned_result

# --- Unified Hybrid Multipart/JSON API Endpoints ---

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe and Align Audio"
)
async def transcribe_endpoint(
    audio_files: List[UploadFile] = File(..., description="One or more audio files to process."),
    request_data: str = Form(..., description="A JSON string of the `TranscriptionRequest` model.")
):
    """
    Processes one or more audio files for transcription and alignment using a hybrid
    multipart/form-data request for high performance.

    - **`request_data` (form field):** Contains the JSON payload with job configurations.
    - **`audio_files` (file fields):** Contains the raw audio data. Each file's name must match a `filename` in the JSON payload.
    
    Example `curl` command:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/transcription" \\
    -F "audio_files=@/path/to/audio1.wav" \\
    -F "audio_files=@/path/to/audio2.wav" \\
    -F 'request_data={
        "jobs": [
            {"filename": "audio1.wav"},
            {"filename": "audio2.wav"}
        ],
        "asr_options": {"language": "en"}
    }'
    ```
    """
    try:
        request = TranscriptionRequest.model_validate_json(request_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in 'request_data' field: {e}")

    audio_map = {file.filename: file for file in audio_files}
    for job in request.jobs:
        if job.filename not in audio_map:
            raise HTTPException(status_code=404, detail=f"Audio file '{job.filename}' mentioned in JSON not found in uploaded files.")

    def blocking_task():
        # Change from list to dict
        results: Dict[str, Union[SuccessfulJobResult, JobError]] = {} 
        asr_options = request.asr_options
        alignment_options = request.alignment_options
        
        # This function processes a single file and handles its own errors
        def process_single_file(job, asr_pipeline, align_model=None, align_metadata=None):
            file = audio_map[job.filename]
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                    tmp.write(file.file.read())
                    tmp_path = tmp.name

                trans_result = _perform_transcription(tmp_path, asr_options, asr_pipeline)
                
                # If align model wasn't pre-loaded, load it now
                if align_model is None:
                    lang_code = trans_result["language"]
                    with model_provider.get_align_model(lang_code) as (am, md):
                        final_result = _apply_alignment(trans_result, tmp_path, am, md, alignment_options)
                else: # Align model was pre-loaded
                    final_result = _apply_alignment(trans_result, tmp_path, align_model, align_metadata, alignment_options)
                
                single_alignment = SingleAlignmentResult(**final_result)
                results[job.filename] = SuccessfulJobResult(data=single_alignment)

            except Exception as e:
                # If anything goes wrong with this file, record the error
                # and continue with the next one.
                results[job.filename] = JobError(message=f"Processing failed: {str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Main processing logic
        if asr_options.language:
            with model_provider.get_asr_model(settings.ASR_MODEL_SIZE) as p, \
                 model_provider.get_align_model(asr_options.language) as (am, md):
                for job in request.jobs:
                    process_single_file(job, p, am, md)
        else:
            with model_provider.get_asr_model(settings.ASR_MODEL_SIZE) as p:
                for job in request.jobs:
                    process_single_file(job, p) # Alignment model loaded inside function
                    
        return {"results": results}

    result = await run_in_threadpool(blocking_task)
    return TranscriptionResponse(**result)


@router.post(
    "/align",
    response_model=ForcedAlignmentResponse,
    summary="Forced Alignment of Text to Audio"
)
async def align_endpoint(
    audio_files: List[UploadFile] = File(..., description="One or more audio files to process."),
    request_data: str = Form(..., description="A JSON string of the `ForcedAlignmentRequest` model.")
):
    """
    Performs forced alignment for one or more audio files using a hybrid multipart request.
    """
    try:
        request = ForcedAlignmentRequest.model_validate_json(request_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in 'request_data' field: {e}")

    audio_map = {file.filename: file for file in audio_files}
    for job in request.jobs:
        if job.filename not in audio_map:
            raise HTTPException(status_code=404, detail=f"Audio file '{job.filename}' not found in uploaded files.")

    def blocking_task():
        results: Dict[str, Union[SuccessfulJobResult, JobError]] = {}
        with model_provider.get_align_model(request.language) as (align_model, metadata):
            for job in request.jobs:
                file = audio_map[job.filename]
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{job.filename}") as tmp:
                        tmp.write(file.file.read())
                        tmp_path = tmp.name
                    
                    aligned_result = _perform_forced_alignment(
                        tmp_path, job.text_content, request.language, request.alignment_options, align_model, metadata
                    )
                    single_alignment = SingleAlignmentResult(**aligned_result)
                    results[job.filename] = SuccessfulJobResult(data=single_alignment)

                except Exception as e:
                    results[job.filename] = JobError(message=f"Alignment failed: {str(e)}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        return {"results": results}

    result = await run_in_threadpool(blocking_task)
    return ForcedAlignmentResponse(**result)


# --- Legacy and Management Endpoints ---

@router.post(
    "/transcribe-audio",
    summary="Legacy Transcription Endpoint",
    description="This endpoint is maintained for backward compatibility. New integrations should use the unified `/transcription` endpoint.",
    tags=["Legacy"],
    include_in_schema=False,
)
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    diarize: bool = Form(False, description="Enable speaker diarization."),
    word_timings: int = Form(0, ge=0, le=1, description="0 for segment-level SRT, 1 for word-level SRT."),
    translate: int = Form(0, ge=0, le=1, description="1 to translate to English, 0 otherwise."),
    response_format: ResponseFormat = Form(ResponseFormat.SRT, description="Desired output format ('srt' or 'json').")
):
    audio_bytes = await file.read()
    def blocking_pipeline():
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name
            
            task = "translate" if bool(translate) else "transcribe"
            
            with model_provider.get_asr_model(settings.ASR_MODEL_SIZE) as asr_model:
                result = asr_model.transcribe(temp_audio_path, batch_size=settings.WHISPER_BATCH_SIZE, task=task)
                lang = result["language"]
                
                with model_provider.get_align_model(lang) as (align_model, metadata):
                    aligned_result = whisperx.align(result["segments"], align_model, metadata, temp_audio_path, device=settings.DEVICE)
                    
                    if diarize:
                        with model_provider.get_diarize_model() as diarize_model:
                            diarize_segments = diarize_model(temp_audio_path)
                            final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                    else:
                        final_result = aligned_result
            return final_result
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    result = await run_in_threadpool(blocking_pipeline)
    
    if response_format == ResponseFormat.JSON:
        return JSONResponse(content=result)
    elif response_format == ResponseFormat.SRT:
        srt_gen = generate_srt_from_words if bool(word_timings) else generate_srt_from_segments
        return Response(content=srt_gen(result), media_type="application/x-subrip",
                        headers={"Content-Disposition": "attachment; filename=subtitles.srt"})
    raise HTTPException(status_code=400, detail="Invalid response format specified.")


@router.get("/health", summary="Check API Health", tags=["Management"])
async def health_check():
    """Checks service status and model loading strategy."""
    status_info = {"status": "ready"}
    try:
        if isinstance(model_provider, CachedModelProvider):
            status_info["mode"] = "cached"
        else:
            status_info["mode"] = "on-demand"
        return JSONResponse(status_code=200, content=status_info)
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "mode": "unknown", "detail": str(e)})