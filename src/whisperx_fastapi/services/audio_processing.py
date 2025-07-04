import ffmpeg
import tempfile
from pathlib import Path
from loguru import logger

def process_audio(file_bytes: bytes, original_filename: str) -> str:
    """
    Saves audio bytes to a temporary file and converts it to 16kHz mono WAV format using ffmpeg.
    Returns the path to the processed temporary file.
    """
    suffix = Path(original_filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(file_bytes)
        input_path = tmp_input.name

    output_path = str(Path(tmp_input.name).with_suffix(".wav"))

    logger.info(f"Processing audio file: converting to 16kHz mono WAV.")
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ac=1, ar=16000, format="wav")
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        logger.info(f"Audio processing complete. Output at: {output_path}")
        # We don't delete the input file here because NamedTemporaryFile will handle it
        # upon garbage collection, but the output file needs to be managed by the caller.
        return output_path
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
        logger.error(f"FFmpeg error: {error_message}")
        raise RuntimeError(f"FFmpeg command failed: {error_message}")
    finally:
        # Clean up the initial temporary file
        Path(input_path).unlink(missing_ok=True)