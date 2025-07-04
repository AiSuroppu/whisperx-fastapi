import io
from typing import Dict, Any

def _format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format HH:MM:SS,ms."""
    if seconds is None:
        return "00:00:00,000"
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds_val = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds_val:02},{ms:03}"

def generate_srt_from_segments(transcript_result: Dict[str, Any]) -> str:
    """Generates an SRT subtitle string from segment-level transcription."""
    srt_buffer = io.StringIO()
    for idx, segment in enumerate(transcript_result.get("segments", [])):
        start_time = _format_timestamp(segment["start"])
        end_time = _format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_buffer.write(f"{idx + 1}\n")
        srt_buffer.write(f"{start_time} --> {end_time}\n")
        srt_buffer.write(f"{text}\n\n")
    return srt_buffer.getvalue()

def generate_srt_from_words(transcript_result: Dict[str, Any]) -> str:
    """Generates an SRT subtitle string from word-level transcription."""
    srt_buffer = io.StringIO()
    word_idx = 1
    for segment in transcript_result.get("segments", []):
        for word_info in segment.get("words", []):
            start_time_val = word_info.get("start")
            end_time_val = word_info.get("end")
            text = word_info.get("word", "").strip()

            if start_time_val is None or end_time_val is None or not text:
                continue

            start_time = _format_timestamp(start_time_val)
            end_time = _format_timestamp(end_time_val)

            srt_buffer.write(f"{word_idx}\n")
            srt_buffer.write(f"{start_time} --> {end_time}\n")
            srt_buffer.write(f"{text}\n\n")
            word_idx += 1
    return srt_buffer.getvalue()