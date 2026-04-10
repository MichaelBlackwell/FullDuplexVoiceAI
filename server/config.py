import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080

    # Audio sample rates
    audio_sample_rate_webrtc: int = 48000
    audio_sample_rate_stt: int = 16000
    audio_sample_rate_tts: int = 24000

    # Audio format
    audio_channels: int = 1
    audio_ptime_ms: int = 20  # packetization time in ms

    # Backpressure
    buffer_max_size: int = 50  # max frames in queue before dropping

    log_level: str = "INFO"

    # VAD settings
    vad_threshold: float = 0.5
    vad_silence_duration_ms: int = 300
    vad_pre_speech_ms: int = 300
    vad_chunk_ms: int = 32  # Silero VAD requires 512 samples at 16kHz (32ms)

    # STT settings
    stt_model_size: str = "medium"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8"
    stt_language: str = "en"
    stt_max_utterance_seconds: float = 30.0
    stt_min_utterance_ms: int = 200

    # LLM settings
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "deepseek-chat")
    )
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    llm_context_max_messages: int = 40
    llm_context_keep_recent: int = 6

    # TTS settings
    tts_language: str = "a"  # Kokoro lang code: "a" = American English
    tts_voice: str = field(
        default_factory=lambda: os.environ.get("TTS_VOICE", "af_heart")
    )
    tts_device: str = field(
        default_factory=lambda: os.environ.get("TTS_DEVICE", "cpu")
    )

    # Barge-in settings
    barge_in_min_duration_ms: int = 300  # min speech duration before triggering interrupt
    barge_in_min_energy: float = 200.0  # min RMS (int16 scale) to confirm barge-in


settings = Settings()
