"""
VoiceShield — Audio Buffer
Sliding window buffer for continuous real-time audio chunk accumulation.
"""

import threading
import logging

logger = logging.getLogger("VoiceShieldAudioBuffer")


class AudioBuffer:
    """
    Thread-safe FIFO sliding window buffer for raw PCM16 audio.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        bytes_per_sample: int = 2,
        window_sec: float = 3.0,
        hop_sec: float = 1.0,
        max_buffer_sec: float = 10.0
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = bytes_per_sample
        self.bytes_per_second = sample_rate * channels * bytes_per_sample

        self.window_bytes = int(window_sec * self.bytes_per_second)
        self.hop_bytes = int(hop_sec * self.bytes_per_second)
        self.max_buffer_bytes = int(max_buffer_sec * self.bytes_per_second)

        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._total_ingested = 0

    def add_chunk(self, chunk: bytes):
        """Append raw PCM chunk to buffer."""
        if not chunk:
            return

        with self._lock:
            self._buffer.extend(chunk)
            self._total_ingested += len(chunk)

            # Cap maximum buffer to prevent unbounded memory growth
            if len(self._buffer) > self.max_buffer_bytes:
                overflow = len(self._buffer) - self.max_buffer_bytes
                del self._buffer[:overflow]

    def has_window(self) -> bool:
        """Check if buffer has accumulated at least one full inference window."""
        with self._lock:
            return len(self._buffer) >= self.window_bytes

    def get_window(self) -> bytes:
        """Get the current window bytes (without advancing)."""
        with self._lock:
            return bytes(self._buffer[:self.window_bytes])

    def advance_hop(self):
        """Advance the sliding window by hop size."""
        with self._lock:
            advance = min(self.hop_bytes, len(self._buffer))
            del self._buffer[:advance]

    def get_and_advance(self) -> bytes:
        """Get window and advance hop in a single atomic step."""
        with self._lock:
            window = bytes(self._buffer[:self.window_bytes])
            advance = min(self.hop_bytes, len(self._buffer))
            del self._buffer[:advance]
            return window

    def reset(self):
        """Clear all buffered audio."""
        with self._lock:
            self._buffer.clear()
            self._total_ingested = 0

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / self.bytes_per_second

    @property
    def total_ingested_bytes(self) -> int:
        with self._lock:
            return self._total_ingested
