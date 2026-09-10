"""
VoiceShield — AI Detection Pipeline Orchestrator
Coordinates AudioBuffer, AudioPreprocessor, ModelManager (SileroVAD, AASIST,
RawNet3, ECAPA-TDNN, BispectralAnalyzer), and FusionEngine in a high-performance
concurrent real-time worker.
Smart India Hackathon 2026 | PS ID: 26104
"""

import time
import queue
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple

from .config import DEFAULT_CONFIG, PipelineConfig
from .models_manager import ModelManager
from .fusion import FusionEngine
from ..audio.buffer import AudioBuffer
from ..audio.preprocessing import AudioPreprocessor

logger = logging.getLogger("VoiceShieldPipeline")


class AIPipeline:
    """
    Asynchronous AI Voice-Cloning Detection Pipeline.
    Processes live audio windows off the WebRTC media path with ~200ms update cadence.
    Uses concurrent multi-threading for lightweight branches and async background worker for RawNet3.
    """

    def __init__(
        self,
        room_id: str,
        on_result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        source_sample_rate: int = 16000,
        channels: int = 1,
        window_sec: Optional[float] = None,
        hop_sec: Optional[float] = None,
        config: PipelineConfig = DEFAULT_CONFIG
    ):
        self.room_id = room_id
        self.on_result_callback = on_result_callback
        self.source_sample_rate = source_sample_rate
        self.channels = channels
        self.config = config

        eff_window_sec = window_sec if window_sec is not None else config.window_sec
        eff_hop_sec = hop_sec if hop_sec is not None else config.hop_sec

        # 1. Audio Buffer and Preprocessor
        self.buffer = AudioBuffer(
            sample_rate=source_sample_rate,
            channels=channels,
            window_sec=eff_window_sec,
            hop_sec=eff_hop_sec
        )
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000)

        # 2. Preloaded AI Model Branches (Singleton)
        logger.info(f"Retrieving AI model instances for room '{room_id}'...")
        models = ModelManager.get_instance(config)
        self.vad = models.vad
        self.aasist = models.aasist
        self.rawnet = models.rawnet
        self.ecapa = models.ecapa
        self.bispectral = models.bispectral

        # 3. Score Fusion Engine with Temporal Smoother
        self.fusion_engine = FusionEngine(config)

        # 4. Concurrency & Async State
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"VoiceShieldWorker-{room_id}")
        self.latest_rawnet_res: Dict[str, Any] = {"available": True, "rawnet_risk": 0.08}
        self.rawnet_busy = False
        self.rawnet_lock = threading.Lock()

        # 5. Async Ingestion Queue & Worker Loop
        self.chunk_queue = queue.Queue(maxsize=300)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.start()

    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"AIPipeline-{self.room_id}"
        )
        self.worker_thread.start()
        logger.info(
            f"AI Pipeline started for room '{self.room_id}' "
            f"(window={self.config.window_sec}s, hop={self.config.hop_sec}s)"
        )

    def ingest_pcm(self, pcm_bytes: bytes):
        """
        Non-blocking ingestion of raw PCM bytes from WebSocket audio tap.
        Drops oldest chunks if queue overflows to avoid blocking WebRTC.
        """
        if not self.running or not pcm_bytes:
            return

        try:
            self.chunk_queue.put_nowait(pcm_bytes)
        except queue.Full:
            try:
                self.chunk_queue.get_nowait()
                self.chunk_queue.put_nowait(pcm_bytes)
            except Exception:
                pass

    def _worker_loop(self):
        """Continuous background processing loop consuming audio chunks."""
        while self.running:
            try:
                chunk = self.chunk_queue.get(timeout=0.1)
                self.buffer.add_chunk(chunk)

                # Process all accumulated hops (usually 1 per iteration)
                while self.buffer.has_window() and self.running:
                    window_bytes = self.buffer.get_and_advance()
                    self._process_window(window_bytes)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Unexpected error in AI pipeline loop: {e}", exc_info=True)

    def _run_rawnet_async(self, waveform: np.ndarray):
        """Background inference for RawNet3 without stalling the 200ms loop."""
        try:
            t0 = time.perf_counter()
            res = self.rawnet.analyze(waveform)
            t_ms = round((time.perf_counter() - t0) * 1000, 2)
            res["duration_ms"] = t_ms
            with self.rawnet_lock:
                self.latest_rawnet_res = res
        except Exception as e:
            logger.error(f"RawNet3 background inference failed: {e}")
        finally:
            with self.rawnet_lock:
                self.rawnet_busy = False

    def _timed_aasist(self, waveform: np.ndarray) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        res = self.aasist.analyze(waveform)
        t_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res, t_ms

    def _timed_ecapa(self, waveform: np.ndarray) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        res = self.ecapa.verify_speaker(waveform, speaker_id="Friend")
        t_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res, t_ms

    def _timed_bispectral(self, waveform: np.ndarray) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        res = self.bispectral.analyze(waveform)
        t_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res, t_ms

    def _process_window(self, window_bytes: bytes):
        """
        Execute multi-branch AI analysis on the accumulated audio window.
        Runs fast branches concurrently to maintain ~200ms update cadence.
        """
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        try:
            # 1. Preprocess window to 16kHz float32
            waveform_16k = self.preprocessor.process(
                window_bytes,
                source_sr=self.source_sample_rate,
                channels=self.channels,
                normalize=True
            )

            # 2. Silero VAD Speech Presence Gating
            t_vad_0 = time.perf_counter()
            vad_res = self.vad.detect_speech(waveform_16k, sample_rate=16000)
            timings["vad_ms"] = round((time.perf_counter() - t_vad_0) * 1000, 2)

            # If no speech is detected, skip neural inference and decay smoothly
            if not vad_res["is_speech"]:
                total_ms = int((time.perf_counter() - t_start) * 1000)
                result = self.fusion_engine.fuse(
                    vad_result=vad_res,
                    rawnet_result={"available": False, "rawnet_risk": None},
                    aasist_result={"available": False, "aasist_risk": None},
                    ecapa_result={"available": False, "speaker_risk": None},
                    bispectral_result={"available": False, "bispectral_risk": None},
                    hop_duration_sec=self.config.hop_sec,
                    timings=timings,
                    total_pipeline_ms=total_ms
                )
                result["room_id"] = self.room_id
                self._dispatch_result(result)
                return

            # 3. Schedule RawNet3 in background if not already busy
            with self.rawnet_lock:
                if not self.rawnet_busy:
                    self.rawnet_busy = True
                    self.executor.submit(self._run_rawnet_async, waveform_16k.copy())

            # 4. Execute Fast Branches Concurrently in ThreadPool
            fut_aasist = self.executor.submit(self._timed_aasist, waveform_16k)
            fut_ecapa = self.executor.submit(self._timed_ecapa, waveform_16k)
            fut_bispectral = self.executor.submit(self._timed_bispectral, waveform_16k)

            aasist_res, timings["aasist_ms"] = fut_aasist.result()
            ecapa_res, timings["ecapa_ms"] = fut_ecapa.result()
            bispectral_res, timings["bispectral_ms"] = fut_bispectral.result()

            with self.rawnet_lock:
                rawnet_res = dict(self.latest_rawnet_res)
            timings["rawnet_ms"] = rawnet_res.get("duration_ms", 0.0)

            # 5. Score Fusion with Temporal Smoothing
            t_fuse_0 = time.perf_counter()
            total_ms = int((time.perf_counter() - t_start) * 1000)

            result = self.fusion_engine.fuse(
                vad_result=vad_res,
                rawnet_result=rawnet_res,
                aasist_result=aasist_res,
                ecapa_result=ecapa_res,
                bispectral_result=bispectral_res,
                hop_duration_sec=self.config.hop_sec,
                timings=timings,
                total_pipeline_ms=total_ms
            )

            timings["fusion_ms"] = round((time.perf_counter() - t_fuse_0) * 1000, 2)
            result["room_id"] = self.room_id

            logger.info(
                f"[AI Verdict] Room: {self.room_id} | Risk: {result['risk_score']} | "
                f"Conf: {result['confidence']} | Status: {result['classification']} | "
                f"AASIST: {result['aasist_risk']} | RawNet3: {result['rawnet_risk']} | "
                f"Latency: {result['latency_ms']}ms"
            )

            self._dispatch_result(result)

        except Exception as e:
            logger.error(f"Error processing audio window: {e}", exc_info=True)

    def _dispatch_result(self, result: Dict[str, Any]):
        """Forward structured AI result to registered callback."""
        if self.on_result_callback:
            try:
                self.on_result_callback(result)
            except Exception as e:
                logger.error(f"Error dispatching AI result: {e}", exc_info=True)

    def reset(self):
        """Clear the audio buffer, pending chunks, and temporal smoother state."""
        self.buffer.reset()
        self.fusion_engine.reset()
        with self.rawnet_lock:
            self.latest_rawnet_res = {"available": True, "rawnet_risk": 0.08}
            self.rawnet_busy = False

        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except Exception:
                break
        logger.info(f"AI Pipeline reset for room '{self.room_id}'")

    def stop(self):
        """Stop background worker and release resources."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
        self.buffer.reset()
        self.fusion_engine.reset()
        logger.info(f"AI Pipeline stopped for room '{self.room_id}'")

