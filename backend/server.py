"""
VoiceShield Backend — Real-Time WebRTC Signaling & AI Audio Pipeline Foundation
Smart India Hackathon 2026 | PS ID: 26104
"""

import json
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VoiceShieldServer")

app = FastAPI(
    title="VoiceShield Backend",
    description="Real-time WebRTC signaling relay and AI voice cloning detection ingestion server",
    version="1.0"
)

# Active connected rooms
# room_id -> list of signaling WebSockets
signal_rooms: Dict[str, List[WebSocket]] = {}

# room_id -> list of audio ingestion WebSockets
audio_rooms: Dict[str, List[WebSocket]] = {}


# ==============================================================================
# AI VOICE CLONING DETECTION PIPELINE INTERFACES & PLACEHOLDERS
# ==============================================================================

class VoiceActivityDetector:
    """
    Interface / placeholder for Silero VAD.
    Responsible for identifying speech segments in the incoming audio.
    """
    def __init__(self):
        self.is_ready = False
        logger.info("[AI Pipeline] VoiceActivityDetector interface initialized (Silero VAD placeholder)")

    def process_frame(self, pcm_chunk: bytes, sample_rate: int = 16000) -> Optional[dict]:
        """
        Future integration point: Run Silero VAD on 16kHz PCM audio chunk.
        DO NOT generate fake speech detection results.
        """
        return None


class SpoofDetector:
    """
    Interface / placeholder for RawNet3.
    Responsible for analyzing raw waveform characteristics for synthetic/cloned voice artifacts.
    """
    def __init__(self):
        self.is_ready = False
        logger.info("[AI Pipeline] SpoofDetector interface initialized (RawNet3 placeholder)")

    def analyze(self, audio_segment: bytes, sample_rate: int = 16000) -> Optional[dict]:
        """
        Future integration point: Pass raw audio waveform into RawNet3 model.
        DO NOT generate fake spoof scores.
        """
        return None


class SpeakerVerifier:
    """
    Interface / placeholder for ECAPA-TDNN.
    Responsible for extracting speaker embeddings and performing verification against enrollment audio.
    """
    def __init__(self):
        self.is_ready = False
        logger.info("[AI Pipeline] SpeakerVerifier interface initialized (ECAPA-TDNN placeholder)")

    def verify(self, audio_segment: bytes, reference_embedding: Optional[bytes] = None) -> Optional[dict]:
        """
        Future integration point: Extract embeddings and compare similarity.
        DO NOT generate fake similarity scores.
        """
        return None


class RiskEngine:
    """
    Interface / placeholder for Multi-Model Risk Fusion.
    Combines VAD speech segments, RawNet3 spoof probability, and ECAPA-TDNN speaker similarity
    into an unified risk score and alert trigger.
    """
    def __init__(self):
        self.is_ready = False
        logger.info("[AI Pipeline] RiskEngine interface initialized (Risk Fusion placeholder)")

    def evaluate_risk(
        self,
        vad_result: Optional[dict] = None,
        spoof_result: Optional[dict] = None,
        speaker_result: Optional[dict] = None
    ) -> Optional[dict]:
        """
        Future fusion point:
        Combines model confidence and probabilities.
        """
        return None


class AudioProcessor:
    """
    Audio Ingestion & Buffer Manager.
    Prepares standard PCM16 Mono 16kHz streams for downstream AI models.
    """
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.buffer = bytearray()
        self.vad = VoiceActivityDetector()
        self.spoof_detector = SpoofDetector()
        self.speaker_verifier = SpeakerVerifier()
        self.risk_engine = RiskEngine()
        self.total_bytes_received = 0

    def ingest_chunk(self, raw_pcm: bytes):
        """
        Receives raw PCM frames from WebRTC audio tap.
        Buffers and prepares data for future 16kHz normalization and model inference.
        """
        self.total_bytes_received += len(raw_pcm)
        self.buffer.extend(raw_pcm)

        # Retain last 3 seconds of audio (at 16kHz 16-bit mono = ~96000 bytes)
        max_buffer = 16000 * 2 * 3
        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]


# Room audio processors: room_id -> AudioProcessor
room_processors: Dict[str, AudioProcessor] = {}


# ==============================================================================
# HTTP ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {
        "service": "VoiceShield Backend",
        "status": "online",
        "supported_features": [
            "webrtc_signaling",
            "audio_ingestion",
            "ai_pipeline_ready"
        ],
        "version": "1.0"
    }


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "signal_rooms_count": len(signal_rooms),
            "audio_rooms_count": len(audio_rooms)
        }
    )


async def remove_client(
    rooms: Dict[str, List[WebSocket]],
    room_id: str,
    websocket: WebSocket
):
    clients = rooms.get(room_id)
    if clients is None:
        return

    if websocket in clients:
        clients.remove(websocket)

    if not clients:
        rooms.pop(room_id, None)
        if room_id in room_processors and room_id not in audio_rooms:
            room_processors.pop(room_id, None)


# ==============================================================================
# WEBSOCKET SIGNALING ENDPOINT
# ==============================================================================

@app.websocket("/ws/{room_id}")
async def signaling_endpoint(
    websocket: WebSocket,
    room_id: str
):
    """
    Signaling relay for WebRTC call orchestration between Phone A and Phone B.
    Relays: call, call_accepted, call_rejected, offer, answer, candidate, hangup.
    """
    await websocket.accept()

    clients = signal_rooms.setdefault(room_id, [])
    clients.append(websocket)

    role = "caller" if len(clients) == 1 else "receiver"
    logger.info(f"Signaling client joined room '{room_id}' as {role}. Total in room: {len(clients)}")

    try:
        # Notify joining client of its assigned role in the room
        await websocket.send_json({
            "type": "room_ready",
            "role": role,
            "room": room_id
        })

        while True:
            message_text = await websocket.receive_text()
            logger.info(f"Signal in room '{room_id}': {message_text[:120]}")

            try:
                msg_json = json.loads(message_text)
                msg_type = msg_json.get("type", "unknown")
            except Exception:
                msg_type = "raw"

            # Broadcast message to all other participants in the room
            recipients = signal_rooms.get(room_id, [])
            disconnected = []

            for client in recipients:
                if client is websocket:
                    continue

                try:
                    await client.send_text(message_text)
                except Exception as exc:
                    logger.warning(f"Error forwarding signaling message: {exc}")
                    disconnected.append(client)

            for client in disconnected:
                if client in recipients:
                    recipients.remove(client)

    except WebSocketDisconnect:
        logger.info(f"Signaling client disconnected from room '{room_id}'")
    except Exception as exc:
        logger.error(f"Signaling error in room '{room_id}': {exc}", exc_info=True)
    finally:
        await remove_client(signal_rooms, room_id, websocket)
        remaining = len(signal_rooms.get(room_id, []))
        logger.info(f"Signaling clients remaining in room '{room_id}': {remaining}")


# ==============================================================================
# WEBSOCKET AUDIO INGESTION ENDPOINT
# ==============================================================================

@app.websocket("/ws/audio/{room_id}")
async def audio_ingestion_endpoint(
    websocket: WebSocket,
    room_id: str
):
    """
    Live audio ingestion endpoint for the future AI detection pipeline.
    Receives raw PCM frames tapped from WebRTC call.
    """
    await websocket.accept()

    clients = audio_rooms.setdefault(room_id, [])
    clients.append(websocket)

    if room_id not in room_processors:
        room_processors[room_id] = AudioProcessor(room_id)

    processor = room_processors[room_id]
    logger.info(f"Audio ingestion client connected to room '{room_id}'")

    try:
        while True:
            # Receive binary raw PCM frames from the phone
            pcm_bytes = await websocket.receive_bytes()
            processor.ingest_chunk(pcm_bytes)

    except WebSocketDisconnect:
        logger.info(
            f"Audio ingestion client disconnected from room '{room_id}'. "
            f"Total bytes ingested: {processor.total_bytes_received}"
        )
    except Exception as exc:
        logger.error(f"Audio ingestion error in room '{room_id}': {exc}", exc_info=True)
    finally:
        await remove_client(audio_rooms, room_id, websocket)
        remaining = len(audio_rooms.get(room_id, []))
        logger.info(f"Audio clients remaining in room '{room_id}': {remaining}")