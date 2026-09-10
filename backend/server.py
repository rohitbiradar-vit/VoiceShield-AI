"""
VoiceShield Backend — Real-Time WebRTC Signaling & Live AI Voice Cloning Detection
Smart India Hackathon 2026 | PS ID: 26104
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Set OpenBLAS to 1 thread on Windows to avoid memory allocation contention
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from backend.ai.pipeline import AIPipeline
from backend.ai.models_manager import ModelManager
from backend.ai.config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VoiceShieldServer")

app = FastAPI(
    title="VoiceShield Backend",
    description="Real-time WebRTC signaling relay and live role-isolated AI voice cloning detection ingestion server",
    version="2.1"
)

# Active connected rooms
# room_id -> list of signaling WebSockets
signal_rooms: Dict[str, List[WebSocket]] = {}

# room_id -> {"caller": WebSocket, "receiver": WebSocket}
signal_roles: Dict[str, Dict[str, WebSocket]] = {}

# audio_key ("{room_id}_{role}") -> list of audio WebSockets
audio_rooms: Dict[str, List[WebSocket]] = {}

# audio_key ("{room_id}_{role}") -> AIPipeline instance
room_pipelines: Dict[str, AIPipeline] = {}

# Reference to the main asyncio event loop for thread-safe broadcasting
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def on_startup():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("VoiceShield server starting up. Pre-loading AI models into memory...")
    # Pre-load singleton models so first call has 0ms loading latency
    await asyncio.to_thread(ModelManager.get_instance)
    logger.info("VoiceShield server startup complete. Models resident and event loop captured.")


# ==============================================================================
# HTTP ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {
        "service": "VoiceShield Backend",
        "status": "online",
        "architecture": "two_way_peer_defense",
        "supported_features": [
            "webrtc_signaling",
            "role_isolated_audio_ingestion",
            "directional_evaluation",
            "silero_vad",
            "aasist_spectral_graph_spoof_detection",
            "rawnet3_anti_spoofing",
            "ecapa_tdnn_speaker_verification",
            "bispectral_higher_order_spectra",
            "temporal_smoothing_rate_limiter",
            "weighted_risk_fusion"
        ],
        "version": "2.1"
    }


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "signal_rooms_count": len(signal_rooms),
            "signal_roles": {r: list(roles.keys()) for r, roles in signal_roles.items()},
            "audio_rooms_count": len(audio_rooms),
            "active_pipelines": list(room_pipelines.keys())
        }
    )


# ==============================================================================
# AI RESULT DISPATCH (ROLE-ISOLATED)
# ==============================================================================

def send_ai_result_to_role_sync(room_id: str, target_role: str, result: dict):
    """
    Called from AIPipeline worker thread.
    Dispatches async result delivery to the main event loop, targeted strictly
    to the phone identified by target_role.
    """
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            _async_send_ai_result_to_role(room_id, target_role, result),
            main_event_loop
        )


async def _async_send_ai_result_to_role(room_id: str, target_role: str, result: dict):
    """
    Send the calculated AI verdict strictly to the target phone's signaling WebSocket.
    Mobile A (caller) receives the analysis of Mobile B's voice.
    Mobile B (receiver) receives the analysis of Mobile A's voice.
    """
    roles = signal_roles.get(room_id, {})
    target_ws = roles.get(target_role)

    evaluated_peer = "receiver" if target_role == "caller" else "caller"
    payload = {
        **result,
        "target_role": target_role,
        "evaluated_peer": evaluated_peer
    }
    message_text = json.dumps(payload)

    if target_ws:
        try:
            await target_ws.send_text(message_text)
            return
        except Exception as exc:
            logger.warning(f"Error sending AI result to {target_role} in {room_id}: {exc}")

    # Fallback: if role mapping wasn't found (e.g. single testing client or reconnect),
    # and exactly 1 client is in the room, send to that client
    clients = signal_rooms.get(room_id, [])
    if len(clients) == 1:
        try:
            await clients[0].send_text(message_text)
        except Exception as exc:
            logger.warning(f"Error sending AI result to fallback client in {room_id}: {exc}")


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
    Maintains role assignment (caller vs receiver) for directional AI result routing.
    """
    await websocket.accept()

    clients = signal_rooms.setdefault(room_id, [])
    clients.append(websocket)
    roles_map = signal_roles.setdefault(room_id, {})

    # Assign caller if not taken, else receiver, else spectator
    if "caller" not in roles_map:
        assigned_role = "caller"
    elif "receiver" not in roles_map:
        assigned_role = "receiver"
    else:
        assigned_role = f"spectator_{len(clients)}"

    roles_map[assigned_role] = websocket
    logger.info(f"Signaling client joined room '{room_id}' as '{assigned_role}'. Total clients: {len(clients)}")

    try:
        # Notify joining client of its assigned role in the room
        await websocket.send_json({
            "type": "room_ready",
            "role": assigned_role,
            "room": room_id
        })

        while True:
            message_text = await websocket.receive_text()
            logger.info(f"Signal from {assigned_role} in room '{room_id}': {message_text[:120]}")

            # If message is a hangup or reject, reset pipelines for this room
            try:
                msg_json = json.loads(message_text)
                if msg_json.get("type") in ("hangup", "call_rejected"):
                    for role_name in ("caller", "receiver"):
                        pipe_key = f"{room_id}_{role_name}"
                        if pipe_key in room_pipelines:
                            room_pipelines[pipe_key].reset()
            except Exception:
                pass

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
        logger.info(f"Signaling client ({assigned_role}) disconnected from room '{room_id}'")
    except Exception as exc:
        logger.error(f"Signaling error for {assigned_role} in room '{room_id}': {exc}", exc_info=True)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        if not clients:
            signal_rooms.pop(room_id, None)

        if room_id in signal_roles:
            signal_roles[room_id].pop(assigned_role, None)
            if not signal_roles[room_id]:
                signal_roles.pop(room_id, None)

        # Clean up pipeline for this role if no audio client exists
        pipe_key = f"{room_id}_{assigned_role}"
        if pipe_key not in audio_rooms:
            p = room_pipelines.pop(pipe_key, None)
            if p:
                p.stop()
                logger.info(f"Cleaned up and stopped AI pipeline for '{pipe_key}'")

        remaining = len(signal_rooms.get(room_id, []))
        logger.info(f"Signaling clients remaining in room '{room_id}': {remaining}")


# ==============================================================================
# WEBSOCKET AUDIO INGESTION ENDPOINTS (ROLE-ISOLATED)
# ==============================================================================

async def handle_audio_stream(
    websocket: WebSocket,
    room_id: str,
    role: str,
    sample_rate: int,
    channels: int
):
    """
    Common handler for audio ingestion streams.
    Spawns/manages an independent AI pipeline keyed by '{room_id}_{role}'.
    Evaluates incoming peer audio and sends results strictly back to that phone.
    """
    await websocket.accept()

    audio_key = f"{room_id}_{role}"
    clients = audio_rooms.setdefault(audio_key, [])
    clients.append(websocket)

    # Initialize live AI pipeline for this specific role in this room if not already running
    if audio_key not in room_pipelines:
        logger.info(
            f"Starting real-time AI detection pipeline for '{audio_key}' "
            f"(sample_rate={sample_rate}, channels={channels}, target_role={role})..."
        )
        pipeline = AIPipeline(
            room_id=audio_key,
            on_result_callback=lambda res: send_ai_result_to_role_sync(room_id, role, res),
            source_sample_rate=sample_rate,
            channels=channels,
            window_sec=DEFAULT_CONFIG.window_sec,
            hop_sec=DEFAULT_CONFIG.hop_sec,
            config=DEFAULT_CONFIG
        )
        room_pipelines[audio_key] = pipeline

    pipeline = room_pipelines[audio_key]
    logger.info(f"Audio ingestion client connected: key='{audio_key}' (sr={sample_rate}, ch={channels})")

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            pipeline.ingest_pcm(pcm_bytes)

    except WebSocketDisconnect:
        logger.info(f"Audio ingestion client disconnected from '{audio_key}'")
    except Exception as exc:
        logger.error(f"Audio ingestion error for '{audio_key}': {exc}", exc_info=True)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        if not clients:
            audio_rooms.pop(audio_key, None)
            # If signaling is also disconnected for this role, stop the pipeline
            if not signal_roles.get(room_id, {}).get(role):
                p = room_pipelines.pop(audio_key, None)
                if p:
                    p.stop()
                    logger.info(f"Cleaned up and stopped AI pipeline for '{audio_key}'")

        remaining = len(audio_rooms.get(audio_key, []))
        logger.info(f"Audio clients remaining for '{audio_key}': {remaining}")


@app.websocket("/ws/audio/{room_id}/{role}")
async def audio_ingestion_role_endpoint(
    websocket: WebSocket,
    room_id: str,
    role: str,
    sample_rate: int = 16000,
    channels: int = 1
):
    """
    Live audio ingestion endpoint with path-based role:
    /ws/audio/{room_id}/{role}?sample_rate=16000&channels=1
    """
    await handle_audio_stream(websocket, room_id, role, sample_rate, channels)


@app.websocket("/ws/audio/{room_id}")
async def audio_ingestion_endpoint(
    websocket: WebSocket,
    room_id: str,
    role: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1
):
    """
    Live audio ingestion endpoint with optional query param:
    /ws/audio/{room_id}?role=caller&sample_rate=16000&channels=1
    """
    chosen_role = role if role else "caller"
    await handle_audio_stream(websocket, room_id, chosen_role, sample_rate, channels)