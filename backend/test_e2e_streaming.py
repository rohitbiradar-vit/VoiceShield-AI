import os
import sys
import time
import json
import asyncio
import threading
import numpy as np
import websockets
import uvicorn

sys.path.insert(0, ".")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from backend.server import app

def run_server():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    srv = uvicorn.Server(cfg)
    srv.run()

async def test_client():
    await asyncio.sleep(2.0)
    print("[E2E] Connecting to signaling ws://127.0.0.1:8765/ws/test_room")
    async with websockets.connect("ws://127.0.0.1:8765/ws/test_room") as sig_ws:
        init_msg = await sig_ws.recv()
        print(f"[E2E] Signaling init received: {init_msg}")

        url = "ws://127.0.0.1:8765/ws/audio/test_room?sample_rate=16000&channels=1"
        print(f"[E2E] Connecting to audio ingestion: {url}")
        async with websockets.connect(url) as audio_ws:
            print("[E2E] Streaming 3.5s of live PCM audio frames...")
            t = np.linspace(0, 3.5, int(3.5 * 16000), endpoint=False)
            f0 = 150.0 + 20.0 * np.sin(2 * np.pi * 3.0 * t)
            phase = 2 * np.pi * np.cumsum(f0) / 16000
            wav = 0.5 * (np.sin(phase) + 0.5 * np.sin(2 * phase))
            pcm = (wav * 32767).astype(np.int16).tobytes()

            chunk_size = 1600
            for i in range(0, len(pcm), chunk_size):
                await audio_ws.send(pcm[i:i+chunk_size])
                await asyncio.sleep(0.01)

            print("[E2E] Audio streamed. Waiting for AI result broadcast on signaling WebSocket...")
            while True:
                msg = await asyncio.wait_for(sig_ws.recv(), timeout=15.0)
                data = json.loads(msg)
                if data.get("type") == "ai_result":
                    print("[E2E SUCCESS] Received live calculated AI result from backend:")
                    print(json.dumps(data, indent=2))
                    assert "status" in data
                    assert "classification" in data
                    break

def main():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    asyncio.run(test_client())
    print("\n[E2E TEST] ALL END-TO-END VERIFICATIONS PASSED!")
    os._exit(0)

if __name__ == "__main__":
    main()
