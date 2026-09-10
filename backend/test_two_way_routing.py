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
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="warning")
    srv = uvicorn.Server(cfg)
    srv.run()

async def test_two_way():
    await asyncio.sleep(2.0)
    print("[TWO-WAY TEST] Connecting Phone A and Phone B signaling...")
    
    async with websockets.connect("ws://127.0.0.1:8766/ws/room_twoway") as ws_a, \
               websockets.connect("ws://127.0.0.1:8766/ws/room_twoway") as ws_b:
        
        # 1. Verify Role Assignment
        init_a = json.loads(await ws_a.recv())
        init_b = json.loads(await ws_b.recv())
        print(f"[TWO-WAY TEST] Phone A init: {init_a}")
        print(f"[TWO-WAY TEST] Phone B init: {init_b}")
        
        assert init_a["role"] == "caller", f"Expected caller, got {init_a['role']}"
        assert init_b["role"] == "receiver", f"Expected receiver, got {init_b['role']}"
        print("[TWO-WAY TEST] PASS: Roles assigned correctly (caller vs receiver).")

        # 2. Connect Audio Streams
        # Phone A streams what it hears from B -> /ws/audio/room_twoway/caller
        # Phone B streams what it hears from A -> /ws/audio/room_twoway/receiver
        url_audio_a = "ws://127.0.0.1:8766/ws/audio/room_twoway/caller?sample_rate=16000&channels=1"
        url_audio_b = "ws://127.0.0.1:8766/ws/audio/room_twoway/receiver?sample_rate=16000&channels=1"
        
        async with websockets.connect(url_audio_a) as audio_a, \
                   websockets.connect(url_audio_b) as audio_b:
            
            print("[TWO-WAY TEST] Audio ingestion sockets connected for caller & receiver.")
            
            # Generate 3.5s synthetic audio for Stream A (Phone B's voice heard by Phone A)
            t = np.linspace(0, 3.5, int(3.5 * 16000), endpoint=False)
            f0 = 160.0 + 20.0 * np.sin(2 * np.pi * 3.0 * t)
            phase = 2 * np.pi * np.cumsum(f0) / 16000
            wav_a = 0.5 * (np.sin(phase) + 0.5 * np.sin(2 * phase))
            pcm_a = (wav_a * 32767).astype(np.int16).tobytes()

            # Stream into Audio A
            print("[TWO-WAY TEST] Streaming audio into Phone A tap (evaluating Phone B)...")
            chunk_size = 1600
            for i in range(0, len(pcm_a), chunk_size):
                await audio_a.send(pcm_a[i:i+chunk_size])
                await asyncio.sleep(0.005)

            # Phone A should receive ai_result for target_role: "caller"
            print("[TWO-WAY TEST] Waiting for AI result on Phone A...")
            msg_a = await asyncio.wait_for(ws_a.recv(), timeout=20.0)
            res_a = json.loads(msg_a)
            print(f"[TWO-WAY TEST] Phone A received: target_role={res_a.get('target_role')}, evaluated_peer={res_a.get('evaluated_peer')}, status={res_a.get('status')}")
            
            assert res_a.get("target_role") == "caller", f"Expected target_role caller, got {res_a.get('target_role')}"
            assert res_a.get("evaluated_peer") == "receiver", f"Expected evaluated_peer receiver, got {res_a.get('evaluated_peer')}"
            
            # Verify Phone B did NOT receive Phone A's result
            try:
                msg_b_leak = await asyncio.wait_for(ws_b.recv(), timeout=0.5)
                leak_data = json.loads(msg_b_leak)
                assert False, f"LEAK DETECTED: Phone B received Phone A's AI result: {leak_data}"
            except asyncio.TimeoutError:
                print("[TWO-WAY TEST] PASS: Phone B did not receive Phone A's result (Zero Cross-Talk).")

            # Now stream into Audio B (Phone A's voice heard by Phone B)
            print("[TWO-WAY TEST] Streaming audio into Phone B tap (evaluating Phone A)...")
            for i in range(0, len(pcm_a), chunk_size):
                await audio_b.send(pcm_a[i:i+chunk_size])
                await asyncio.sleep(0.005)

            print("[TWO-WAY TEST] Waiting for AI result on Phone B...")
            msg_b = await asyncio.wait_for(ws_b.recv(), timeout=20.0)
            res_b = json.loads(msg_b)
            print(f"[TWO-WAY TEST] Phone B received: target_role={res_b.get('target_role')}, evaluated_peer={res_b.get('evaluated_peer')}, status={res_b.get('status')}")
            
            assert res_b.get("target_role") == "receiver", f"Expected target_role receiver, got {res_b.get('target_role')}"
            assert res_b.get("evaluated_peer") == "caller", f"Expected evaluated_peer caller, got {res_b.get('evaluated_peer')}"
            
            print("[TWO-WAY TEST] PASS: Both directional pipelines routed with 100% precision!")

def main():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    asyncio.run(test_two_way())
    print("\n[ALL TWO-WAY VERIFICATIONS PASSED SUCCESSFULLY!]")
    os._exit(0)

if __name__ == "__main__":
    main()
