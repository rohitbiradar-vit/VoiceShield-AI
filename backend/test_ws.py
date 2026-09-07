"""
VoiceShield â€” End-to-End Signaling & Audio Ingestion Verification Test
"""

import asyncio
import json
import websockets

SERVER_SIGNAL_URI = "ws://127.0.0.1:8000/ws/room1"
SERVER_AUDIO_URI = "ws://127.0.0.1:8000/ws/audio/room1"


async def run_signaling_and_audio_test():
    print("\n==========================================")
    print("TEST 1: Connecting Phone A and Phone B...")
    print("==========================================")

    async with websockets.connect(SERVER_SIGNAL_URI) as phone_a, \
               websockets.connect(SERVER_SIGNAL_URI) as phone_b:

        # Step 1: Verify room_ready welcome messages
        msg_a = json.loads(await phone_a.recv())
        msg_b = json.loads(await phone_b.recv())

        print(f"Phone A received: {msg_a}")
        print(f"Phone B received: {msg_b}")
        assert msg_a["type"] == "room_ready"
        assert msg_b["type"] == "room_ready"
        print("âœ” Room assignment verified!")

        # Step 2: Phone A calls Phone B
        print("\nTEST 2: Phone A sends 'call' -> Phone B receives 'call'...")
        await phone_a.send(json.dumps({"type": "call"}))
        b_incoming = json.loads(await phone_b.recv())
        print(f"Phone B received call notification: {b_incoming}")
        assert b_incoming["type"] == "call"
        print("âœ” Incoming call notification verified!")

        # Step 3: Phone B accepts -> Phone A receives 'call_accepted'
        print("\nTEST 3: Phone B sends 'call_accepted' -> Phone A receives 'call_accepted'...")
        await phone_b.send(json.dumps({"type": "call_accepted"}))
        a_accepted = json.loads(await phone_a.recv())
        print(f"Phone A received acceptance: {a_accepted}")
        assert a_accepted["type"] == "call_accepted"
        print("âœ” Call acceptance handshake verified!")

        # Step 4: Phone A creates WebRTC offer -> Phone B receives offer
        print("\nTEST 4: Phone A sends WebRTC Offer -> Phone B receives Offer...")
        mock_offer = {"type": "offer", "sdp": "v=0\r\no=VoiceShield 123 456 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"}
        await phone_a.send(json.dumps(mock_offer))
        b_offer = json.loads(await phone_b.recv())
        print(f"Phone B received offer: {b_offer['type']}")
        assert b_offer["type"] == "offer"
        print("âœ” SDP Offer exchange verified!")

        # Step 5: Phone B sends WebRTC answer -> Phone A receives answer
        print("\nTEST 5: Phone B sends WebRTC Answer -> Phone A receives Answer...")
        mock_answer = {"type": "answer", "sdp": "v=0\r\no=VoiceShield 789 101 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"}
        await phone_b.send(json.dumps(mock_answer))
        a_answer = json.loads(await phone_a.recv())
        print(f"Phone A received answer: {a_answer['type']}")
        assert a_answer["type"] == "answer"
        print("âœ” SDP Answer exchange verified!")

        # Step 6: ICE Candidate exchange
        print("\nTEST 6: ICE Candidate exchange between Phone A and Phone B...")
        cand_a = {"type": "candidate", "candidate": "candidate:1 1 UDP 2122260223 192.168.0.106 50000 typ host", "sdpMid": "0", "sdpMLineIndex": 0}
        await phone_a.send(json.dumps(cand_a))
        b_cand = json.loads(await phone_b.recv())
        assert b_cand["type"] == "candidate"

        cand_b = {"type": "candidate", "candidate": "candidate:2 1 UDP 2122260223 192.168.0.107 50001 typ host", "sdpMid": "0", "sdpMLineIndex": 0}
        await phone_b.send(json.dumps(cand_b))
        a_cand = json.loads(await phone_a.recv())
        assert a_cand["type"] == "candidate"
        print("âœ” ICE candidates exchange verified!")

        # Step 7: Audio Ingestion Stream
        print("\nTEST 7: Testing live raw PCM audio ingestion to /ws/audio/room1...")
        async with websockets.connect(SERVER_AUDIO_URI) as audio_ws:
            # Send simulated 16kHz PCM audio frame (e.g. 640 bytes = 20ms)
            fake_pcm_frame = bytes([0] * 640)
            await audio_ws.send(fake_pcm_frame)
            await asyncio.sleep(0.1)
            print("âœ” Raw audio frame sent to backend successfully!")

        # Step 8: Call Teardown / Hangup
        print("\nTEST 8: Phone A sends 'hangup' -> Phone B receives 'hangup'...")
        await phone_a.send(json.dumps({"type": "hangup"}))
        b_hangup = json.loads(await phone_b.recv())
        assert b_hangup["type"] == "hangup"
        print("âœ” Hangup handshake verified!")

    print("\n==========================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================\n")


if __name__ == "__main__":
    asyncio.run(run_signaling_and_audio_test())
