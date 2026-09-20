# -*- coding: utf-8 -*-
"""Verification test script for Leibnitz 5.0 for Sarvam."""
import sys
import os
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.adapter import registry
import core.dsp_classical
import core.dsp_speech
from sarvam.client import SarvamClient

def run_tests():
    print("--- [TEST 1] Registered Blocks ---")
    blocks = registry.list_blocks()
    for b in blocks:
        print(f"  * {b['id']} - {b['name']} ({b['version']})")
    assert len(blocks) >= 4, "Expected at least 4 blocks registered"

    print("\n--- [TEST 2] Running Modular Pipeline ---")
    sr = 16000.0
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    sig = 0.6 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.random.normal(0, 1, len(t))

    pipeline = [
        {"id": "vad_cleaner", "params": {}},
        {"id": "telephony_bandpass", "params": {}},
        {"id": "indic_phonetics", "params": {}},
        {"id": "fft_spectrum", "params": {}}
    ]

    result = registry.execute_pipeline(sig, sr, pipeline)
    assert "processed_signal" in result
    print(f"  Processed signal samples: {len(result['processed_signal'])}")
    for stage in result["stages"]:
        print(f"  [OK] Stage: {stage['name']} -> metrics: {list(stage['metrics'].keys())}")

    print("\n--- [TEST 3] Sarvam AI Client Simulation ---")
    client = SarvamClient()
    stt = client.speech_to_text(b"dummy_wav_data", language_code="hi-IN")
    print(f"  [STT OK] Transcript: {stt['transcript']}")
    assert "नमस्ते" in stt["transcript"] or "सर्वम्" in stt["transcript"]

    tts = client.text_to_speech("नमस्ते संसार", target_language_code="hi-IN")
    print(f"  [TTS OK] Generated audio b64 length: {len(tts['audios'][0])}")
    assert len(tts["audios"][0]) > 100

    chat = client.chat_completion("Explain signal SNR", {"snr_db": 25.0, "mean_pitch_f0_hz": 180.0}, "hi")
    print(f"  [Chat OK] Copilot response: {chat['message'][:80]}...")
    assert len(chat["message"]) > 10

    print("\n--- [TEST 4] Flask Web Endpoints ---")
    from app import app
    test_client = app.test_client()
    r1 = test_client.get('/')
    assert r1.status_code == 200, f"Expected 200 from /, got {r1.status_code}"
    print("  [OK] GET / -> 200")

    r2 = test_client.get('/api/status')
    assert r2.status_code == 200, f"Expected 200 from /api/status, got {r2.status_code}"
    print("  [OK] GET /api/status -> 200")

    r3 = test_client.post('/api/load-preset/sanskrit_chant')
    assert r3.status_code == 200 and r3.json.get("success"), "Failed loading preset"
    filename = r3.json.get("filename")
    print(f"  [OK] POST /api/load-preset/sanskrit_chant -> {filename}")

    r4 = test_client.post('/api/process-pipeline', json={"filename": filename})
    assert r4.status_code == 200 and r4.json.get("success"), "Failed pipeline"
    print(f"  [OK] POST /api/process-pipeline -> stages: {len(r4.json.get('stages'))}")

    r5 = test_client.post('/api/sarvam/stt', json={"filename": filename, "language_code": "hi-IN"})
    assert r5.status_code == 200 and "transcript" in r5.json, "Failed STT"
    print(f"  [OK] POST /api/sarvam/stt -> transcript returned")

    r6 = test_client.post('/api/sarvam/tts', json={"text": "नमस्ते", "target_language_code": "hi-IN"})
    assert r6.status_code == 200 and r6.json.get("success"), "Failed TTS"
    print(f"  [OK] POST /api/sarvam/tts -> generated WAV {r6.json.get('filename')}")

    r7 = test_client.post('/api/sarvam/chat', json={"message": "What is the pitch?", "language": "en"})
    assert r7.status_code == 200 and "message" in r7.json, "Failed Chat"
    print(f"  [OK] POST /api/sarvam/chat -> copilot responded")

    print("\n========================================================")
    print(" ALL LEIBNITZ-SARVAM MODULE & ENDPOINT TESTS PASSED! ")
    print("========================================================")

if __name__ == "__main__":
    run_tests()
