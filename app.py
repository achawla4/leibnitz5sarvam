# -*- coding: utf-8 -*-
"""
Leibnitz 5.0 for Sarvam - Application Server
============================================
Seamlessly connects the Leibnitz mathematical and signal processing workflow
with Sarvam AI's Indic language foundation models, speech recognition (Saaras),
voice synthesis (Bulbul), and Indic voice agent pipelines.

Engineered with modular block abstraction for forward compatibility with
future Leibnitz 8.0, 10.0, or higher releases.
"""

import os
import io
import uuid
import wave
import json
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Core DSP and adapter modules
from core.adapter import registry, LEIBNITZ_SPEC_VERSION
import core.dsp_classical  # registers default blocks
import core.dsp_speech
import core.upgrade_guide

# Sarvam AI service
from sarvam.client import SarvamClient, INDIC_LANGUAGES

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "leibnitz-sarvam-indic-secret-key-2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Initialize Sarvam Client
sarvam_client = SarvamClient()

def read_audio_data(file_bytes: bytes, filename: str):
    """Parses WAV audio bytes or CSV/NPY files into 1D float numpy array and sample rate."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.wav') or filename.startswith('mic_recording'):
        try:
            with wave.open(io.BytesIO(file_bytes), 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                raw_frames = wav_file.readframes(n_frames)
                
                if sampwidth == 2:
                    data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                elif sampwidth == 4:
                    data = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                if n_channels > 1:
                    data = data[::n_channels]  # Extract channel 0 (mono)
                    
                return data, float(framerate)
        except Exception as e:
            # Fallback to scipy if standard wave parser encounters exotic chunks
            from scipy.io import wavfile
            rate, data = wavfile.read(io.BytesIO(file_bytes))
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype.kind in 'iu':
                max_v = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_v
            return data.astype(np.float32), float(rate)

    elif filename_lower.endswith('.csv'):
        # CSV load
        arr = np.loadtxt(io.BytesIO(file_bytes), delimiter=',', skiprows=1)
        if arr.ndim == 2 and arr.shape[1] > 1:
            diffs = np.diff(arr[:, 0])
            rate = 1.0 / np.mean(diffs) if np.all(diffs > 0) else 1000.0
            return arr[:, 1].astype(np.float32), float(round(rate, 1))
        return arr.flatten().astype(np.float32), 1000.0

    elif filename_lower.endswith('.npy'):
        arr = np.load(io.BytesIO(file_bytes))
        return arr.flatten().astype(np.float32), 1000.0

    else:
        # Generic text
        arr = np.loadtxt(io.BytesIO(file_bytes))
        return arr.flatten().astype(np.float32), 1000.0

def save_as_wav(audio_data: np.ndarray, sample_rate: float, filepath: str):
    """Saves float numpy array as 16-bit PCM mono WAV."""
    norm_audio = np.clip(audio_data, -1.0, 1.0)
    int_data = (norm_audio * 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int_data.tobytes())

# ============================ ROUTES ============================

@app.route('/')
def index():
    return render_template(
        'index.html',
        leibnitz_version=LEIBNITZ_SPEC_VERSION,
        languages=INDIC_LANGUAGES,
        sarvam_live=sarvam_client.is_live()
    )

@app.route('/api/status', methods=['GET'])
def api_status():
    """Health check, version negotiation info, and registered blocks."""
    return jsonify({
        "edition": "Leibnitz 5.0 for Sarvam",
        "leibnitz_spec_version": LEIBNITZ_SPEC_VERSION,
        "upgradable_targets": ["5.0", "8.0", "10.0+"],
        "sarvam_connected": True,
        "sarvam_mode": "Live Sarvam API" if sarvam_client.is_live() else "High-Fidelity Simulation / Demo Mode",
        "blocks_available": registry.list_blocks()
    })

@app.route('/api/blocks', methods=['GET'])
def list_blocks():
    return jsonify({"blocks": registry.list_blocks()})

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify({
        "presets": [
            {
                "id": "sanskrit_chant",
                "name": "संस्कृत मन्त्रः (Sanskrit Vedic Recitation)",
                "description": "Vedic recitation with classical pitch swaras (Udatta, Anudatta, Svarita).",
                "sample_rate": 16000,
                "duration_s": 2.5
            },
            {
                "id": "hindi_telephony",
                "name": "हिन्दी कॉल-सेंटर वार्ता (Hindi Telephony Voice)",
                "description": "Conversational speech with simulated 8 kHz telephony acoustic noise.",
                "sample_rate": 8000,
                "duration_s": 3.0
            },
            {
                "id": "synthetic_chirp",
                "name": "नाद-चीत्कारः (Acoustic Sweep 100-3000 Hz)",
                "description": "Harmonic acoustic sweep for frequency response testing.",
                "sample_rate": 16000,
                "duration_s": 1.5
            }
        ]
    })

@app.route('/api/load-preset/<preset_id>', methods=['POST'])
def load_preset(preset_id):
    """Generates synthetic preset audio signal on the fly."""
    uid = uuid.uuid4().hex[:8]
    if preset_id == "sanskrit_chant":
        sr = 16000.0
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        # 3 distinct intonation swaras (140 Hz, 165 Hz, 190 Hz)
        f_profile = np.piecewise(t, [t < 1.0, (t >= 1.0) & (t < 2.0), t >= 2.0], [140.0, 165.0, 140.0])
        phase = 2 * np.pi * np.cumsum(f_profile) / sr
        sig = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        sig *= np.abs(np.sin(2 * np.pi * 1.5 * t)) ** 0.5
        sig += 0.02 * np.random.normal(0, 1, len(t))
        filename = f"preset_sanskrit_{uid}.wav"
    elif preset_id == "hindi_telephony":
        sr = 8000.0
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        sig = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        sig *= np.abs(np.cos(2 * np.pi * 2.0 * t))
        sig += 0.08 * np.random.normal(0, 1, len(t)) # telephony background noise
        filename = f"preset_hindi_telecom_{uid}.wav"
    else:
        sr = 16000.0
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = np.sin(2 * np.pi * (100.0 + (3000.0 - 100.0) * t / (2 * 2.0)) * t)
        filename = f"preset_chirp_{uid}.wav"

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    save_as_wav(sig, sr, save_path)

    # Downsample points for frontend preview waveform
    step = max(1, len(sig) // 300)
    waveform = [round(float(v), 3) for v in sig[::step]]

    return jsonify({
        "success": True,
        "filename": filename,
        "sample_rate": sr,
        "duration_s": round(len(sig) / sr, 2),
        "waveform_preview": waveform,
        "audio_url": f"/uploads/{filename}"
    })

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Handles file uploads or recorded microphone audio from the browser."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_bytes = file.read()
    
    try:
        audio, sr = read_audio_data(file_bytes, file.filename)
    except Exception as e:
        return jsonify({"error": f"Failed to parse audio format: {str(e)}"}), 400

    # Persist as standard WAV
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    save_as_wav(audio, sr, target_path)

    step = max(1, len(audio) // 300)
    waveform = [round(float(v), 3) for v in audio[::step]]

    return jsonify({
        "success": True,
        "filename": filename,
        "sample_rate": sr,
        "duration_s": round(len(audio) / sr, 2),
        "waveform_preview": waveform,
        "audio_url": f"/uploads/{filename}"
    })

@app.route('/api/process-pipeline', methods=['POST'])
def process_pipeline():
    """
    Executes modular Leibnitz DSP pipeline on the selected audio file.
    Works identically for Leibnitz 5.0 and future Leibnitz 8.0/10.0+ registered blocks.
    """
    data = request.get_json() or {}
    filename = data.get("filename")
    pipeline_spec = data.get("pipeline", [
        {"id": "vad_cleaner", "params": {}},
        {"id": "indic_phonetics", "params": {}},
        {"id": "fft_spectrum", "params": {}}
    ])

    if not filename:
        return jsonify({"error": "Missing filename parameter"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Audio file not found"}), 404

    with open(filepath, 'rb') as f:
        audio, sr = read_audio_data(f.read(), filename)

    try:
        pipeline_result = registry.execute_pipeline(audio, sr, pipeline_spec)
    except Exception as e:
        return jsonify({"error": f"Pipeline execution failed: {str(e)}"}), 500

    processed_signal = pipeline_result["processed_signal"]
    processed_filename = f"processed_{uuid.uuid4().hex[:8]}.wav"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    save_as_wav(processed_signal, sr, processed_path)

    # Downsampled preview for frontend visualization
    step = max(1, len(processed_signal) // 300)
    proc_waveform = [round(float(v), 3) for v in processed_signal[::step]]

    return jsonify({
        "success": True,
        "processed_filename": processed_filename,
        "audio_url": f"/processed/{processed_filename}",
        "processed_waveform": proc_waveform,
        "stages": pipeline_result["stages"],
        "engine_version": pipeline_result["engine_version"]
    })

# ====================== SARVAM AI ENDPOINTS ======================

@app.route('/api/sarvam/stt', methods=['POST'])
def sarvam_stt():
    """Transcribes audio using Sarvam Saaras STT."""
    data = request.get_json() or {}
    filename = data.get("filename")
    lang = data.get("language_code", "hi-IN")

    if not filename:
        return jsonify({"error": "Filename required"}), 400

    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Audio file not found"}), 404

    with open(filepath, 'rb') as f:
        wav_bytes = f.read()

    res = sarvam_client.speech_to_text(wav_bytes, language_code=lang)
    return jsonify(res)

@app.route('/api/sarvam/tts', methods=['POST'])
def sarvam_tts():
    """Synthesizes speech using Sarvam Bulbul TTS and passes audio directly into Leibnitz DSP."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    lang = data.get("target_language_code", "hi-IN")
    speaker = data.get("speaker", "meera")

    if not text:
        return jsonify({"error": "Text is required for TTS synthesis"}), 400

    tts_res = sarvam_client.text_to_speech(text, target_language_code=lang, speaker=speaker)

    if "audios" in tts_res and len(tts_res["audios"]) > 0:
        b64_audio = tts_res["audios"][0]
        audio_bytes = base64.b64decode(b64_audio)
        filename = f"sarvam_tts_{uuid.uuid4().hex[:8]}.wav"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(save_path, 'wb') as f:
            f.write(audio_bytes)

        audio, sr = read_audio_data(audio_bytes, filename)
        step = max(1, len(audio) // 300)
        waveform = [round(float(v), 3) for v in audio[::step]]

        return jsonify({
            "success": True,
            "filename": filename,
            "audio_url": f"/uploads/{filename}",
            "sample_rate": sr,
            "duration_s": round(len(audio) / sr, 2),
            "waveform_preview": waveform,
            "mode": tts_res.get("mode"),
            "provider": tts_res.get("provider")
        })

    return jsonify({"error": "TTS synthesis failed"}), 500

@app.route('/api/sarvam/chat', methods=['POST'])
def sarvam_chat():
    """Consults the Sarvam Indic LLM Copilot for acoustic and DSP insights."""
    data = request.get_json() or {}
    message = data.get("message", "")
    signal_context = data.get("signal_context", {})
    language = data.get("language", "hi")

    if not message:
        return jsonify({"error": "Message required"}), 400

    resp = sarvam_client.chat_completion(message, signal_context=signal_context, language=language)
    return jsonify(resp)

# ====================== STATIC & DOWNLOADS ======================

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/api/download/<path:filename>')
def download_file(filename):
    clean_name = os.path.basename(filename)
    p_path = os.path.join(app.config['PROCESSED_FOLDER'], clean_name)
    if os.path.exists(p_path):
        return send_from_directory(app.config['PROCESSED_FOLDER'], clean_name, as_attachment=True)
    u_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_name)
    if os.path.exists(u_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], clean_name, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5005))
    print(f"==================================================")
    print(f"  Leibnitz 5.0 for Sarvam (Indic AI & DSP)")
    print(f"  Spec Version: {LEIBNITZ_SPEC_VERSION} | Modular Upgrades: v8.0/v10.0 Ready")
    print(f"  Sarvam Mode: {'LIVE API' if sarvam_client.is_live() else 'DEMO / SIMULATION'}")
    print(f"  Listening on http://0.0.0.0:{port}")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
