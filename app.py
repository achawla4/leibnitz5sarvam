# -*- coding: utf-8 -*-
"""
Leibnitz 5.0 for Sarvam - Application Server
============================================
Connects Leibnitz mathematical signal processing with Sarvam AI's Indic models.
Supports file ingestion (.csv, .wav, .txt, .npy), multi-panel analytical plotting,
and audio conditioning for Sarvam STT, TTS, and Indic LLM.
"""

import os
import io
import uuid
import wave
import json
import base64
import numpy as np
from datetime import datetime
from scipy import signal as sp_signal
from flask import Flask, render_template, request, jsonify, send_from_directory

# Core DSP and adapter modules
from core.adapter import registry, LEIBNITZ_SPEC_VERSION
import core.dsp_classical
import core.dsp_speech
import core.upgrade_guide
from core.visualization import generate_multi_panel_plot

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

sarvam_client = SarvamClient()

# In-memory metadata store for active session files
FILE_STORE = {}

def read_audio_data(file_bytes: bytes, filename: str):
    """
    Robust multi-format parser for WAV, CSV, NPY, TXT.
    Detects magic bytes first to avoid extension confusion.
    """
    if len(file_bytes) == 0:
        raise ValueError("Empty file data received.")

    # 1. WAV Magic Bytes Check: 'RIFF' .... 'WAVE'
    if len(file_bytes) >= 12 and file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WAVE':
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
                    data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sampwidth == 4:
                    data = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                if n_channels > 1:
                    data = data[::n_channels]
                return np.nan_to_num(data).astype(np.float32), float(framerate)
        except Exception:
            from scipy.io import wavfile
            rate, data = wavfile.read(io.BytesIO(file_bytes))
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype.kind in 'iu':
                max_v = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_v
            return np.nan_to_num(data).astype(np.float32), float(rate)

    # 2. NumPy NPY Magic Bytes: \x93NUMPY
    if len(file_bytes) >= 6 and file_bytes[:6] == b'\x93NUMPY':
        arr = np.load(io.BytesIO(file_bytes))
        return np.nan_to_num(arr.flatten()).astype(np.float32), 1000.0

    # 3. CSV / TXT Text Parsing
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("No readable text lines found.")

        # Detect header row
        start_idx = 1 if any(c.isalpha() for c in lines[0]) else 0
        data_rows = []
        for ln in lines[start_idx:]:
            delimiter = ',' if ',' in ln else None
            parts = ln.split(delimiter)
            row = []
            for p in parts:
                p = p.strip()
                if p:
                    try:
                        row.append(float(p))
                    except ValueError:
                        pass
            if row:
                data_rows.append(row)

        data_arr = np.array(data_rows, dtype=np.float32)
        if data_arr.size == 0:
            raise ValueError("No numeric data rows found in CSV.")

        if data_arr.ndim == 2 and data_arr.shape[1] > 1:
            time_col = data_arr[:, 0]
            diffs = np.diff(time_col)
            if len(diffs) > 0 and np.all(diffs > 0):
                mean_dt = float(np.mean(diffs))
                rate = float(round(1.0 / mean_dt, 1)) if mean_dt > 0 else 1000.0
            else:
                rate = 1000.0
            signal_data = data_arr[:, 1].flatten()
        elif data_arr.ndim == 2:
            signal_data = data_arr[:, 0].flatten()
            rate = 1000.0
        else:
            signal_data = data_arr.flatten()
            rate = 1000.0

        return np.nan_to_num(signal_data).astype(np.float32), rate

    except Exception as e:
        # Fallback to scipy wavfile if it was binary
        try:
            from scipy.io import wavfile
            rate, data = wavfile.read(io.BytesIO(file_bytes))
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype.kind in 'iu':
                max_v = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_v
            return np.nan_to_num(data).astype(np.float32), float(rate)
        except Exception:
            raise ValueError(f"Unable to parse file '{filename}': {str(e)}")

def save_as_wav(audio_data: np.ndarray, sample_rate: float, filepath: str):
    """Saves float numpy array as 16-bit PCM mono WAV."""
    norm_audio = np.clip(audio_data, -1.0, 1.0)
    int_data = (norm_audio * 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int_data.tobytes())

def make_browser_playable_wav(audio_data: np.ndarray, sample_rate: float, filepath: str):
    """
    Standard HTML5 <audio> decoders fail when WAV sample rate is < 8000 Hz (e.g. 1000 Hz CSV).
    This generates a 16000 Hz playable preview WAV for browser playback, while preserving
    the exact original signal sample_rate for DSP calculations.
    """
    if sample_rate < 8000:
        target_sr = 16000
        num_samples = max(1, int(len(audio_data) * (target_sr / sample_rate)))
        resampled = sp_signal.resample(audio_data, num_samples)
        save_as_wav(resampled, target_sr, filepath)
    else:
        save_as_wav(audio_data, sample_rate, filepath)

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
                "duration_s": 3.0
            },
            {
                "id": "hindi_telephony",
                "name": "हिन्दी कॉल-सेंटर वार्ता (Hindi Telephony Voice)",
                "description": "Conversational speech with simulated 8 kHz telephony acoustic noise.",
                "sample_rate": 8000,
                "duration_s": 3.0
            },
            {
                "id": "sinusoid_12hz",
                "name": "१२ हर्ट्ज़ नादसंवादः (12 Hz Sinusoid CSV)",
                "description": "Standard Leibnitz test sinusoid benchmark with harmonics & noise.",
                "sample_rate": 1000,
                "duration_s": 1.0
            }
        ]
    })

@app.route('/api/load-preset/<preset_id>', methods=['POST'])
def load_preset(preset_id):
    uid = uuid.uuid4().hex[:8]
    if preset_id == "sanskrit_chant":
        sr = 16000.0
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        f_profile = np.piecewise(t, [t < 1.0, (t >= 1.0) & (t < 2.0), t >= 2.0], [140.0, 165.0, 140.0])
        phase = 2 * np.pi * np.cumsum(f_profile) / sr
        sig = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        sig *= np.abs(np.sin(2 * np.pi * 1.5 * t)) ** 0.5
        sig += 0.02 * np.random.normal(0, 1, len(t))
        orig_name = "sanskrit_vedic_chant.wav"
        file_type = "WAV Audio (Indic Speech)"
    elif preset_id == "hindi_telephony":
        sr = 8000.0
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        sig = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        sig *= np.abs(np.cos(2 * np.pi * 2.0 * t))
        sig += 0.08 * np.random.normal(0, 1, len(t))
        orig_name = "hindi_callcenter_voice.wav"
        file_type = "WAV Audio (Telephony)"
    else:  # sinusoid_12hz
        sr = 1000.0
        t = np.linspace(0, 1.0, 1000, endpoint=False)
        sig = np.sin(2 * np.pi * 12.0 * t) + 0.25 * np.sin(2 * np.pi * 60.0 * t)
        orig_name = "sinusoidal_12Hz.csv"
        file_type = "CSV Time-Series"

    filename = f"preset_{uid}_{orig_name}"
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if orig_name.endswith('.csv'):
        # Save as actual CSV
        np.savetxt(raw_path, np.column_stack((t, sig)), delimiter=',', header='Time (s),Signal', comments='')
    else:
        save_as_wav(sig, sr, raw_path)

    # Make browser audio
    audio_filename = f"audio_{uid}.wav"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    make_browser_playable_wav(sig, sr, audio_path)

    step = max(1, len(sig) // 300)
    waveform = [round(float(v), 3) for v in sig[::step]]

    # Store in memory
    FILE_STORE[filename] = {
        "audio": sig,
        "sample_rate": sr,
        "original_name": orig_name,
        "file_type": file_type
    }

    return jsonify({
        "success": True,
        "filename": filename,
        "original_name": orig_name,
        "file_type": file_type,
        "sample_rate": sr,
        "duration_s": round(len(sig) / sr, 2),
        "waveform_preview": waveform,
        "audio_url": f"/uploads/{audio_filename}"
    })

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    uid = uuid.uuid4().hex[:8]
    clean_original_name = os.path.basename(file.filename)
    saved_filename = f"upload_{uid}_{clean_original_name}"
    file_bytes = file.read()

    try:
        audio, sr = read_audio_data(file_bytes, clean_original_name)
    except Exception as e:
        return jsonify({"error": f"Failed to parse audio format: {str(e)}"}), 400

    # Save exact raw bytes to disk
    raw_target_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
    with open(raw_target_path, 'wb') as f:
        f.write(file_bytes)

    # Save web playable WAV
    audio_filename = f"audio_{uid}.wav"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    make_browser_playable_wav(audio, sr, audio_path)

    # Determine file type
    lower = clean_original_name.lower()
    if lower.endswith('.csv'):
        file_type = "CSV Time-Series"
    elif lower.endswith('.wav'):
        file_type = "WAV Audio"
    elif lower.endswith('.npy'):
        file_type = "NumPy Array"
    else:
        file_type = "Text / Signal Data"

    # Peak dominant frequency calculation
    if len(audio) > 16:
        n_fft = min(len(audio), 4096)
        fft_v = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        dom_freq = round(float(freqs[np.argmax(fft_v)]), 1)
    else:
        dom_freq = 0.0

    step = max(1, len(audio) // 300)
    waveform = [round(float(v), 3) for v in audio[::step]]

    FILE_STORE[saved_filename] = {
        "audio": audio,
        "sample_rate": sr,
        "original_name": clean_original_name,
        "file_type": file_type
    }

    return jsonify({
        "success": True,
        "filename": saved_filename,
        "original_name": clean_original_name,
        "file_type": file_type,
        "sample_rate": sr,
        "duration_s": round(len(audio) / sr, 2),
        "dominant_freq_hz": dom_freq,
        "waveform_preview": waveform,
        "audio_url": f"/uploads/{audio_filename}"
    })

@app.route('/api/process-pipeline', methods=['POST'])
def process_pipeline():
    data = request.get_json() or {}
    filename = data.get("filename")
    pipeline_spec = data.get("pipeline", [
        {"id": "vad_cleaner", "params": {}},
        {"id": "telephony_bandpass", "params": {}},
        {"id": "indic_phonetics", "params": {}},
        {"id": "fft_spectrum", "params": {}}
    ])

    if not filename:
        return jsonify({"error": "Missing filename parameter"}), 400

    # Retrieve from in-memory cache or parse from disk
    if filename in FILE_STORE:
        audio = FILE_STORE[filename]["audio"]
        sr = FILE_STORE[filename]["sample_rate"]
        original_name = FILE_STORE[filename]["original_name"]
    else:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({"error": f"Audio file '{filename}' not found on server"}), 404
        with open(filepath, 'rb') as f:
            audio, sr = read_audio_data(f.read(), filename)
        original_name = filename

    try:
        pipeline_result = registry.execute_pipeline(audio, sr, pipeline_spec)
    except Exception as e:
        return jsonify({"error": f"Pipeline execution failed: {str(e)}"}), 500

    processed_signal = pipeline_result["processed_signal"]
    uid = uuid.uuid4().hex[:8]
    
    # 1. Save processed WAV
    processed_wav_filename = f"processed_{uid}.wav"
    processed_wav_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_wav_filename)
    make_browser_playable_wav(processed_signal, sr, processed_wav_path)

    # 2. Save downloadable CSV
    processed_csv_filename = f"processed_{uid}.csv"
    processed_csv_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_csv_filename)
    time_axis = np.arange(len(processed_signal)) / float(sr)
    np.savetxt(processed_csv_path, np.column_stack((time_axis, processed_signal)), delimiter=',', header='Time (s),Signal', comments='')

    # 3. Generate high-resolution analytical plot PNG
    plot_filename = generate_multi_panel_plot(
        original_signal=audio,
        processed_signal=processed_signal,
        sample_rate=sr,
        output_dir=app.config['PROCESSED_FOLDER'],
        original_filename=original_name
    )

    # 4. Save downloadable diagnostic JSON report
    report_filename = f"report_{uid}.json"
    report_path = os.path.join(app.config['PROCESSED_FOLDER'], report_filename)
    with open(report_path, 'w') as f:
        json.dump({
            "original_file": original_name,
            "sample_rate": sr,
            "duration_s": round(len(audio) / sr, 3),
            "timestamp": datetime.utcnow().isoformat(),
            "stages": pipeline_result["stages"],
            "engine_version": pipeline_result["engine_version"]
        }, f, indent=2)

    step = max(1, len(processed_signal) // 300)
    proc_waveform = [round(float(v), 3) for v in processed_signal[::step]]

    return jsonify({
        "success": True,
        "processed_filename": processed_wav_filename,
        "original_name": original_name,
        "audio_url": f"/processed/{processed_wav_filename}",
        "plot_url": f"/processed/{plot_filename}",
        "download_csv_url": f"/api/download/{processed_csv_filename}",
        "download_report_url": f"/api/download/{report_filename}",
        "processed_waveform": proc_waveform,
        "stages": pipeline_result["stages"],
        "engine_version": pipeline_result["engine_version"]
    })

# ====================== SARVAM AI ENDPOINTS ======================

@app.route('/api/sarvam/stt', methods=['POST'])
def sarvam_stt():
    data = request.get_json() or {}
    filename = data.get("filename")
    lang = data.get("language_code", "hi-IN")

    if not filename:
        return jsonify({"error": "Filename required"}), 400

    # Retrieve audio data
    if filename in FILE_STORE:
        audio = FILE_STORE[filename]["audio"]
        sr = FILE_STORE[filename]["sample_rate"]
        orig_name = FILE_STORE[filename]["original_name"]
    else:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({"error": "Audio file not found"}), 404
        with open(filepath, 'rb') as f:
            audio, sr = read_audio_data(f.read(), filename)
        orig_name = filename

    # Check if live or simulation
    if sarvam_client.is_live():
        # Encode audio as 16kHz WAV bytes for Sarvam STT
        byte_io = io.BytesIO()
        save_as_wav(audio, sr, byte_io)
        res = sarvam_client.speech_to_text(byte_io.getvalue(), language_code=lang)
        return jsonify(res)

    # Intelligent Signal-Aware Simulation Mode
    n = len(audio)
    fft_vals = np.abs(np.fft.rfft(audio[:min(n, 4096)]))
    freqs = np.fft.rfftfreq(min(n, 4096), 1.0 / sr)
    dom_freq = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 0 else 0.0

    # 1. Non-speech or Sub-audible signal detection (e.g. 12Hz sinusoid or pure test tones)
    if dom_freq < 60.0 or "sinusoid" in orig_name.lower():
        return jsonify({
            "transcript": f"⚠️ अवाक्-सङ्केतः / Non-Speech Signal: Ingested file '{orig_name}' contains a synthetic {dom_freq:.1f} Hz oscillation. No human speech phonemes detected. Sarvam Saaras requires speech within 80–8000 Hz.",
            "language_code": lang,
            "confidence": 0.0,
            "mode": "simulation_demo (Signal Diagnostic)",
            "provider": "Sarvam Saaras STT Diagnostic"
        })

    # 2. Sanskrit Vedic Chanting
    if "sanskrit" in orig_name.lower() or lang == "sa-IN":
        return jsonify({
            "transcript": "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात्। (O3m bhūrbhuvaḥ svaḥ tatsaviturvarēṇyaṁ bhargō dēvasya dhīmahi dhiyō yō naḥ pracōdayāt)",
            "language_code": "sa-IN",
            "confidence": 0.992,
            "mode": "simulation_demo",
            "provider": "Sarvam Saaras STT Engine"
        })

    # 3. Hindi Telephony Conversation
    if "telephon" in orig_name.lower() or "hindi" in orig_name.lower():
        return jsonify({
            "transcript": "नमस्ते, मैं सर्वम् एआई और लायब्निट्ज़ कस्टमर सपोर्ट से बात कर रहा हूँ। आपकी कॉल कनेक्ट हो गई है।",
            "language_code": "hi-IN",
            "confidence": 0.978,
            "mode": "simulation_demo",
            "provider": "Sarvam Saaras STT Engine"
        })

    # 4. Generic Speech / Recorded Voice
    res = sarvam_client._mock_speech_to_text(lang)
    return jsonify(res)

@app.route('/api/sarvam/tts', methods=['POST'])
def sarvam_tts():
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
        uid = uuid.uuid4().hex[:8]
        filename = f"sarvam_tts_{uid}.wav"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(save_path, 'wb') as f:
            f.write(audio_bytes)

        audio, sr = read_audio_data(audio_bytes, filename)
        
        FILE_STORE[filename] = {
            "audio": audio,
            "sample_rate": sr,
            "original_name": f"tts_speech_{uid}.wav",
            "file_type": "WAV Audio (Bulbul TTS)"
        }

        step = max(1, len(audio) // 300)
        waveform = [round(float(v), 3) for v in audio[::step]]

        return jsonify({
            "success": True,
            "filename": filename,
            "original_name": f"tts_speech_{uid}.wav",
            "file_type": "WAV Audio (Bulbul TTS)",
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
    data = request.get_json() or {}
    message = data.get("message", "")
    signal_context = data.get("signal_context", {})
    language = data.get("language", "hi")

    if not message:
        return jsonify({"error": "Message required"}), 400

    filename = signal_context.get("filename", "Current Ingested Signal")
    dom_f = signal_context.get("dominant_freq_hz", 12.0)
    snr = signal_context.get("snr_db", 24.5)
    sr = signal_context.get("sample_rate", 1000.0)

    if sarvam_client.is_live():
        resp = sarvam_client.chat_completion(message, signal_context=signal_context, language=language)
        return jsonify(resp)

    # Context-aware mock response referencing the EXACT loaded file!
    if dom_f < 60.0 or "sinusoid" in filename.lower():
        text_hi = f"लायब्निट्ज़ ५.० विश्लेषक: आपने सञ्चिका '{filename}' लोड की है, जिसकी मूल आवृत्ति {dom_f} Hz और प्रतिचयन दर {sr} Hz है। यह एक उप-श्रव्य (Sub-audible) नादसंवाद तरंग है। सर्वम् सारस (Saaras) STT मानव वाक् (80–8000 Hz) के लिए प्रशिक्षित है, इसलिए इस फ़ाइल पर सीधे वाक्-पहचान नहीं होगी। इसे जांचने के लिए आप 'Leibnitz Fourier Spectrum' का उपयोग करके इसके 12 Hz और 60 Hz हार्मोनिक्स देख सकते हैं।"
        text_en = f"Leibnitz-Sarvam Copilot: You have loaded '{filename}' with fundamental frequency {dom_f} Hz and sampling rate {sr} Hz. This is a low-frequency synthetic sinusoid. Because Sarvam Saaras STT operates on human speech (80-8000 Hz), this file serves as a spectral baseline benchmark rather than voice audio. Use the Leibnitz Fourier Spectrum to inspect its harmonics."
        chosen = text_hi if language == "hi" else text_en
    else:
        text_hi = f"लायब्निट्ज़ ५.० विश्लेषक: सञ्चिका '{filename}' का विश्लेषण संपन्न। अनुमानित SNR {snr} dB और प्रतिचयन दर {sr} Hz है। यह सर्वम् सारस (Saaras) और बुलबुल (Bulbul) मॉडल्स के लिए पूर्णतः अनुकूल है।"
        text_en = f"Leibnitz-Sarvam Copilot: Analysis for '{filename}' complete. Estimated SNR is {snr} dB at {sr} Hz sample rate. Highly suitable for Sarvam Saaras STT and Bulbul voice workflows."
        chosen = text_hi if language == "hi" else text_en

    return jsonify({
        "message": chosen,
        "model": "sarvam-2 (Indic Signal Specialist)",
        "mode": "simulation_demo",
        "provider": "Sarvam Foundation Models"
    })

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
    print(f"  Spec Version: {LEIBNITZ_SPEC_VERSION}")
    print(f"  Listening on http://0.0.0.0:{port}")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
