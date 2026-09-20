# -*- coding: utf-8 -*-
"""
Sarvam AI API Client & Workflow Orchestrator
============================================
Provides unified access to:
- Sarvam Saaras (Speech-to-Text)
- Sarvam Bulbul (Text-to-Speech)
- Sarvam Indic LLM / Copilot (Sarvam-1 / Sarvam-2)
- Sarvam Mayura (Indic Translation)

Supports dual-mode execution:
- LIVE MODE: When SARVAM_API_KEY is configured in env or request headers.
- MOCK/DEMO MODE: High-fidelity realistic Indic responses for offline evaluation & testing.
"""

import os
import json
import requests
import numpy as np
import base64
import io
import wave
from typing import Dict, Any, Optional

SARVAM_API_BASE = os.environ.get("SARVAM_API_BASE", "https://api.sarvam.ai")
DEFAULT_API_KEY = os.environ.get("SARVAM_API_KEY", "")

INDIC_LANGUAGES = {
    "hi-IN": "Hindi (हिन्दी)",
    "sa-IN": "Sanskrit (संस्कृतम्)",
    "ta-IN": "Tamil (தமிழ்)",
    "te-IN": "Telugu (తెలుగు)",
    "bn-IN": "Bengali (বাংলা)",
    "mr-IN": "Marathi (मराठी)",
    "gu-IN": "Gujarati (ગુજરાતી)",
    "kn-IN": "Kannada (ಕನ್ನಡ)",
    "ml-IN": "Malayalam (മലയാളം)",
    "pa-IN": "Punjabi (ਪੰਜਾਬੀ)",
    "od-IN": "Odia (ଓଡ଼ିଆ)",
    "en-IN": "Indian English"
}

class SarvamClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = SARVAM_API_BASE

    def is_live(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    # -------------------------------------------------------------
    # 1. Speech-to-Text (Saaras)
    # -------------------------------------------------------------
    def speech_to_text(self, audio_bytes: bytes, language_code: str = "hi-IN", model: str = "saaras:v1") -> Dict[str, Any]:
        """Transcribes speech audio using Sarvam Saaras ASR."""
        if self.is_live():
            try:
                headers = {"api-subscription-key": self.api_key}
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data = {"language_code": language_code, "model": model}
                
                resp = requests.post(f"{self.base_url}/speech-to-text", headers=headers, files=files, data=data, timeout=30)
                if resp.status_code == 200:
                    res_json = resp.json()
                    res_json["mode"] = "live_sarvam_api"
                    return res_json
                else:
                    print(f"Sarvam STT API returned error {resp.status_code}: {resp.text}. Falling back to demo mode.")
            except Exception as ex:
                print(f"Sarvam STT connection error: {ex}. Falling back to demo mode.")

        # High-fidelity mock response
        return self._mock_speech_to_text(language_code)

    def _mock_speech_to_text(self, language_code: str) -> Dict[str, Any]:
        transcripts = {
            "hi-IN": "नमस्ते, सर्वम् एआई और लायब्निट्ज़ ५.० में आपका स्वागत है। आपका स्वर संकेत साफ़ और स्पष्ट है।",
            "sa-IN": "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात्। लायब्निट्ज़-सर्वम् सङ्केतसंसाधनम्।",
            "ta-IN": "வணக்கம், சர்வம் ஏஐ மற்றும் லீப்னிட்ஸ் 5.0 க்கு தங்களை வரவேற்கிறோம்.",
            "te-IN": "నమస్కారం, సర్వం ఏఐ మరియు లీబ్నిట్జ్ 5.0 కి స్వాగతం.",
            "bn-IN": "নমস্কার, সর্বম এআই এবং লিবনিজ ৫.০ এ আপনাকে স্বাগতম।",
            "mr-IN": "नमस्कार, सर्वम एआय आणि लायब्निट्झ ५.० मध्ये आपले स्वागत आहे.",
            "gu-IN": "નમસ્તે, સર્વમ એઆઈ અને લાઇબનિટ્ઝ ૫.૦ માં આપનું સ્વાગત છે.",
            "en-IN": "Welcome to Leibnitz 5.0 for Sarvam AI. Acoustic signal conditioned and transcribed with high fidelity."
        }
        text = transcripts.get(language_code, transcripts["hi-IN"])
        return {
            "transcript": text,
            "language_code": language_code,
            "confidence": 0.985,
            "mode": "simulation_demo",
            "provider": "Sarvam Saaras STT Engine"
        }

    # -------------------------------------------------------------
    # 2. Text-to-Speech (Bulbul)
    # -------------------------------------------------------------
    def text_to_speech(self, text: str, target_language_code: str = "hi-IN", speaker: str = "meera") -> Dict[str, Any]:
        """Synthesizes speech from text using Sarvam Bulbul TTS."""
        if self.is_live():
            try:
                headers = {
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": [text],
                    "target_language_code": target_language_code,
                    "speaker": speaker,
                    "model": "bulbul:v1"
                }
                resp = requests.post(f"{self.base_url}/text-to-speech", headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    res_json = resp.json()
                    res_json["mode"] = "live_sarvam_api"
                    return res_json
                else:
                    print(f"Sarvam TTS API returned error {resp.status_code}: {resp.text}. Falling back to demo mode.")
            except Exception as ex:
                print(f"Sarvam TTS connection error: {ex}. Falling back to demo mode.")

        # Realistic mock audio synthesis (generates valid audible WAV carrier modulated with speech envelope)
        return self._mock_text_to_speech(text, target_language_code)

    def _mock_text_to_speech(self, text: str, lang_code: str) -> Dict[str, Any]:
        sample_rate = 16000
        # Estimate duration based on syllable count
        duration = max(1.2, min(8.0, len(text) * 0.065))
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Synthesize harmonic vocal tract carrier with pitch modulation
        f0 = 210.0 if "female" in lang_code or True else 130.0
        pitch_mod = f0 + 15.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(pitch_mod) / sample_rate
        voice_wave = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
        
        # Envelope to simulate word cadences
        env = np.abs(np.sin(2 * np.pi * 3.2 * t)) ** 0.8
        speech_synth = (voice_wave * env * 0.6).astype(np.float32)
        
        # Convert to WAV bytes
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            int_data = (speech_synth * 32767).astype(np.int16)
            wav_file.writeframes(int_data.tobytes())
            
        b64_audio = base64.b64encode(byte_io.getvalue()).decode('utf-8')
        return {
            "audios": [b64_audio],
            "sample_rate": sample_rate,
            "mode": "simulation_demo",
            "provider": "Sarvam Bulbul TTS Engine",
            "duration_s": round(duration, 2)
        }

    # -------------------------------------------------------------
    # 3. Indic LLM / Signal Copilot (Sarvam-1 / Sarvam-2)
    # -------------------------------------------------------------
    def chat_completion(self, user_message: str, signal_context: Optional[Dict[str, Any]] = None, language: str = "hi") -> Dict[str, Any]:
        """Provides expert acoustic and DSP insights via Sarvam Indic LLM."""
        if self.is_live():
            try:
                headers = {
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json"
                }
                messages = [
                    {
                        "role": "system",
                        "content": "You are the Leibnitz 5.0 Indic Signal Copilot powered by Sarvam AI. "
                                   "You specialize in Indian language speech acoustic analysis, Vedic phonetics, "
                                   "telephony DSP filters, and signal feature extraction. "
                                   "Answer concisely and authoritatively in the requested language."
                    },
                    {
                        "role": "user",
                        "content": f"Signal Context: {json.dumps(signal_context or {})}\n\nUser Question: {user_message}"
                    }
                ]
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={"messages": messages, "model": "sarvam-2"},
                    timeout=30
                )
                if resp.status_code == 200:
                    res_json = resp.json()
                    res_json["mode"] = "live_sarvam_api"
                    return res_json
            except Exception as ex:
                print(f"Sarvam Chat connection error: {ex}. Falling back to demo mode.")

        return self._mock_chat_completion(user_message, signal_context, language)

    def _mock_chat_completion(self, user_message: str, signal_context: Optional[Dict[str, Any]], language: str) -> Dict[str, Any]:
        snr = (signal_context or {}).get("snr_db", 24.5)
        pitch = (signal_context or {}).get("mean_pitch_f0_hz", 185.0)
        
        responses = {
            "hi": f"लायब्निट्ज़ ५.० विश्लेषक: आपके ध्वनि संकेत का SNR {snr} dB है, जो सर्वम् सारस (Saaras) STT मॉडल के लिए उपयुक्त है। मूल आवृत्ति (Pitch F0) लगभग {pitch} Hz प्राप्त हुई है। यदि कॉल-सेंटर या टेलीफोनी शोर उपस्थित है, तो 'Telecom Voice Conditioner' (300-3400 Hz) का प्रयोग करने से प्रतिलेखन सटीकता में १५-२०% सुधार होगा।",
            "sa": f"लायब्निट्ज़-सर्वम् विमर्शनम्: अस्य ध्वनि-सङ्केतस्य SNR-मानं {snr} dB अस्ति। मूल-स्वरः (F0) {pitch} Hz परिमितः। वैदिक-उच्चारणे उदात्त-अनुदात्त-स्वरित-भेदानां स्पष्टीकरणाय 'Indic Phonetic & Pitch Analyzer' इत्यस्य परिणामः अनुकूलः वर्तते।",
            "en": f"Leibnitz-Sarvam Copilot Analysis: The processed audio exhibits an estimated SNR of {snr} dB and mean fundamental pitch (F0) of {pitch} Hz. This acoustic signature is well-calibrated for Sarvam Saaras STT ingestion. For high-noise telephony environments, applying the 300-3400 Hz bandpass conditioner ensures zero acoustic leakage outside human vocal cord formants."
        }
        chosen = responses.get(language, responses["hi"])
        return {
            "message": chosen,
            "model": "sarvam-2 (Indic DSP Specialist)",
            "mode": "simulation_demo",
            "provider": "Sarvam Foundation Models"
        }
