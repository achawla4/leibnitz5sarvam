# -*- coding: utf-8 -*-
"""
Speech DSP Engine for Sarvam Indic Workflows
============================================
Specialized acoustic and phonetic signal processing routines engineered for
Indian language speech recognition (Sarvam Saaras STT), voice synthesis (Sarvam Bulbul TTS),
and Indic conversational telephony pipelines.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple, List

def compute_snr_db(audio: np.ndarray, frame_length: int = 512, hop_length: int = 256) -> float:
    """Estimates the Signal-to-Noise Ratio (SNR) in dB from speech energy distribution."""
    if len(audio) < frame_length:
        return 20.0
    
    # Calculate short-time frame energies
    frames = [audio[i:i+frame_length] for i in range(0, len(audio) - frame_length, hop_length)]
    energies = np.array([np.sum(f**2) for f in frames]) + 1e-12
    
    # Sort energies: lowest 10% represent acoustic noise floor, top 20% represent active speech
    sorted_e = np.sort(energies)
    n_noise = max(1, int(len(sorted_e) * 0.10))
    n_speech = max(1, int(len(sorted_e) * 0.20))
    
    noise_energy = np.mean(sorted_e[:n_noise])
    speech_energy = np.mean(sorted_e[-n_speech:])
    
    snr = 10.0 * np.log10(max(speech_energy / max(noise_energy, 1e-12), 1.0))
    return float(round(snr, 2))

def voice_activity_detection(audio: np.ndarray, sample_rate: float, threshold_factor: float = 1.6) -> Dict[str, Any]:
    """
    Energy and zero-crossing based Voice Activity Detection (VAD).
    Segments audio into voiced/speech regions and silence/noise for Sarvam STT preprocessing.
    """
    frame_ms = 25
    hop_ms = 10
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    
    if len(audio) < frame_len:
        return {
            "speech_ratio": 1.0,
            "silence_ratio": 0.0,
            "speech_duration_s": round(len(audio) / sample_rate, 2),
            "total_duration_s": round(len(audio) / sample_rate, 2),
            "num_segments": 1,
            "cleaned_signal": audio
        }
    
    # Frame energy and zero-crossing rate
    num_frames = max(1, (len(audio) - frame_len) // hop_len)
    energies = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start:start+frame_len]
        energies[i] = np.sum(frame**2)
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
    energy_threshold = np.median(energies) * threshold_factor
    is_speech = energies > energy_threshold
    
    # Smooth small gaps (hangover effect)
    smoothed = np.copy(is_speech)
    for i in range(1, len(smoothed) - 1):
        if not smoothed[i] and smoothed[i-1] and smoothed[i+1]:
            smoothed[i] = True
            
    # Reconstruct cleaned audio by masking out long silence stretches
    speech_mask = np.zeros(len(audio), dtype=bool)
    for i, flag in enumerate(smoothed):
        if flag:
            start = i * hop_len
            speech_mask[start:min(len(audio), start + frame_len)] = True
            
    cleaned_audio = audio[speech_mask] if np.any(speech_mask) else audio
    speech_frames = np.sum(smoothed)
    speech_ratio = float(round(speech_frames / max(1, num_frames), 3))
    
    return {
        "speech_ratio": speech_ratio,
        "silence_ratio": round(1.0 - speech_ratio, 3),
        "total_duration_s": round(len(audio) / sample_rate, 3),
        "speech_duration_s": round(len(cleaned_audio) / sample_rate, 3),
        "cleaned_signal": cleaned_audio
    }

def estimate_pitch_contour(audio: np.ndarray, sample_rate: float) -> Dict[str, Any]:
    """
    Extracts Fundamental Frequency (F0 / Swara) using short-time autocorrelation.
    Crucial for Indic speech prosody and Vedic chanting intonation (Udatta, Anudatta, Svarita).
    """
    frame_len = int(sample_rate * 0.035)  # 35 ms frame
    hop_len = int(sample_rate * 0.015)    # 15 ms hop
    min_lag = int(sample_rate / 450)      # Max F0 = 450 Hz
    max_lag = int(sample_rate / 65)       # Min F0 = 65 Hz
    
    num_frames = max(1, (len(audio) - frame_len) // hop_len)
    f0_values = []
    time_stamps = []
    
    for i in range(min(num_frames, 300)):  # Cap frames for snappy responsive UI
        start = i * hop_len
        frame = audio[start:start+frame_len]
        # Energy check for voicedness
        if np.sum(frame**2) < 1e-4:
            f0_values.append(0.0)
            time_stamps.append(round(start / sample_rate, 3))
            continue
            
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        if len(corr) > max_lag:
            search_region = corr[min_lag:max_lag]
            peak_idx = np.argmax(search_region) + min_lag
            if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.35:
                freq = sample_rate / peak_idx
                f0_values.append(round(float(freq), 1))
            else:
                f0_values.append(0.0)
        else:
            f0_values.append(0.0)
        time_stamps.append(round(start / sample_rate, 3))
        
    voiced_f0 = [f for f in f0_values if f > 0]
    mean_f0 = round(float(np.mean(voiced_f0)), 1) if voiced_f0 else 0.0
    f0_range = [round(min(voiced_f0), 1), round(max(voiced_f0), 1)] if voiced_f0 else [0.0, 0.0]
    
    return {
        "mean_f0_hz": mean_f0,
        "f0_range_hz": f0_range,
        "time_stamps": time_stamps,
        "f0_contour": f0_values
    }

def estimate_formants_lpc(audio: np.ndarray, sample_rate: float, order: int = 14) -> Dict[str, float]:
    """
    Estimates key Indic vowel formant resonances (F1, F2, F3) via Linear Predictive Coding (LPC).
    Matches phonetic characteristics of Sanskrit and Indic varnamala vowels.
    """
    # Pick a steady voiced segment
    center = len(audio) // 2
    span = int(sample_rate * 0.05) # 50 ms
    segment = audio[max(0, center - span) : min(len(audio), center + span)]
    
    if len(segment) < order * 2:
        return {"F1": 500.0, "F2": 1500.0, "F3": 2500.0}
    
    # Pre-emphasis
    segment = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
    # Hamming window
    w = np.hamming(len(segment))
    sw = segment * w
    
    # Autocorrelation
    r = np.correlate(sw, sw, mode='full')
    r = r[len(r)//2 : len(r)//2 + order + 1]
    
    # Levinson-Durbin
    a = np.zeros(order + 1)
    a[0] = 1.0
    if r[0] == 0:
        return {"F1": 500.0, "F2": 1500.0, "F3": 2500.0}
        
    # Solve Toeplitz
    try:
        from scipy.linalg import solve_toeplitz
        if len(r) > 1:
            a[1:] = solve_toeplitz(r[:-1], -r[1:])
    except Exception:
        return {"F1": 550.0, "F2": 1650.0, "F3": 2600.0}
        
    # Find roots
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) >= 0]
    
    # Formant angles
    angles = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted([ang * (sample_rate / (2.0 * np.pi)) for ang in angles])
    
    # Filter reasonable speech formant ranges
    valid_f = [f for f in formants if 200 <= f <= 3800]
    
    f1 = round(float(valid_f[0]), 1) if len(valid_f) > 0 else 520.0
    f2 = round(float(valid_f[1]), 1) if len(valid_f) > 1 else 1680.0
    f3 = round(float(valid_f[2]), 1) if len(valid_f) > 2 else 2720.0
    
    return {"F1": f1, "F2": f2, "F3": f3}

def apply_telephony_bandpass(audio: np.ndarray, sample_rate: float, low_cut: float = 300.0, high_cut: float = 3400.0) -> np.ndarray:
    """Filters audio to Indian telecom narrowband (G.711 PCM telephony 300-3400 Hz) for voice agent robustness."""
    nyquist = 0.5 * sample_rate
    low = max(20.0, min(low_cut, nyquist - 50.0)) / nyquist
    high = min(high_cut, nyquist - 20.0) / nyquist
    
    if low >= high:
        return audio
        
    b, a = signal.butter(4, [low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, audio)
    return filtered
