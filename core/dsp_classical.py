# -*- coding: utf-8 -*-
"""
Classical & Mathematical DSP Blocks for Leibnitz 5.0 (Sarvam Edition)
====================================================================
Integrates classical Fourier analysis, spectral decomposition, and filters
into the modular LeibnitzCoreRegistry.
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any

from .adapter import registry, LeibnitzBlock
from .dsp_speech import (
    voice_activity_detection,
    compute_snr_db,
    estimate_pitch_contour,
    estimate_formants_lpc,
    apply_telephony_bandpass
)

def block_vad_cleaner(audio: np.ndarray, sample_rate: float, params: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(params.get("threshold", 1.6))
    res = voice_activity_detection(audio, sample_rate, threshold_factor=threshold)
    cleaned = res["cleaned_signal"]
    
    return {
        "output_signal": cleaned,
        "metrics": {
            "speech_ratio": res["speech_ratio"],
            "silence_removed_s": round(res["total_duration_s"] - res["speech_duration_s"], 2),
            "original_duration_s": res["total_duration_s"],
            "cleaned_duration_s": res["speech_duration_s"]
        }
    }

def block_telephony_filter(audio: np.ndarray, sample_rate: float, params: Dict[str, Any]) -> Dict[str, Any]:
    low_cut = float(params.get("low_cut", 300.0))
    high_cut = float(params.get("high_cut", 3400.0))
    filtered = apply_telephony_bandpass(audio, sample_rate, low_cut, high_cut)
    snr_val = compute_snr_db(filtered)
    
    return {
        "output_signal": filtered,
        "metrics": {
            "bandwidth": f"{int(low_cut)}-{int(high_cut)} Hz",
            "snr_db": snr_val,
            "filter_type": "4th-order Butterworth Telephony Bandpass"
        }
    }

def block_fft_spectrum(audio: np.ndarray, sample_rate: float, params: Dict[str, Any]) -> Dict[str, Any]:
    n = len(audio)
    if n == 0:
        return {"output_signal": audio, "metrics": {}}
        
    windowed = audio * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mag = np.abs(fft_vals) / (n / 2)
    
    peak_idx = np.argmax(mag)
    dominant_freq = round(float(freqs[peak_idx]), 2)
    
    # Downsample points for frontend chart payload (max 250 points)
    step = max(1, len(freqs) // 250)
    chart_freqs = [round(float(f), 1) for f in freqs[::step]]
    chart_mags = [round(float(m), 4) for m in mag[::step]]
    
    return {
        "output_signal": audio,
        "metrics": {
            "dominant_frequency_hz": dominant_freq,
            "spectral_energy": round(float(np.sum(mag**2)), 3),
            "nyquist_hz": sample_rate / 2
        },
        "visuals": {
            "chart_type": "line",
            "x": chart_freqs,
            "y": chart_mags,
            "x_label": "Frequency (Hz)",
            "y_label": "Magnitude"
        }
    }

def block_indic_phonetics(audio: np.ndarray, sample_rate: float, params: Dict[str, Any]) -> Dict[str, Any]:
    pitch_res = estimate_pitch_contour(audio, sample_rate)
    formants = estimate_formants_lpc(audio, sample_rate)
    snr = compute_snr_db(audio)
    
    return {
        "output_signal": audio,
        "metrics": {
            "mean_pitch_f0_hz": pitch_res["mean_f0_hz"],
            "pitch_range_hz": pitch_res["f0_range_hz"],
            "formant_f1_hz": formants["F1"],
            "formant_f2_hz": formants["F2"],
            "formant_f3_hz": formants["F3"],
            "snr_db": snr
        },
        "visuals": {
            "time_axis": pitch_res["time_stamps"],
            "f0_contour": pitch_res["f0_contour"]
        }
    }

def register_default_blocks():
    """Register core blocks for Leibnitz 5.0 (Sarvam Edition)."""
    registry.register(LeibnitzBlock(
        block_id="vad_cleaner",
        name="Indic VAD & Silence Masker",
        category="speech_pre",
        version="5.0",
        handler=block_vad_cleaner,
        description="Extracts clean speech segments and removes ambient dead-air for Sarvam Saaras STT."
    ))
    
    registry.register(LeibnitzBlock(
        block_id="telephony_bandpass",
        name="Telecom Voice Conditioner",
        category="speech_filter",
        version="5.0",
        handler=block_telephony_filter,
        description="Conditions audio to standard 300-3400 Hz Indian telecom voice bandpass."
    ))

    registry.register(LeibnitzBlock(
        block_id="fft_spectrum",
        name="Leibnitz Fourier Spectral Engine",
        category="spectral",
        version="5.0",
        handler=block_fft_spectrum,
        description="Computes high-resolution frequency magnitude distribution and dominant resonant modes."
    ))

    registry.register(LeibnitzBlock(
        block_id="indic_phonetics",
        name="Indic Phonetic & Pitch Analyzer",
        category="acoustic",
        version="5.0",
        handler=block_indic_phonetics,
        description="Calculates F0 pitch contour, intonation nuances, and F1-F3 formant frequencies for Indic languages."
    ))

# Auto-register upon module load
register_default_blocks()
