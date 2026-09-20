# -*- coding: utf-8 -*-
"""
Leibnitz 5.0 for Sarvam - High-Resolution Visualizer
===================================================
Renders multi-panel dark-themed analytical plots comparing original signals,
conditioned waveforms, Fourier frequency spectra, and time-frequency spectrograms.
"""

import os
import uuid
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

def generate_multi_panel_plot(
    original_signal: np.ndarray,
    processed_signal: np.ndarray,
    sample_rate: float,
    output_dir: str,
    original_filename: str = "signal",
    f0_info: dict = None
) -> str:
    """
    Generates a 4-panel analytical plot:
    1. Original Input Signal (Time Domain)
    2. Conditioned / Filtered Signal (Time Domain)
    3. Fourier Frequency Spectrum (FFT Magnitude with peak detection)
    4. Audio Spectrogram (STFT Time-Frequency Heatmap)
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"plot_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(output_dir, plot_filename)

    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), facecolor='#060913')

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor('#0c1222')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.grid(True, color='#1e293b', alpha=0.6, linestyle='--')

    # Panel 1: Original Signal
    orig_n = len(original_signal)
    t_orig = np.arange(orig_n) / float(sample_rate)
    step_orig = max(1, orig_n // 2000)
    ax1.plot(t_orig[::step_orig], original_signal[::step_orig], color='#00d4ff', linewidth=1.2, alpha=0.9)
    ax1.set_title(f"1. Ingested Signal: {os.path.basename(original_filename)}", color='#ffffff', fontsize=10, pad=8, weight='bold')
    ax1.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=8)
    ax1.set_ylabel("Amplitude", color='#94a3b8', fontsize=8)

    # Panel 2: Processed / Conditioned Signal
    proc_n = len(processed_signal)
    t_proc = np.arange(proc_n) / float(sample_rate)
    step_proc = max(1, proc_n // 2000)
    ax2.plot(t_proc[::step_proc], processed_signal[::step_proc], color='#ff7700', linewidth=1.2, alpha=0.9)
    ax2.set_title("2. Leibnitz Conditioned Output (VAD & Filter)", color='#ffffff', fontsize=10, pad=8, weight='bold')
    ax2.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=8)
    ax2.set_ylabel("Amplitude", color='#94a3b8', fontsize=8)

    # Panel 3: FFT Frequency Spectrum
    n_fft = min(orig_n, 8192)
    sig_slice = original_signal[:n_fft] * np.hanning(n_fft)
    fft_vals = np.abs(np.fft.rfft(sig_slice)) / (n_fft / 2)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    max_freq = sample_rate / 2.0
    disp_limit = min(max_freq, 4000.0) if max_freq > 4000 else max_freq
    mask = freqs <= disp_limit

    ax3.plot(freqs[mask], fft_vals[mask], color='#00ff88', linewidth=1.4)
    ax3.fill_between(freqs[mask], fft_vals[mask], color='#00ff88', alpha=0.15)
    
    peak_idx = np.argmax(fft_vals[mask])
    peak_freq = freqs[mask][peak_idx]
    peak_mag = fft_vals[mask][peak_idx]
    ax3.annotate(
        f"Peak: {peak_freq:.1f} Hz\n({peak_mag:.3f})",
        xy=(peak_freq, peak_mag),
        xytext=(peak_freq + (disp_limit * 0.08), max(peak_mag * 0.85, 0.05)),
        arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=1, headwidth=5),
        color='#ffd700', fontsize=8, weight='bold'
    )
    ax3.set_title("3. Fourier Magnitude Spectrum (FFT)", color='#ffffff', fontsize=10, pad=8, weight='bold')
    ax3.set_xlabel("Frequency (Hz)", color='#94a3b8', fontsize=8)
    ax3.set_ylabel("Magnitude", color='#94a3b8', fontsize=8)
    ax3.set_xlim(0, disp_limit)

    # Panel 4: Spectrogram / STFT
    try:
        nperseg = min(256, orig_n // 4) if orig_n >= 32 else 16
        if nperseg >= 8:
            f, t_spec, Sxx = sp_signal.spectrogram(original_signal, fs=sample_rate, nperseg=nperseg)
            Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
            im = ax4.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='inferno')
            ax4.set_ylim(0, disp_limit)
            ax4.set_title("4. Time-Frequency Spectrogram (STFT)", color='#ffffff', fontsize=10, pad=8, weight='bold')
            ax4.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=8)
            ax4.set_ylabel("Frequency (Hz)", color='#94a3b8', fontsize=8)
            cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='#94a3b8', labelsize=7)
            cbar.set_label("Power (dB)", color='#94a3b8', fontsize=8)
        else:
            raise ValueError("Too few samples")
    except Exception:
        ax4.plot(np.angle(np.fft.rfft(sig_slice))[:200], color='#f59e0b', linewidth=1)
        ax4.set_title("4. Phase Spectrum", color='#ffffff', fontsize=10, pad=8)

    fig.suptitle(
        f"Leibnitz 5.0 x Sarvam AI -- Signal Analysis Studio ({os.path.basename(original_filename)})",
        color='#ff7700', fontsize=12, weight='bold', y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=120, facecolor='#060913')
    plt.close(fig)

    return plot_filename
