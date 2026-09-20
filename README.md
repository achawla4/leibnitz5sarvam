# Leibnitz 5.0 for Sarvam (Indic Audio & Signal AI)

[![Leibnitz](https://img.shields.io/badge/Leibnitz-5.0--DSP-00d4ff)](https://github.com/)
[![Sarvam AI](https://img.shields.io/badge/Sarvam%20AI-Saaras%20%7C%20Bulbul%20%7C%20LLM-ff7700)](https://sarvam.ai)
[![Upgradable](https://img.shields.io/badge/Architecture-v8%20%26%20v10%20Upgradable-00ff88)](#modular-upgradability-to-leibnitz-80--100)
[![Render Deploy](https://img.shields.io/badge/Deploy%20on-Render-46e3b7)](https://render.com)

**Leibnitz 5.0 for Sarvam** is an intelligent acoustic and signal processing suite tailored specifically for **Sarvam AI**'s users, voice developers, and Indic speech workflows. It bridges classical and quantum mathematical signal processing with Sarvam's state-of-the-art Indic language foundation models.

---

## Key Capabilities

1. **Acoustic Conditioning for Sarvam Saaras (STT)**:
   - **Energy-based Voice Activity Detection (VAD)**: Eliminates leading/trailing dead-air and extracts voiced phonemes.
   - **Indian Telephony Voice Bandpass**: 300–3400 Hz 4th-order Butterworth filter to condition noisy call center / 8 kHz audio.
   - **SNR Estimation**: Live Signal-to-Noise Ratio (dB) calculation to verify acoustic suitability prior to ASR ingestion.
2. **Prosody & Intonation Analysis for Sarvam Bulbul (TTS)**:
   - **Fundamental Pitch ($F_0$) Extraction**: Autocorrelation-based pitch contour tracking for Indic tonal nuances and Vedic chanting swaras (*Udatta*, *Anudatta*, *Svarita*).
   - **Formant Resonance Estimation ($F_1, F_2, F_3$)**: Linear Predictive Coding (LPC) formant tracking for Indic vowels.
3. **Sarvam Indic DSP Copilot**:
   - Conversational AI assistant powered by Sarvam Indic LLMs (Sarvam-1 / Sarvam-2) diagnosing spectral anomalies and recommending filter pipelines in Hindi, Sanskrit, or English.
4. **Dual-Mode Execution**:
   - **Live Mode**: When `SARVAM_API_KEY` is provided, connects directly to production endpoints at `https://api.sarvam.ai`.
   - **High-Fidelity Simulation Mode**: Fully functional offline demo out of the box with realistic transcripts, synthesized audio, and intelligent responses.

---

## Modular Upgradability to Leibnitz 8.0 / 10.0+

The system is decoupled using the **`LeibnitzCoreRegistry`** pattern (`core/adapter.py`).

When **Leibnitz 8.0** or **Leibnitz 10.0** releases new blocks (such as Quantum SFT, Neural Diffusion Denoising, or Multi-channel Beamforming), you can register them in one line using the `@leibnitz_extension` decorator without modifying any web routing, frontend templates, or Sarvam client contracts:

```python
from core.upgrade_guide import leibnitz_extension

@leibnitz_extension(
    block_id="quantum_diffusion_denoise",
    name="Quantum Diffusion Denoiser",
    category="quantum_dsp",
    version="8.0",
    description="Neural diffusion denoising from Leibnitz 8.0"
)
def block_quantum_denoise(signal, sample_rate, params):
    # Your Leibnitz 8.0 implementation
    return {
        "output_signal": cleaned_signal,
        "metrics": {"diffusion_steps": 20, "snr_boost_db": 14.2}
    }
```

The Sarvam Studio UI and `/api/blocks` endpoint dynamically discover and expose the new blocks immediately.

---

## Project Structure

```
qLeibnitzSarvam/
├── app.py                     # Flask application server & REST endpoints
├── core/
│   ├── adapter.py             # Versioned LeibnitzCoreRegistry & execution engine
│   ├── dsp_speech.py          # VAD, pitch F0, LPC formants, telephony filter, SNR
│   ├── dsp_classical.py       # FFT spectrum, filter blocks, core registration
│   └── upgrade_guide.py       # Forward-compatibility decorator for v8/v10+
├── sarvam/
│   └── client.py              # Sarvam AI API client (Saaras, Bulbul, LLM) + mock engine
├── templates/
│   └── index.html             # Cyber-Indic responsive interactive studio
├── static/
│   ├── css/sarvam.css         # Dark neon theme with Saffron & Cyan accents
│   └── js/main.js             # Canvas waveforms, mic recorder, pipeline executor
├── uploads/                   # Temporary audio upload storage
├── processed/                 # Processed audio output storage
├── requirements.txt           # Production dependencies
├── Procfile                   # Web process definition for Render (gunicorn app:app)
├── render.yaml                # Render Infrastructure as Code Blueprint
└── README.md                  # Documentation & Deployment Instructions
```

---

## Quickstart (Local Testing)

```bash
# 1. Navigate to the folder
cd qLeibnitzSarvam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: configure your live Sarvam API Key
# export SARVAM_API_KEY="your-sarvam-key-here"

# 4. Start the server
python app.py
```

Open [http://localhost:5005](http://localhost:5005) in your browser.

---

## Git Repository Setup Instructions

To push this project to its own dedicated GitHub repository:

```bash
# 1. Navigate to the qLeibnitzSarvam subfolder
cd c:\Users\acer\Documents\qLeibnitz2\qLeibnitzSarvam

# 2. Initialize a clean Git repository
git init

# 3. Stage all files
git add .

# 4. Create the initial commit
git commit -m "feat: initial commit for Leibnitz 5.0 for Sarvam"

# 5. Link your new GitHub repository (replace with your repository URL)
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<NEW_REPO_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

---

## Render Deployment Instructions

Follow these steps to deploy **Leibnitz 5.0 for Sarvam** on **[Render](https://render.com)** as a production Web Service:

### Method A: One-Click / Web Service Deployment via Render Dashboard

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Web Service**.
3. Connect the GitHub repository you created above (`<NEW_REPO_NAME>`).
4. Configure the service settings:
   - **Name**: `leibnitz-sarvam` (or your preferred name)
   - **Region**: `Oregon (US West)` or `Singapore (Southeast Asia)`
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free` (or Starter)
5. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.10.12`
   - `SECRET_KEY`: *(click Generate for a secure random string)*
   - `SARVAM_API_KEY`: *(optional: paste your live Sarvam API key, or leave blank to run in simulation/demo mode)*
6. Click **Create Web Service**.
7. Render will build the image, install dependencies, and launch Gunicorn. Your app will be live at `https://<your-service-name>.onrender.com`.

### Method B: Blueprint Deployment (using `render.yaml`)

1. In Render, select **Blueprints** from the top navigation.
2. Connect your Git repository containing `render.yaml`.
3. Render will automatically configure the build command, start command, and environment variables declared in `render.yaml`.
4. Click **Apply**.
