/**
 * Leibnitz 5.0 for Sarvam - Client Controller
 * Handles audio recording, canvas waveform visualization, DSP execution,
 * and Sarvam AI Indic API interactions.
 */

let currentAudioFilename = null;
let currentProcessedFilename = null;
let activeSampleRate = 16000;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let activeSignalContext = {
    snr_db: 24.5,
    mean_pitch_f0_hz: 185.0
};

// Canvas references
const rawCanvas = document.getElementById('rawWaveformCanvas');
const procCanvas = document.getElementById('procWaveformCanvas');
const rawCtx = rawCanvas ? rawCanvas.getContext('2d') : null;
const procCtx = procCanvas ? procCanvas.getContext('2d') : null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initRecording();
    loadPreset('sanskrit_chant');
});

// Tab Switcher
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
}

// Canvas Waveform Drawing
function drawWaveform(canvas, ctx, waveformData, strokeColor = '#00d4ff') {
    if (!canvas || !ctx || !waveformData || waveformData.length === 0) return;
    
    // Support HiDPI
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement.clientWidth;
    const height = 110;
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    ctx.clearRect(0, 0, width, height);
    
    // Center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    
    // Waveform line
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    
    const sliceWidth = width / waveformData.length;
    let x = 0;
    
    for (let i = 0; i < waveformData.length; i++) {
        const val = waveformData[i];
        const y = (height / 2) - (val * (height * 0.42));
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    ctx.stroke();
}

// Load Presets
async function loadPreset(presetId) {
    showLoading(true);
    try {
        const resp = await fetch(`/api/load-preset/${presetId}`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;
            
            drawWaveform(rawCanvas, rawCtx, data.waveform_preview, '#00d4ff');
            document.getElementById('rawAudioSource').src = data.audio_url;
            document.getElementById('rawAudioPlayer').load();
            
            document.getElementById('metaSampleRate').innerText = `${Math.round(data.sample_rate)} Hz`;
            document.getElementById('metaDuration').innerText = `${data.duration_s} s`;
            
            // Auto run pipeline
            runPipeline();
        }
    } catch (e) {
        console.error('Error loading preset:', e);
    } finally {
        showLoading(false);
    }
}

// File Upload
async function handleFileUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;
            
            drawWaveform(rawCanvas, rawCtx, data.waveform_preview, '#00d4ff');
            document.getElementById('rawAudioSource').src = data.audio_url;
            document.getElementById('rawAudioPlayer').load();
            
            document.getElementById('metaSampleRate').innerText = `${Math.round(data.sample_rate)} Hz`;
            document.getElementById('metaDuration').innerText = `${data.duration_s} s`;
            
            runPipeline();
        } else {
            alert(data.error || 'Upload failed');
        }
    } catch (e) {
        console.error('Upload error:', e);
        alert('Upload failed. Check file format.');
    } finally {
        showLoading(false);
    }
}

// In-Browser Audio Recording
function initRecording() {
    const recBtn = document.getElementById('recordBtn');
    if (!recBtn) return;
    
    recBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const file = new File([audioBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    showLoading(true);
                    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                    const data = await resp.json();
                    if (data.success) {
                        currentAudioFilename = data.filename;
                        activeSampleRate = data.sample_rate;
                        drawWaveform(rawCanvas, rawCtx, data.waveform_preview, '#00d4ff');
                        document.getElementById('rawAudioSource').src = data.audio_url;
                        document.getElementById('rawAudioPlayer').load();
                        runPipeline();
                    }
                    showLoading(false);
                };
                
                mediaRecorder.start();
                isRecording = true;
                recBtn.classList.add('recording');
                recBtn.innerHTML = '🛑 ध्वनि-मुद्रणं स्थग्यताम् (Stop Recording)';
            } catch (err) {
                alert('Microphone access permission denied or unavailable.');
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            recBtn.classList.remove('recording');
            recBtn.innerHTML = '🎙️ स्वरं मुद्रयतु (Record Voice)';
        }
    });
}

// Run Leibnitz Modular DSP Pipeline
async function runPipeline() {
    if (!currentAudioFilename) return;
    
    // Check which blocks are selected
    const selectedBlocks = [];
    if (document.getElementById('chk_vad').checked) {
        selectedBlocks.push({ id: 'vad_cleaner', params: { threshold: 1.5 } });
    }
    if (document.getElementById('chk_telephony').checked) {
        selectedBlocks.push({ id: 'telephony_bandpass', params: { low_cut: 300, high_cut: 3400 } });
    }
    if (document.getElementById('chk_phonetics').checked) {
        selectedBlocks.push({ id: 'indic_phonetics', params: {} });
    }
    if (document.getElementById('chk_fft').checked) {
        selectedBlocks.push({ id: 'fft_spectrum', params: {} });
    }
    
    showLoading(true);
    try {
        const resp = await fetch('/api/process-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: currentAudioFilename,
                pipeline: selectedBlocks
            })
        });
        const data = await resp.json();
        if (data.success) {
            currentProcessedFilename = data.processed_filename;
            
            drawWaveform(procCanvas, procCtx, data.processed_waveform, '#ff7700');
            document.getElementById('procAudioSource').src = data.audio_url;
            document.getElementById('procAudioPlayer').load();
            
            // Extract and update metrics
            updateMetrics(data.stages);
            
            // Auto update STT button label
            document.getElementById('btnRunSTT').innerText = `🚀 Transcribe via Sarvam Saaras (${document.getElementById('sttLanguage').value})`;
        }
    } catch (e) {
        console.error('Pipeline error:', e);
    } finally {
        showLoading(false);
    }
}

function updateMetrics(stages) {
    stages.forEach(stage => {
        const metrics = stage.metrics || {};
        if (metrics.snr_db !== undefined) {
            document.getElementById('metricSNR').innerText = `${metrics.snr_db} dB`;
            activeSignalContext.snr_db = metrics.snr_db;
        }
        if (metrics.mean_pitch_f0_hz !== undefined) {
            document.getElementById('metricPitch').innerText = `${metrics.mean_pitch_f0_hz} Hz`;
            activeSignalContext.mean_pitch_f0_hz = metrics.mean_pitch_f0_hz;
        }
        if (metrics.formant_f1_hz !== undefined) {
            document.getElementById('metricFormant').innerText = `F1: ${metrics.formant_f1_hz} Hz | F2: ${metrics.formant_f2_hz} Hz`;
        }
        if (metrics.speech_ratio !== undefined) {
            document.getElementById('metricSpeechRatio').innerText = `${Math.round(metrics.speech_ratio * 100)}%`;
        }
    });
}

// -------------------------------------------------------------
// Sarvam Saaras STT
// -------------------------------------------------------------
async function runSarvamSTT() {
    const fileToUse = currentProcessedFilename || currentAudioFilename;
    if (!fileToUse) {
        alert('Please record or select an audio trace first.');
        return;
    }
    
    const lang = document.getElementById('sttLanguage').value;
    const transcriptCard = document.getElementById('sttTranscript');
    transcriptCard.innerHTML = '<span style="color:var(--text-muted);">🔄 Transcribing with Sarvam Saaras Indic Engine...</span>';
    
    try {
        const resp = await fetch('/api/sarvam/stt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: fileToUse,
                language_code: lang
            })
        });
        const data = await resp.json();
        
        transcriptCard.innerHTML = `<strong>${data.transcript || 'No transcript returned'}</strong>`;
        document.getElementById('sttMeta').innerHTML = `
            <span>Provider: <strong>${data.provider || 'Sarvam AI'}</strong></span>
            <span>Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></span>
            <span>Mode: <strong>${data.mode}</strong></span>
        `;
    } catch (e) {
        transcriptCard.innerText = 'Error calling Sarvam STT service.';
    }
}

// -------------------------------------------------------------
// Sarvam Bulbul TTS
// -------------------------------------------------------------
async function runSarvamTTS() {
    const text = document.getElementById('ttsInputText').value.trim();
    if (!text) {
        alert('Please enter text to synthesize.');
        return;
    }
    
    const lang = document.getElementById('ttsLanguage').value;
    const speaker = document.getElementById('ttsSpeaker').value;
    const btn = document.getElementById('btnRunTTS');
    btn.innerText = '🔄 Synthesizing with Bulbul...';
    btn.disabled = true;
    
    try {
        const resp = await fetch('/api/sarvam/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                target_language_code: lang,
                speaker: speaker
            })
        });
        const data = await resp.json();
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;
            
            drawWaveform(rawCanvas, rawCtx, data.waveform_preview, '#00d4ff');
            document.getElementById('rawAudioSource').src = data.audio_url;
            document.getElementById('rawAudioPlayer').load();
            
            document.getElementById('metaSampleRate').innerText = `${Math.round(data.sample_rate)} Hz`;
            document.getElementById('metaDuration').innerText = `${data.duration_s} s`;
            
            // Pass straight through Leibnitz DSP
            runPipeline();
        } else {
            alert(data.error || 'TTS Synthesis failed');
        }
    } catch (e) {
        alert('Failed to connect to Sarvam TTS.');
    } finally {
        btn.innerText = '🔊 Synthesize & Send to Leibnitz DSP';
        btn.disabled = false;
    }
}

// -------------------------------------------------------------
// Sarvam Indic LLM Copilot Chat
// -------------------------------------------------------------
async function sendCopilotMessage() {
    const input = document.getElementById('copilotInput');
    const msg = input.value.trim();
    if (!msg) return;
    
    const chatWin = document.getElementById('copilotChatWindow');
    
    // Append user message
    chatWin.innerHTML += `<div class="chat-bubble user"><strong>You:</strong> ${msg}</div>`;
    input.value = '';
    chatWin.scrollTop = chatWin.scrollHeight;
    
    const lang = document.getElementById('copilotLang').value;
    
    try {
        const resp = await fetch('/api/sarvam/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                signal_context: activeSignalContext,
                language: lang
            })
        });
        const data = await resp.json();
        
        chatWin.innerHTML += `
            <div class="chat-bubble bot">
                <div style="font-size:0.75rem; color:var(--sarvam-amber); margin-bottom:0.25rem;">
                    🤖 ${data.provider} (${data.model})
                </div>
                ${data.message}
            </div>
        `;
        chatWin.scrollTop = chatWin.scrollHeight;
    } catch (e) {
        chatWin.innerHTML += `<div class="chat-bubble bot" style="border-color:#ef4444;">Error reaching Sarvam LLM.</div>`;
    }
}

function showLoading(show) {
    const loader = document.getElementById('globalLoader');
    if (loader) loader.style.display = show ? 'flex' : 'none';
}
