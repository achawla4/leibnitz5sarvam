/**
 * Leibnitz 5.0 for Sarvam - Client Controller
 * Matches animations, interactions, and workflow of https://leibnitz5.onrender.com/ (in English)
 */

let currentAudioFilename = null;
let currentProcessedFilename = null;
let activeSampleRate = 1000.0;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

let activeSignalContext = {
    filename: "sinusoidal_12Hz.csv",
    sample_rate: 1000.0,
    duration_s: 1.0,
    dominant_freq_hz: 12.0,
    snr_db: 24.5
};

document.addEventListener('DOMContentLoaded', () => {
    initHeroSignal();
    initFFTViz();
    initFilterViz();
    initDropZone();
    initTabs();
    initMicrophone();
    
    // Load default benchmark preset
    loadDemoSignal('sinusoid_12hz');
});

// ===================== HERO SIGNAL CANVAS ANIMATION =====================
function initHeroSignal() {
    const canvas = document.getElementById('heroSignal');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    let time = 0;
    const frequency = 0.02;
    const amplitude = 60;

    function drawGrid() {
        ctx.strokeStyle = 'rgba(26, 40, 71, 0.5)';
        ctx.lineWidth = 1;
        const gridSize = 30;

        for (let x = 0; x < canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }

        for (let y = 0; y < canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }

    function animate() {
        ctx.fillStyle = '#0f1535';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        drawGrid();

        const centerY = canvas.height / 2;

        // Draw primary signal (signal green)
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let x = 0; x < canvas.width; x++) {
            const y = centerY - amplitude * Math.sin((x * frequency + time) * 0.1) * Math.cos((x * frequency + time) * 0.05);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw secondary signal (signal blue with phase shift)
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        for (let x = 0; x < canvas.width; x++) {
            const y = centerY - (amplitude * 0.6) * Math.sin((x * frequency + time) * 0.1 + Math.PI / 3);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Draw data points
        ctx.fillStyle = '#ff3344';
        for (let x = 0; x < canvas.width; x += 40) {
            const y = centerY - amplitude * Math.sin((x * frequency + time) * 0.1) * Math.cos((x * frequency + time) * 0.05);
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        }

        time += 1;
        requestAnimationFrame(animate);
    }

    animate();
}

// ===================== SPECS MINI CANVASES =====================
function initFFTViz() {
    const canvas = document.getElementById('fftViz');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    let time = 0;

    function animate() {
        ctx.fillStyle = 'rgba(5, 8, 18, 0.4)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / 32;
        const centerY = canvas.height - 20;

        for (let i = 0; i < 32; i++) {
            const freq = i / 32;
            const height = canvas.height * 0.7 * Math.abs(Math.sin(freq * 5 + time * 0.05)) * (1 - freq * 0.5);
            ctx.fillStyle = i % 3 === 0 ? '#ff3344' : '#00ff88';
            ctx.fillRect(i * barWidth + 2, centerY - height, barWidth - 4, height);
        }

        time++;
        requestAnimationFrame(animate);
    }

    animate();
}

function initFilterViz() {
    const canvas = document.getElementById('filterViz');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    ctx.fillStyle = 'rgba(5, 8, 18, 0.9)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw axis
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Draw filter response
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
        const freq = (x / canvas.width) * 5;
        const response = Math.exp(-(freq - 1.5) * (freq - 1.5) / 0.5) + 0.2 * Math.sin(freq * 3);
        const y = canvas.height / 2 - response * canvas.height * 0.35;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw passband indicator
    ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
    ctx.fillRect(canvas.width * 0.2, 0, canvas.width * 0.6, canvas.height);
}

// Window resize listener for responsive canvases
window.addEventListener('resize', () => {
    const hero = document.getElementById('heroSignal');
    if (hero) {
        hero.width = hero.offsetWidth;
        hero.height = hero.offsetHeight;
    }
});

// ===================== TABS SWITCHER =====================
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-nav-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
}

// ===================== DRAG & DROP & FILE HANDLING =====================
function initDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.background = 'rgba(0, 255, 136, 0.08)';
        dropZone.style.borderColor = 'var(--signal-blue)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.background = 'rgba(5, 8, 18, 0.4)';
        dropZone.style.borderColor = 'var(--signal-green)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.background = 'rgba(5, 8, 18, 0.4)';
        dropZone.style.borderColor = 'var(--signal-green)';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });
}

function uploadFile(file) {
    const fileInfo = document.getElementById('fileInfo');
    const operationBtns = document.getElementById('operationBtns');
    const settingsContainer = document.getElementById('settingsContainer');

    fileInfo.innerHTML = `
        <div style="display:inline-flex; align-items:center; gap:0.6rem; background:rgba(0, 212, 255, 0.12); border:1px solid var(--signal-blue); padding:0.45rem 1.2rem; border-radius:24px;">
            <span style="color:var(--signal-blue); font-weight:700;">Uploading:</span> 
            <strong style="color:#ffffff;">${file.name}</strong> 
            <span style="color:var(--text-secondary); font-size:0.85rem;">(${(file.size/1024).toFixed(1)} KB)</span>
        </div>
    `;
    fileInfo.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;

            fileInfo.innerHTML = `
                <div style="display:inline-flex; align-items:center; gap:0.6rem; background:rgba(0, 255, 136, 0.12); border:1px solid var(--signal-green); padding:0.45rem 1.2rem; border-radius:24px;">
                    <span style="color:var(--signal-green); font-weight:700;">✔ Active Ingested Signal:</span> 
                    <strong style="color:#ffffff;">${data.original_name}</strong> 
                    <span style="color:var(--text-secondary); font-size:0.85rem;">[${data.file_type} • ${Math.round(data.sample_rate)} Hz • ${data.duration_s} s]</span>
                </div>
            `;
            fileInfo.style.display = 'block';
            operationBtns.style.display = 'flex';
            settingsContainer.style.display = 'block';

            if (data.sample_rate) {
                document.getElementById('paramSampleRate').value = data.sample_rate;
                const rateLabel = document.getElementById('rateDetectionLabel');
                rateLabel.innerText = `✔ Auto-detected rate: ${data.sample_rate} Hz`;
                rateLabel.style.display = 'inline-block';
            }

            activeSignalContext.filename = data.original_name;
            activeSignalContext.sample_rate = data.sample_rate;
            activeSignalContext.duration_s = data.duration_s;
            if (data.dominant_freq_hz !== undefined) {
                activeSignalContext.dominant_freq_hz = data.dominant_freq_hz;
            }

            // Immediately run analysis to populate results
            executeSuiteOperation('pipeline');
        } else {
            alert("Upload failed: " + (data.error || 'Unknown error'));
        }
    })
    .catch(err => {
        console.error("Upload error:", err);
        alert("Upload error. Please verify the file format.");
    });
}

function loadDemoSignal(type) {
    const fileInfo = document.getElementById('fileInfo');
    const operationBtns = document.getElementById('operationBtns');
    const settingsContainer = document.getElementById('settingsContainer');

    fileInfo.innerHTML = `<span style="color:var(--signal-blue); font-size:0.95rem;">⏳ Loading benchmark preset (${type})...</span>`;
    fileInfo.style.display = 'block';

    fetch(`/api/load-preset/${type}`, { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;

            fileInfo.innerHTML = `
                <div style="display:inline-flex; align-items:center; gap:0.6rem; background:rgba(0, 255, 136, 0.12); border:1px solid var(--signal-green); padding:0.45rem 1.2rem; border-radius:24px;">
                    <span style="color:var(--signal-green); font-weight:700;">✔ Active Benchmark Preset:</span> 
                    <strong style="color:#ffffff;">${data.original_name}</strong> 
                    <span style="color:var(--text-secondary); font-size:0.85rem;">[${data.file_type} • ${Math.round(data.sample_rate)} Hz • ${data.duration_s} s]</span>
                </div>
            `;
            fileInfo.style.display = 'block';
            operationBtns.style.display = 'flex';
            settingsContainer.style.display = 'block';

            if (data.sample_rate) {
                document.getElementById('paramSampleRate').value = data.sample_rate;
                const rateLabel = document.getElementById('rateDetectionLabel');
                rateLabel.innerText = `✔ Preset rate: ${data.sample_rate} Hz`;
                rateLabel.style.display = 'inline-block';
            }

            activeSignalContext.filename = data.original_name;
            activeSignalContext.sample_rate = data.sample_rate;
            activeSignalContext.duration_s = data.duration_s;

            // Execute pipeline
            executeSuiteOperation('pipeline');
        } else {
            alert("Error loading demo preset: " + data.error);
        }
    })
    .catch(err => {
        console.error("Preset load error:", err);
    });
}

// ===================== MICROPHONE VOICE RECORDING =====================
function initMicrophone() {
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
                    const file = new File([audioBlob], `voice_recording_${Date.now()}.wav`, { type: 'audio/wav' });
                    uploadFile(file);
                };

                mediaRecorder.start();
                isRecording = true;
                recBtn.style.background = 'linear-gradient(135deg, #ff3344 0%, #cc0022 100%)';
                recBtn.innerHTML = '🛑 Stop Recording';
            } catch (err) {
                alert('Microphone access denied or unavailable in this browser.');
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            recBtn.style.background = '';
            recBtn.innerHTML = '🎙️ Record Microphone Voice';
        }
    });
}

// ===================== LEIBNITZ DSP OPERATIONS =====================
function executeSuiteOperation(opType) {
    if (!currentAudioFilename) {
        alert("Please load or upload a signal file first.");
        return;
    }

    const sampleRate = parseFloat(document.getElementById('paramSampleRate').value) || activeSampleRate;
    const fftWindow = document.getElementById('paramFftWindow').value;

    let pipeline = [];
    if (opType === 'fft') {
        pipeline = [{ id: 'fft_spectrum', params: { window: fftWindow } }];
    } else if (opType === 'telephony') {
        pipeline = [{ id: 'telephony_bandpass', params: { low_cut: 300, high_cut: 3400 } }];
    } else if (opType === 'vad') {
        pipeline = [{ id: 'vad_cleaner', params: { threshold: 1.5 } }];
    } else if (opType === 'phonetics') {
        pipeline = [{ id: 'indic_phonetics', params: {} }];
    } else {
        // Full pipeline
        pipeline = [
            { id: 'vad_cleaner', params: { threshold: 1.5 } },
            { id: 'telephony_bandpass', params: { low_cut: 300, high_cut: 3400 } },
            { id: 'indic_phonetics', params: {} },
            { id: 'fft_spectrum', params: { window: fftWindow } }
        ];
    }

    fetch('/api/process-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: currentAudioFilename,
            sample_rate: sampleRate,
            pipeline: pipeline
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            currentProcessedFilename = data.processed_filename;

            // Update 4-panel plot
            const plotImg = document.getElementById('analyticalPlotImg');
            if (plotImg && data.plot_url) {
                plotImg.src = data.plot_url + '?t=' + Date.now();
            }

            // Update audio player
            const audioPlayer = document.getElementById('procAudioPlayer');
            const audioSource = document.getElementById('procAudioSource');
            if (audioPlayer && audioSource && data.audio_url) {
                audioSource.src = data.audio_url;
                audioPlayer.load();
            }

            // Update download buttons
            document.getElementById('btnDownloadCsv').href = data.download_csv_url;
            document.getElementById('btnDownloadAudio').href = data.audio_url;
            document.getElementById('btnDownloadReport').href = data.download_report_url;

            // Extract and display metrics
            updateMetrics(data.stages);

            // Reveal results section
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            alert("Processing error: " + (data.error || 'Execution failed'));
        }
    })
    .catch(err => {
        console.error("Pipeline error:", err);
    });
}

function updateMetrics(stages) {
    stages.forEach(stage => {
        const metrics = stage.metrics || {};
        if (metrics.snr_db !== undefined) {
            document.getElementById('resSNR').innerText = `${metrics.snr_db} dB`;
            activeSignalContext.snr_db = metrics.snr_db;
        }
        if (metrics.mean_pitch_f0_hz !== undefined) {
            document.getElementById('resPitch').innerText = `${metrics.mean_pitch_f0_hz} Hz`;
            activeSignalContext.mean_pitch_f0_hz = metrics.mean_pitch_f0_hz;
        }
        if (metrics.speech_ratio !== undefined) {
            document.getElementById('resSpeechRatio').innerText = `${Math.round(metrics.speech_ratio * 100)}%`;
        }
        if (metrics.dominant_frequency_hz !== undefined) {
            document.getElementById('resDomFreq').innerText = `${metrics.dominant_frequency_hz} Hz`;
            activeSignalContext.dominant_freq_hz = metrics.dominant_frequency_hz;
        }
    });
}

function resetUpload() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('operationBtns').style.display = 'none';
    document.getElementById('settingsContainer').style.display = 'none';
    document.getElementById('rateDetectionLabel').style.display = 'none';
    currentAudioFilename = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
}

// ===================== SARVAM AI API CALLS =====================

function runSarvamSTT() {
    const fileToUse = currentProcessedFilename || currentAudioFilename;
    if (!fileToUse) {
        alert("Please load or process a signal first.");
        return;
    }

    const lang = document.getElementById('sttLanguage').value;
    const outBox = document.getElementById('sttTranscript');
    outBox.innerHTML = `<span style="color:var(--signal-blue);">🔄 Transcribing with Sarvam Saaras Indic Engine (${lang})...</span>`;

    fetch('/api/sarvam/stt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: fileToUse,
            language_code: lang
        })
    })
    .then(res => res.json())
    .then(data => {
        outBox.innerHTML = `<strong>${data.transcript || 'No transcript returned'}</strong>`;
        document.getElementById('sttMeta').innerHTML = `
            <span>Engine: <strong>${data.provider || 'Sarvam Saaras'}</strong></span>
            <span>Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></span>
            <span>Mode: <strong>${data.mode}</strong></span>
        `;
    })
    .catch(err => {
        outBox.innerText = "Error communicating with Sarvam STT service.";
    });
}

function runSarvamTTS() {
    const text = document.getElementById('ttsInputText').value.trim();
    if (!text) {
        alert("Please enter text to synthesize.");
        return;
    }

    const lang = document.getElementById('ttsLanguage').value;
    const speaker = document.getElementById('ttsSpeaker').value;
    const btn = document.getElementById('btnRunTTS');
    btn.innerText = "🔄 Synthesizing with Bulbul...";
    btn.disabled = true;

    fetch('/api/sarvam/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            target_language_code: lang,
            speaker: speaker
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            currentAudioFilename = data.filename;
            activeSampleRate = data.sample_rate;

            const fileInfo = document.getElementById('fileInfo');
            fileInfo.innerHTML = `
                <div style="display:inline-flex; align-items:center; gap:0.6rem; background:rgba(0, 212, 255, 0.12); border:1px solid var(--signal-blue); padding:0.45rem 1.2rem; border-radius:24px;">
                    <span style="color:var(--signal-blue); font-weight:700;">✔ Synthesized Voice:</span> 
                    <strong style="color:#ffffff;">${data.original_name}</strong> 
                    <span style="color:var(--text-secondary); font-size:0.85rem;">[${data.file_type} • ${Math.round(data.sample_rate)} Hz • ${data.duration_s} s]</span>
                </div>
            `;
            fileInfo.style.display = 'block';
            document.getElementById('operationBtns').style.display = 'flex';
            document.getElementById('settingsContainer').style.display = 'block';

            activeSignalContext.filename = data.original_name;
            activeSignalContext.sample_rate = data.sample_rate;
            activeSignalContext.duration_s = data.duration_s;

            // Execute full Leibnitz pipeline on the synthesized speech!
            executeSuiteOperation('pipeline');
        } else {
            alert("TTS Synthesis error: " + (data.error || 'Failed'));
        }
    })
    .catch(err => {
        console.error("TTS error:", err);
    })
    .finally(() => {
        btn.innerText = "🔊 Synthesize & Load into Leibnitz Studio";
        btn.disabled = false;
    });
}

function sendCopilotMessage() {
    const input = document.getElementById('copilotInput');
    const msg = input.value.trim();
    if (!msg) return;

    const chatWin = document.getElementById('copilotChatWindow');
    chatWin.innerHTML += `<div class="chat-bubble-leibnitz user"><strong>You:</strong> ${msg}</div>`;
    input.value = '';
    chatWin.scrollTop = chatWin.scrollHeight;

    const lang = document.getElementById('copilotLang').value;

    fetch('/api/sarvam/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: msg,
            signal_context: activeSignalContext,
            language: lang
        })
    })
    .then(res => res.json())
    .then(data => {
        chatWin.innerHTML += `
            <div class="chat-bubble-leibnitz bot">
                <div style="font-size:0.75rem; color:var(--sarvam-amber); margin-bottom:0.25rem;">
                    🤖 ${data.provider} (${data.model})
                </div>
                ${data.message}
            </div>
        `;
        chatWin.scrollTop = chatWin.scrollHeight;
    })
    .catch(err => {
        chatWin.innerHTML += `<div class="chat-bubble-leibnitz bot" style="border-color:#ff3344;">Error reaching Sarvam Indic Copilot.</div>`;
    });
}
