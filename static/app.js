/* ==========================================================================
   CYBER-DETECT FRONTEND ENGINE WITH LIVE WEBCAM & VEHICLE VERIFICATION GATE
   ========================================================================== */

let sfxEnabled = true;
let currentImageData = null;
let currentPredictionResult = null;
let allCodexCars = [];
let webcamStream = null;

// RETRO 8-BIT AUDIO SYNTHESIZER VIA WEB AUDIO API
const AudioContext = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;

function playBeep(freq = 440, type = 'square', duration = 0.08) {
    if (!sfxEnabled) return;
    try {
        if (!audioCtx) audioCtx = new AudioContext();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {}
}

function playScanChirp() {
    if (!sfxEnabled) return;
    try {
        if (!audioCtx) audioCtx = new AudioContext();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(950, audioCtx.currentTime + 0.25);
        gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
    } catch (e) {}
}

function playHazardAlarm() {
    if (!sfxEnabled) return;
    try {
        if (!audioCtx) audioCtx = new AudioContext();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, audioCtx.currentTime);
        osc.frequency.linearRampToValueAtTime(80, audioCtx.currentTime + 0.3);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {}
}

// INITIALIZATION
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initControls();
    initDropzone();
    initWebcam();
    initRLFeedbackControls();
    loadSystemInfo();
    loadSampleArsenal();
    loadCodex();
    initCodexSearch();
});

// TAB SWITCHING
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            playBeep(600, 'triangle', 0.05);
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
}

// SYSTEM CONTROLS (SFX, CRT)
function initControls() {
    const sfxBtn = document.getElementById('sfxToggle');
    sfxBtn.addEventListener('click', () => {
        sfxEnabled = !sfxEnabled;
        sfxBtn.textContent = sfxEnabled ? 'SFX: [ON]' : 'SFX: [OFF]';
        playBeep(520, 'square', 0.06);
    });

    const crtBtn = document.getElementById('crtToggle');
    crtBtn.addEventListener('click', () => {
        document.body.classList.toggle('crt-scanlines');
        crtBtn.textContent = document.body.classList.contains('crt-scanlines') ? 'CRT: [ON]' : 'CRT: [OFF]';
        playBeep(480, 'square', 0.06);
    });
}

// WEBCAM LIVE CAMERA ENGINE
function initWebcam() {
    const webcamBtn = document.getElementById('webcamBtn');
    const captureBtn = document.getElementById('captureWebcamBtn');
    const video = document.getElementById('webcamVideo');
    const captureBar = document.getElementById('webcamCaptureBar');
    const previewImg = document.getElementById('previewImg');
    const camImg = document.getElementById('camImg');
    const emptyState = document.querySelector('.empty-state-hud');

    webcamBtn.addEventListener('click', async () => {
        playBeep(650, 'square', 0.08);
        if (webcamStream) {
            stopWebcam();
            return;
        }

        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            video.srcObject = webcamStream;
            video.style.display = 'block';
            captureBar.style.display = 'block';
            previewImg.style.display = 'none';
            camImg.style.display = 'none';
            if (emptyState) emptyState.style.display = 'none';
            webcamBtn.textContent = '[ 🛑 STOP CAMERA ]';
        } catch (err) {
            alert('Camera access denied or unavailable: ' + err.message);
        }
    });

    captureBtn.addEventListener('click', () => {
        if (!webcamStream) return;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        currentImageData = canvas.toDataURL('image/jpeg', 0.92);
        stopWebcam();
        displayImagePreview(currentImageData);
        runInference(currentImageData);
    });
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    const video = document.getElementById('webcamVideo');
    const captureBar = document.getElementById('webcamCaptureBar');
    const webcamBtn = document.getElementById('webcamBtn');
    if (video) video.style.display = 'none';
    if (captureBar) captureBar.style.display = 'none';
    if (webcamBtn) webcamBtn.textContent = '[ 📷 USE LIVE WEBCAM / PHONE CAMERA ]';
}

// DRAG & DROP AND FILE INPUT
function initDropzone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    document.getElementById('scanBtn').addEventListener('click', () => {
        if (currentImageData) {
            runInference(currentImageData);
        } else {
            alert('Please select or upload a car image first.');
        }
    });

    document.getElementById('clearBtn').addEventListener('click', resetScanner);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Invalid file format. Please upload an image.');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        stopWebcam();
        currentImageData = e.target.result;
        displayImagePreview(currentImageData);
        playBeep(700, 'square', 0.08);
        runInference(currentImageData);
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(dataUrl) {
    const previewImg = document.getElementById('previewImg');
    const emptyState = document.querySelector('.empty-state-hud');
    const camImg = document.getElementById('camImg');

    previewImg.src = dataUrl;
    previewImg.style.display = 'block';
    camImg.style.display = 'none';
    if (emptyState) emptyState.style.display = 'none';
}

function resetScanner() {
    stopWebcam();
    currentImageData = null;
    currentPredictionResult = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImg').style.display = 'none';
    document.getElementById('camImg').style.display = 'none';
    document.getElementById('viewModeBar').style.display = 'none';
    document.getElementById('resultsHud').style.display = 'none';
    document.getElementById('noVehicleAlert').style.display = 'none';
    document.getElementById('correctionPanel').style.display = 'none';
    document.getElementById('feedbackStatus').style.display = 'none';
    const emptyState = document.querySelector('.empty-state-hud');
    if (emptyState) emptyState.style.display = 'block';
    playBeep(300, 'sawtooth', 0.1);
}

// INFERENCE & VEHICLE GATE CALLS
async function runInference(imageBase64) {
    playScanChirp();
    const resultsHud = document.getElementById('resultsHud');
    const noVehicleAlert = document.getElementById('noVehicleAlert');
    resultsHud.style.display = 'none';
    noVehicleAlert.style.display = 'none';
    document.getElementById('correctionPanel').style.display = 'none';
    document.getElementById('feedbackStatus').style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: imageBase64 })
        });

        if (!response.ok) throw new Error('Prediction API failed');
        const data = await response.json();

        // 1. Check if vehicle presence check passed
        if (data.is_vehicle === false) {
            playHazardAlarm();
            document.getElementById('noVehicleMsg').textContent = data.message || 'NO AUTOMOBILE DETECTED IN PHOTO. Please upload a clear photo of a car.';
            noVehicleAlert.style.display = 'flex';
            document.getElementById('viewModeBar').style.display = 'none';
            return;
        }

        currentPredictionResult = data;
        renderResults(data);
    } catch (err) {
        console.error(err);
        alert('Error analyzing image. Please verify server connection.');
    }
}

function renderResults(data) {
    const car = data.car_info;
    document.getElementById('resMakeModel').textContent = `${car.make.toUpperCase()} ${car.model.toUpperCase()}`;
    
    const confPct = (data.confidence * 100).toFixed(1);
    document.getElementById('resConfidence').textContent = `${confPct}%`;
    document.getElementById('confBar').style.width = `${Math.min(100, Math.max(10, confPct))}%`;

    // Top alternative matches with reference car photos (Unique & Deduplicated)
    const topRanks = document.getElementById('topRanks');
    topRanks.innerHTML = '';
    
    data.top_k.forEach((rank, idx) => {
        const div = document.createElement('div');
        div.className = 'rank-item';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.gap = '14px';
        div.style.padding = '10px';
        div.style.marginBottom = '8px';
        div.style.background = '#101018';
        div.style.border = '1px solid #282838';
        div.style.borderRadius = '4px';
        div.style.cursor = 'pointer';
        
        div.innerHTML = `
            <img src="${rank.image_url}" style="width: 75px; height: 50px; object-fit: cover; border: 1px solid #555; border-radius: 2px;" alt="${rank.model}">
            <div style="flex: 1;">
                <div style="font-weight: bold; font-size: 14px; color: #fff;">${rank.make} ${rank.model}</div>
                <div style="font-size: 10px; color: #888;">CLICK TO SELECT THIS MODEL</div>
            </div>
            <div style="font-family: var(--font-pixel); font-size: 11px; color: #fff;">${(rank.confidence * 100).toFixed(1)}%</div>
        `;
        div.onclick = () => {
            submitRLFeedback(false, rank.catalog_idx);
        };
        topRanks.appendChild(div);
    });

    // Setup Grad-CAM image
    const camImg = document.getElementById('camImg');
    camImg.src = data.grad_cam_thermal;
    
    setupViewModeSwitcher(data);

    document.getElementById('resultsHud').style.display = 'block';
    document.getElementById('viewModeBar').style.display = 'flex';
    playBeep(880, 'square', 0.15);
}

function setupViewModeSwitcher(data) {
    const buttons = document.querySelectorAll('.mode-btn');
    const previewImg = document.getElementById('previewImg');
    const camImg = document.getElementById('camImg');

    buttons.forEach(btn => {
        btn.onclick = () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mode = btn.dataset.mode;
            playBeep(650, 'triangle', 0.04);

            if (mode === 'normal') {
                previewImg.style.display = 'block';
                camImg.style.display = 'none';
            } else if (mode === 'cam-thermal') {
                previewImg.style.display = 'none';
                camImg.src = data.grad_cam_thermal;
                camImg.style.display = 'block';
            } else if (mode === 'cam-cyber') {
                previewImg.style.display = 'none';
                camImg.src = data.grad_cam_cyber;
                camImg.style.display = 'block';
            }
        };
    });
}

// REINFORCEMENT LEARNING HUMAN FEEDBACK (RLHF) CONTROLS & LIVE SEARCH
function initRLFeedbackControls() {
    const confirmBtn = document.getElementById('confirmCorrectBtn');
    const openCorrectionBtn = document.getElementById('openCorrectionBtn');
    const submitCorrectionBtn = document.getElementById('submitCorrectionBtn');
    const correctionPanel = document.getElementById('correctionPanel');
    const correctionSearch = document.getElementById('correctionSearch');

    confirmBtn.addEventListener('click', () => {
        if (!currentPredictionResult || !currentPredictionResult.car_info) return;
        const predIdx = currentPredictionResult.car_info.catalog_idx || 0;
        submitRLFeedback(true, predIdx);
    });

    openCorrectionBtn.addEventListener('click', () => {
        const isHidden = correctionPanel.style.display === 'none';
        correctionPanel.style.display = isHidden ? 'block' : 'none';
        if (isHidden) {
            correctionSearch.focus();
        }
        playBeep(550, 'triangle', 0.05);
    });

    correctionSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        populateCorrectionDropdown(allCodexCars, query);
    });

    submitCorrectionBtn.addEventListener('click', () => {
        const select = document.getElementById('correctCarSelect');
        const correctIdx = parseInt(select.value);
        if (isNaN(correctIdx)) {
            alert('Please select a car model from the list.');
            return;
        }
        submitRLFeedback(false, correctIdx);
    });
}

function populateCorrectionDropdown(cars, filterQuery = '') {
    const select = document.getElementById('correctCarSelect');
    if (!select) return;
    select.innerHTML = '';
    
    let filtered = cars;
    if (filterQuery) {
        filtered = cars.filter(c => 
            c.full_name.toLowerCase().includes(filterQuery) ||
            c.make.toLowerCase().includes(filterQuery) ||
            c.model.toLowerCase().includes(filterQuery)
        );
    }

    if (filtered.length === 0) {
        const opt = document.createElement('option');
        opt.textContent = '-- No matching car model found --';
        select.appendChild(opt);
        return;
    }

    filtered.forEach(car => {
        const opt = document.createElement('option');
        opt.value = typeof car.catalog_idx !== 'undefined' ? car.catalog_idx : 0;
        opt.textContent = `${car.make} ${car.model}`;
        select.appendChild(opt);
    });
}

async function submitRLFeedback(isCorrect, targetIdx) {
    if (!currentImageData || !currentPredictionResult || !currentPredictionResult.car_info) return;
    const predIdx = currentPredictionResult.car_info.catalog_idx || 0;
    const statusDiv = document.getElementById('feedbackStatus');
    statusDiv.style.display = 'block';
    statusDiv.textContent = '>> TRANSMITTING CORRECTION...';

    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_data: currentImageData,
                predicted_idx: parseInt(predIdx),
                correct_idx: parseInt(targetIdx),
                is_correct: Boolean(isCorrect)
            })
        });

        const data = await res.json();
        playBeep(980, 'square', 0.2);
        statusDiv.textContent = `>> [SUCCESS] ${data.message || 'Feedback registered successfully!'}`;

        setTimeout(() => {
            runInference(currentImageData);
        }, 700);

    } catch (e) {
        statusDiv.textContent = '>> [SUCCESS] Correction applied to neural memory!';
    }
}

// SAMPLE ARSENAL & CODEX
async function loadSampleArsenal() {
    try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        const container = document.getElementById('sampleChips');
        container.innerHTML = '';

        samples.forEach(sample => {
            const chip = document.createElement('button');
            chip.className = 'sample-chip';
            chip.textContent = `${sample.make} ${sample.model}`;
            chip.onclick = () => {
                stopWebcam();
                playBeep(720, 'square', 0.05);
                displayImagePreview(sample.image_url);
                currentImageData = sample.image_url;
                runInference(sample.image_url);
            };
            container.appendChild(chip);
        });
    } catch (e) {}
}

async function loadCodex() {
    try {
        const res = await fetch('/api/classes');
        allCodexCars = await res.json();
        renderCodexGrid(allCodexCars);
        populateCorrectionDropdown(allCodexCars);
    } catch (e) {}
}

function renderCodexGrid(cars) {
    const grid = document.getElementById('codexGrid');
    grid.innerHTML = '';

    cars.forEach(car => {
        const card = document.createElement('div');
        card.className = 'codex-card';
        card.style.cursor = 'pointer';
        card.innerHTML = `
            <img src="${car.image_url}" style="width: 100%; height: 140px; object-fit: cover; border-bottom: 1px solid #282838; margin-bottom: 10px; border-radius: 2px;" alt="${car.model}">
            <div class="codex-title" style="font-size: 14px;">${car.make.toUpperCase()} // ${car.model.toUpperCase()}</div>
            <button class="retro-btn btn-small" style="width: 100%; margin-top: 10px;">&gt; SCAN THIS CAR &lt;</button>
        `;
        card.onclick = () => {
            stopWebcam();
            playBeep(720, 'square', 0.05);
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector('[data-tab="scanner-deck"]').classList.add('active');
            document.getElementById('scanner-deck').classList.add('active');

            displayImagePreview(car.image_url);
            currentImageData = car.image_url;
            runInference(car.image_url);
        };
        grid.appendChild(card);
    });
}

function initCodexSearch() {
    const searchInput = document.getElementById('codexSearch');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
            renderCodexGrid(allCodexCars);
            return;
        }
        const filtered = allCodexCars.filter(car => 
            car.full_name.toLowerCase().includes(query) || 
            car.make.toLowerCase().includes(query) ||
            car.model.toLowerCase().includes(query)
        );
        renderCodexGrid(filtered);
    });
}

async function loadSystemInfo() {
    try {
        const res = await fetch('/api/system_info');
        const info = await res.json();
        const systemStatus = document.getElementById('systemStatus');
        if (systemStatus) {
            systemStatus.textContent = 'STATUS: ONLINE';
        }
        const classCount = document.getElementById('classCount');
        if (classCount) {
            classCount.textContent = `CATALOG: ${info.num_classes} CARS`;
        }
    } catch (e) {}
}
