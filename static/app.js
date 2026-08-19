/* ==========================================================================
   CYBER-DETECT FRONTEND ENGINE WITH ACTIVE REINFORCEMENT LEARNING & SEARCH
   ========================================================================== */

let sfxEnabled = true;
let currentImageData = null;
let currentPredictionResult = null;
let allCodexCars = [];

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

// INITIALIZATION
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initControls();
    initDropzone();
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
    currentImageData = null;
    currentPredictionResult = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImg').style.display = 'none';
    document.getElementById('camImg').style.display = 'none';
    document.getElementById('viewModeBar').style.display = 'none';
    document.getElementById('resultsHud').style.display = 'none';
    document.getElementById('correctionPanel').style.display = 'none';
    document.getElementById('feedbackStatus').style.display = 'none';
    const emptyState = document.querySelector('.empty-state-hud');
    if (emptyState) emptyState.style.display = 'block';
    playBeep(300, 'sawtooth', 0.1);
}

// INFERENCE & GRAD-CAM CALLS
async function runInference(imageBase64) {
    playScanChirp();
    const resultsHud = document.getElementById('resultsHud');
    resultsHud.style.display = 'none';
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
        currentPredictionResult = data;
        renderResults(data);
    } catch (err) {
        console.error(err);
        alert('Error analyzing image. Please verify server status.');
    }
}

function renderResults(data) {
    const car = data.car_info;
    document.getElementById('resMakeModel').textContent = `${car.make.toUpperCase()} ${car.model.toUpperCase()}`;
    
    const confPct = (data.confidence * 100).toFixed(1);
    document.getElementById('resConfidence').textContent = `${confPct}%`;
    document.getElementById('confBar').style.width = `${Math.min(100, Math.max(10, confPct))}%`;

    // Top alternative matches with reference car photos
    const topRanks = document.getElementById('topRanks');
    topRanks.innerHTML = '';
    data.top_k.forEach(rank => {
        const div = document.createElement('div');
        div.className = 'rank-item';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.gap = '14px';
        div.style.padding = '10px';
        div.style.marginBottom = '8px';
        div.style.background = '#111';
        div.style.border = '1px solid #333';
        div.style.cursor = 'pointer';
        
        div.innerHTML = `
            <img src="${rank.image_url}" style="width: 70px; height: 48px; object-fit: cover; border: 1px solid #666;" alt="${rank.model}">
            <div style="flex: 1;">
                <div style="font-weight: bold; font-size: 14px; color: #fff;">${rank.make} ${rank.model}</div>
                <div style="font-size: 10px; color: #888;">CLICK TO SET AS TRUE MODEL (RL UPDATE)</div>
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
        if (!currentPredictionResult) return;
        const predIdx = currentPredictionResult.car_info.catalog_idx;
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

    // Instant search filter inside correction panel
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
        // Look up original index in allCodexCars
        const origIdx = allCodexCars.findIndex(c => c.id === car.id);
        opt.value = origIdx >= 0 ? origIdx : 0;
        opt.textContent = `${car.make} ${car.model}`;
        select.appendChild(opt);
    });
}

async function submitRLFeedback(isCorrect, targetIdx) {
    if (!currentImageData || !currentPredictionResult) return;
    const predIdx = currentPredictionResult.car_info.catalog_idx;
    const statusDiv = document.getElementById('feedbackStatus');
    statusDiv.style.display = 'block';
    statusDiv.textContent = '>> TRANSMITTING RL UPDATE TO RTX 3050 GPU...';

    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_data: currentImageData,
                predicted_idx: predIdx,
                correct_idx: targetIdx,
                is_correct: isCorrect
            })
        });

        const data = await res.json();
        playBeep(980, 'square', 0.2);
        statusDiv.textContent = `>> [SUCCESS] ${data.message}`;
        
        const badge = document.getElementById('rlBadge');
        if (data.rl_stats) {
            badge.textContent = `RL REINFORCED: ${data.rl_stats.total_feedbacks} TIMES`;
        }

        setTimeout(() => {
            runInference(currentImageData);
        }, 600);

    } catch (e) {
        statusDiv.textContent = '>> [ERROR] Failed to apply feedback.';
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
            <img src="${car.image_url}" style="width: 100%; height: 140px; object-fit: cover; border-bottom: 1px solid #333; margin-bottom: 10px;" alt="${car.model}">
            <div class="codex-title" style="font-size: 15px;">${car.make.toUpperCase()} // ${car.model.toUpperCase()}</div>
            <button class="retro-btn btn-small" style="width: 100%; margin-top: 10px;">&gt; SCAN THIS CAR &lt;</button>
        `;
        card.onclick = () => {
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
        const gpuStatus = document.getElementById('gpuStatus');
        if (info.cuda_available) {
            gpuStatus.textContent = `GPU: ${info.gpu_name} (CUDA ACTIVE)`;
        } else {
            gpuStatus.textContent = 'GPU: CPU MODE';
        }
        document.getElementById('classCount').textContent = `CATALOG: ${info.num_classes} CARS`;
        if (info.rl_stats && info.rl_stats.total_feedbacks > 0) {
            document.getElementById('rlBadge').textContent = `RL REINFORCED: ${info.rl_stats.total_feedbacks} TIMES`;
        }
    } catch (e) {}
}
