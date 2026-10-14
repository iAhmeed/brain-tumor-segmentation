/**
 * NeuroSegment AI — Client-Side Application Logic
 * Fast, reactive multi-layer canvas rendering and API integration
 */

const API_BASE = window.location.origin.includes('8000')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

// Global application state
const state = {
  currentSampleId: null,
  activeCaseName: null,
  baseImage: null,
  masks: { WT: null, TC: null, ET: null },
  dice: null,
  visibility: { WT: true, TC: true, ET: true },
  opacity: 0.50,
  loading: false,
  samples: [],
  width: 192,
  height: 192,
};

// DOM Element references
const DOM = {
  // Status
  connIndicator: document.getElementById('connection-status'),
  connText: document.getElementById('connection-text'),
  connDot: document.querySelector('.status-dot'),
  
  // Gallery & Upload
  gallery: document.getElementById('sample-gallery'),
  toggleUploadBtn: document.getElementById('toggle-upload-btn'),
  uploadPanel: document.getElementById('upload-panel'),
  tabBtnNii: document.getElementById('tab-btn-nii'),
  tabBtnNpy: document.getElementById('tab-btn-npy'),
  niiSection: document.getElementById('nii-upload-section'),
  npySection: document.getElementById('npy-upload-section'),
  
  // Canvas & Viewer
  canvasStage: document.getElementById('canvas-stage'),
  canvas: document.getElementById('mri-canvas'),
  canvasLoader: document.getElementById('canvas-loader'),
  canvasEmpty: document.getElementById('canvas-empty'),
  activeCaseTag: document.getElementById('active-case-tag'),
  btnDownload: document.getElementById('btn-download-overlay'),
  
  // Controls
  toggleWt: document.getElementById('toggle-wt'),
  toggleTc: document.getElementById('toggle-tc'),
  toggleEt: document.getElementById('toggle-et'),
  opacitySlider: document.getElementById('opacity-slider'),
  opacityVal: document.getElementById('opacity-val'),
  
  // Metrics
  diceMeanVal: document.getElementById('dice-mean-val'),
  diceMeanUnit: document.getElementById('dice-mean-unit'),
  diceStatusNote: document.getElementById('dice-status-note'),
  diceWtVal: document.getElementById('dice-wt-val'),
  diceTcVal: document.getElementById('dice-tc-val'),
  diceEtVal: document.getElementById('dice-et-val'),
  pixelsWtVal: document.getElementById('pixels-wt-val'),
  pixelsTcVal: document.getElementById('pixels-tc-val'),
  pixelsEtVal: document.getElementById('pixels-et-val'),
  barWt: document.getElementById('bar-wt'),
  barTc: document.getElementById('bar-tc'),
  barEt: document.getElementById('bar-et'),
  
  // Upload inputs & buttons
  btnSubmitNii: document.getElementById('btn-submit-nii'),
  btnSubmitNpy: document.getElementById('btn-submit-npy'),
  toastContainer: document.getElementById('toast-container'),
};

const ctx = DOM.canvas ? DOM.canvas.getContext('2d') : null;

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  checkBackendHealth();
  loadSampleGallery();
});

function initEventListeners() {
  // Layer visibility toggles
  DOM.toggleWt.addEventListener('change', (e) => {
    state.visibility.WT = e.target.checked;
    renderCanvas();
  });
  DOM.toggleTc.addEventListener('change', (e) => {
    state.visibility.TC = e.target.checked;
    renderCanvas();
  });
  DOM.toggleEt.addEventListener('change', (e) => {
    state.visibility.ET = e.target.checked;
    renderCanvas();
  });

  // Opacity slider
  DOM.opacitySlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    state.opacity = val / 100.0;
    DOM.opacityVal.textContent = `${val}%`;
    renderCanvas();
  });

  // Download composite image
  DOM.btnDownload.addEventListener('click', downloadCanvas);

  // Toggle Upload Drawer
  DOM.toggleUploadBtn.addEventListener('click', () => {
    const isCollapsed = DOM.uploadPanel.classList.toggle('collapsed');
    DOM.toggleUploadBtn.setAttribute('aria-expanded', !isCollapsed);
    DOM.uploadPanel.setAttribute('aria-hidden', isCollapsed);
  });

  // Upload format tabs
  DOM.tabBtnNii.addEventListener('click', () => {
    DOM.tabBtnNii.classList.add('active');
    DOM.tabBtnNpy.classList.remove('active');
    DOM.niiSection.classList.remove('hidden');
    DOM.npySection.classList.add('hidden');
  });

  DOM.tabBtnNpy.addEventListener('click', () => {
    DOM.tabBtnNpy.classList.add('active');
    DOM.tabBtnNii.classList.remove('active');
    DOM.npySection.classList.remove('hidden');
    DOM.niiSection.classList.add('hidden');
  });

  // Setup file dropboxes
  setupDropZone('dropbox-t1', 'file-input-t1', 'hint-t1');
  setupDropZone('dropbox-t1ce', 'file-input-t1ce', 'hint-t1ce');
  setupDropZone('dropbox-t2', 'file-input-t2', 'hint-t2');
  setupDropZone('dropbox-flair', 'file-input-flair', 'hint-flair');
  setupDropZone('dropbox-seg', 'file-input-seg', 'hint-seg');
  setupDropZone('dropbox-input-npy', 'file-input-npy', 'hint-input-npy');
  setupDropZone('dropbox-gt-npy', 'file-gt-npy', 'hint-gt-npy');

  // Submit buttons
  DOM.btnSubmitNii.addEventListener('click', handleNiiSubmit);
  DOM.btnSubmitNpy.addEventListener('click', handleNpySubmit);
}

// ============================================================================
// Backend Connection & Sample Gallery
// ============================================================================
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      DOM.connDot.className = 'status-dot status-online';
      DOM.connText.textContent = `Online (${data.device.toUpperCase()})`;
    } else {
      setOfflineStatus();
    }
  } catch (err) {
    setOfflineStatus();
  }
}

function setOfflineStatus() {
  DOM.connDot.className = 'status-dot status-offline';
  DOM.connText.textContent = 'Offline';
}

async function loadSampleGallery() {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const samples = await res.json();
    state.samples = samples;
    renderSampleCards(samples);

    // Auto-select first sample for immediate wow factor
    if (samples.length > 0 && !state.currentSampleId) {
      selectSample(samples[0].id);
    }
  } catch (err) {
    DOM.gallery.innerHTML = `
      <div class="empty-state">
        <p>Could not connect to backend server at <code>${API_BASE}</code>.</p>
        <p style="font-size:0.8rem; margin-top:0.4rem; color:var(--text-muted)">Ensure FastAPI is running with: <code>uvicorn app.main:app --reload --port 8000</code></p>
      </div>`;
  }
}

function renderSampleCards(samples) {
  if (!samples || samples.length === 0) {
    DOM.gallery.innerHTML = `<p class="empty-text">No pre-bundled samples found.</p>`;
    return;
  }

  DOM.gallery.innerHTML = samples.map((sample) => `
    <div class="sample-card ${state.currentSampleId === sample.id ? 'active' : ''}" 
         id="sample-card-${sample.id}" 
         tabindex="0"
         role="button"
         aria-label="Select ${sample.label}"
         onkeydown="if(event.key === 'Enter' || event.key === ' ') { selectSample('${sample.id}'); event.preventDefault(); }"
         onclick="selectSample('${sample.id}')">
      <div class="sample-img-container">
        <img class="sample-thumb" src="${API_BASE}${sample.preview_url}" alt="${sample.label}" loading="lazy">
      </div>
      <div class="sample-meta">
        <span class="sample-title" title="${sample.label}">${sample.label}</span>
        <div class="sample-details-row">
          <span>Slice #${sample.slice_idx}</span>
          <span class="tumor-badge">${sample.tumor_pixels.toLocaleString()} voxels</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ============================================================================
// Inference Execution
// ============================================================================
async function selectSample(sampleId) {
  state.currentSampleId = sampleId;
  const sampleObj = state.samples.find(s => s.id === sampleId);
  state.activeCaseName = sampleObj ? sampleObj.label : `Sample ${sampleId}`;
  DOM.activeCaseTag.textContent = state.activeCaseName;

  // Update active styling on cards
  document.querySelectorAll('.sample-card').forEach(card => {
    card.classList.remove('active');
  });
  const activeCard = document.getElementById(`sample-card-${sampleId}`);
  if (activeCard) activeCard.classList.add('active');

  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/predict/sample/${sampleId}`, { method: 'POST' });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    handlePredictionResponse(data);
    showToast(`Inference completed for ${state.activeCaseName}`, 'success');
  } catch (err) {
    showToast(`Segmentation failed: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

function handlePredictionResponse(data) {
  state.masks = data.masks;
  state.dice = data.dice;
  state.width = data.width || 192;
  state.height = data.height || 192;

  // Update canvas internal resolution
  DOM.canvas.width = state.width;
  DOM.canvas.height = state.height;

  // Load preview base image
  const img = new Image();
  img.onload = () => {
    state.baseImage = img;
    DOM.canvasEmpty.classList.add('hidden');
    DOM.canvas.classList.remove('hidden');
    renderCanvas();
    updateMetricsUI();
  };
  img.src = data.preview_base64;
}

function setLoading(isLoading) {
  state.loading = isLoading;
  if (isLoading) {
    DOM.canvasLoader.classList.remove('hidden');
  } else {
    DOM.canvasLoader.classList.add('hidden');
  }
}

// ============================================================================
// Multi-Layer Canvas Renderer
// ============================================================================
function renderCanvas() {
  if (!ctx || !state.baseImage) return;

  const W = state.width;
  const H = state.height;

  // 1. Draw base grayscale MRI slice
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(state.baseImage, 0, 0, W, H);

  // If no masks or opacity is 0, we're done
  if (!state.masks.WT || state.opacity <= 0.001) return;

  // 2. Build multi-color overlay buffer
  const imgData = ctx.getImageData(0, 0, W, H);
  const pixels = imgData.data;

  const wtMask = state.masks.WT;
  const tcMask = state.masks.TC;
  const etMask = state.masks.ET;

  const showWt = state.visibility.WT;
  const showTc = state.visibility.TC;
  const showEt = state.visibility.ET;

  const alpha = state.opacity;

  // Tumor color definitions (RGB)
  // WT: Coral Red (255, 83, 83)
  // TC: Emerald Green (16, 185, 129)
  // ET: Cyan Blue (0, 210, 255)

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;

      const hasEt = showEt && etMask[y][x] === 1;
      const hasTc = showTc && tcMask[y][x] === 1;
      const hasWt = showWt && wtMask[y][x] === 1;

      if (!hasEt && !hasTc && !hasWt) continue;

      let r = 0, g = 0, b = 0;

      // Hierarchical display: ET inside TC inside WT
      if (hasEt) {
        // Enhancing tumor
        r = 0; g = 210; b = 255;
      } else if (hasTc) {
        // Tumor core (non-enhancing / necrotic)
        r = 16; g = 185; b = 129;
      } else if (hasWt) {
        // Whole tumor / peritumoral edema
        r = 255; g = 83; b = 83;
      }

      // Alpha blend on top of base grayscale pixel
      pixels[idx]     = Math.round(pixels[idx]     * (1 - alpha) + r * alpha);
      pixels[idx + 1] = Math.round(pixels[idx + 1] * (1 - alpha) + g * alpha);
      pixels[idx + 2] = Math.round(pixels[idx + 2] * (1 - alpha) + b * alpha);
    }
  }

  // Write composite pixels back to canvas
  ctx.putImageData(imgData, 0, 0);
}

// ============================================================================
// Metrics & Statistics Display
// ============================================================================
function updateMetricsUI() {
  const dice = state.dice;
  const masks = state.masks;

  // Calculate positive pixel counts
  const countPixels = (mask) => {
    if (!mask) return 0;
    let sum = 0;
    for (let i = 0; i < mask.length; i++) {
      for (let j = 0; j < mask[i].length; j++) {
        if (mask[i][j]) sum++;
      }
    }
    return sum;
  };

  const wtPixels = countPixels(masks.WT);
  const tcPixels = countPixels(masks.TC);
  const etPixels = countPixels(masks.ET);

  DOM.pixelsWtVal.textContent = `${wtPixels.toLocaleString()} px`;
  DOM.pixelsTcVal.textContent = `${tcPixels.toLocaleString()} px`;
  DOM.pixelsEtVal.textContent = `${etPixels.toLocaleString()} px`;

  if (dice) {
    // Ground truth was provided -> display Dice metrics
    const meanPct = (dice.mean * 100).toFixed(1);
    DOM.diceMeanVal.textContent = meanPct;
    DOM.diceMeanUnit.textContent = '%';
    DOM.diceStatusNote.textContent = 'Evaluated vs Ground Truth';

    const wtPct = Math.round(dice.WT * 100);
    const tcPct = Math.round(dice.TC * 100);
    const etPct = Math.round(dice.ET * 100);

    DOM.diceWtVal.textContent = `${(dice.WT * 100).toFixed(1)}%`;
    DOM.diceTcVal.textContent = `${(dice.TC * 100).toFixed(1)}%`;
    DOM.diceEtVal.textContent = `${(dice.ET * 100).toFixed(1)}%`;

    DOM.barWt.style.width = `${wtPct}%`;
    DOM.barTc.style.width = `${tcPct}%`;
    DOM.barEt.style.width = `${etPct}%`;
  } else {
    // Custom upload without ground truth
    DOM.diceMeanVal.textContent = 'N/A';
    DOM.diceMeanUnit.textContent = '';
    DOM.diceStatusNote.textContent = 'Raw scan (No ground truth mask provided)';

    DOM.diceWtVal.textContent = 'N/A';
    DOM.diceTcVal.textContent = 'N/A';
    DOM.diceEtVal.textContent = 'N/A';

    DOM.barWt.style.width = '0%';
    DOM.barTc.style.width = '0%';
    DOM.barEt.style.width = '0%';
  }
}

// ============================================================================
// Custom Scan Upload Handlers
// ============================================================================
function setupDropZone(boxId, inputId, hintId) {
  const box = document.getElementById(boxId);
  const input = document.getElementById(inputId);
  const hint = document.getElementById(hintId);

  if (!box || !input) return;

  box.addEventListener('click', () => input.click());

  box.addEventListener('dragover', (e) => {
    e.preventDefault();
    box.classList.add('drag-over');
  });

  box.addEventListener('dragleave', () => {
    box.classList.remove('drag-over');
  });

  box.addEventListener('drop', (e) => {
    e.preventDefault();
    box.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      input.files = e.dataTransfer.files;
      handleFileSelected(input.files[0], box, hint);
    }
  });

  input.addEventListener('change', () => {
    if (input.files.length > 0) {
      handleFileSelected(input.files[0], box, hint);
    }
  });
}

function handleFileSelected(file, box, hint) {
  box.classList.add('has-file');
  hint.textContent = file.name;
  hint.style.color = '#34D399';
}

async function handleNiiSubmit() {
  const t1 = document.getElementById('file-input-t1').files[0];
  const t1ce = document.getElementById('file-input-t1ce').files[0];
  const t2 = document.getElementById('file-input-t2').files[0];
  const flair = document.getElementById('file-input-flair').files[0];
  const seg = document.getElementById('file-input-seg').files[0];
  const sliceInput = document.getElementById('custom-slice-idx').value;

  if (!t1 || !t1ce || !t2 || !flair) {
    showToast('Please select all 4 required modalities (T1, T1ce, T2, FLAIR)', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('t1', t1);
  formData.append('t1ce', t1ce);
  formData.append('t2', t2);
  formData.append('flair', flair);
  if (seg) formData.append('seg', seg);
  if (sliceInput !== '') formData.append('slice_idx', sliceInput);

  state.activeCaseName = `Custom NIfTI (${t1.name.replace('.nii.gz', '').replace('.nii', '')})`;
  DOM.activeCaseTag.textContent = state.activeCaseName;
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/predict/upload-nii`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    handlePredictionResponse(data);
    showToast(`NIfTI scan segmented! (Slice #${data.slice_idx})`, 'success');
  } catch (err) {
    showToast(`NIfTI segmentation error: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

async function handleNpySubmit() {
  const inputNpy = document.getElementById('file-input-npy').files[0];
  const gtNpy = document.getElementById('file-gt-npy').files[0];

  if (!inputNpy) {
    showToast('Please select a .npy 4-channel input slice file', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('input_file', inputNpy);
  if (gtNpy) formData.append('gt_file', gtNpy);

  state.activeCaseName = `Custom NPY (${inputNpy.name})`;
  DOM.activeCaseTag.textContent = state.activeCaseName;
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/predict/upload-npy`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    handlePredictionResponse(data);
    showToast(`.NPY slice segmented successfully!`, 'success');
  } catch (err) {
    showToast(`.NPY processing error: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

// ============================================================================
// Export & Utilities
// ============================================================================
function downloadCanvas() {
  if (!DOM.canvas || !state.baseImage) {
    showToast('No active segmentation to export', 'error');
    return;
  }

  const link = document.createElement('a');
  link.download = `neurosegment-${state.currentSampleId || 'custom'}.png`;
  link.href = DOM.canvas.toDataURL('image/png');
  link.click();
  showToast('Diagnostic overlay exported to PNG', 'success');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  DOM.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
