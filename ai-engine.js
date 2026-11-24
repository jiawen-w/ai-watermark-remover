/**
 * Pure Client-Side AI Watermark Removal & Auto-Detection Engine
 * - Runs LaMa ONNX (lama_fp32.onnx) directly in the browser via ONNX Runtime Web (WASM/WebGPU)
 * - Caches the ~198MB model in IndexedDB so visitors only download it once
 * - Uses smart crop-around-mask compositing to preserve 100% original image resolution
 * - Implements pure-JS multi-image Sobel gradient-consistency auto-watermark detection
 */

const MODEL_DB_NAME = 'ai_watermark_remover_cache_v1';
const MODEL_STORE = 'models';
const MODEL_KEY = 'lama_fp32_v1';
const MODEL_URLS = [
  'https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
  'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
];

let ortSession = null;
let sessionPromise = null;

function openModelDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(MODEL_DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(MODEL_STORE)) {
        req.result.createObjectStore(MODEL_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getCachedModel() {
  try {
    const db = await openModelDB();
    return await new Promise((resolve) => {
      const tx = db.transaction(MODEL_STORE, 'readonly');
      const req = tx.objectStore(MODEL_STORE).get(MODEL_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function saveCachedModel(buffer) {
  try {
    const db = await openModelDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(MODEL_STORE, 'readwrite');
      tx.objectStore(MODEL_STORE).put(buffer, MODEL_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (e) {
    console.warn('IndexedDB cache skipped:', e);
  }
}

async function fetchModelWithProgress(onProgress) {
  let lastErr = null;
  for (const url of MODEL_URLS) {
    try {
      const resp = await fetch(url, { mode: 'cors' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const total = Number(resp.headers.get('content-length') || 208044816);
      const reader = resp.body.getReader();
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        if (onProgress) onProgress(loaded, total);
      }
      const out = new Uint8Array(loaded);
      let offset = 0;
      for (const c of chunks) {
        out.set(c, offset);
        offset += c.byteLength;
      }
      return out.buffer;
    } catch (err) {
      console.warn('Model mirror failed:', url, err);
      lastErr = err;
    }
  }
  throw lastErr || new Error('模型下载失败，请检查网络连接');
}

export async function ensureModelLoaded(onStatus) {
  if (ortSession) return ortSession;
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web 未加载完成，请刷新页面重试');
    }
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    // Single-threaded WASM works everywhere including GitHub Pages without COOP/COEP headers
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    onStatus && onStatus('checking', '正在检查本地浏览器 AI 模型缓存...');
    let buf = await getCachedModel();
    if (!buf) {
      onStatus && onStatus('downloading', '首次使用正在下载本地 AI 模型 (约 198MB，仅需一次)...', 0);
      buf = await fetchModelWithProgress((loaded, total) => {
        const pct = Math.min(99, Math.round((loaded / total) * 100));
        const mb = (loaded / 1048576).toFixed(1);
        const totMb = (total / 1048576).toFixed(0);
        onStatus && onStatus('downloading', `首次加载 AI 模型：${mb} MB / ${totMb} MB (${pct}%)`, pct);
      });
      onStatus && onStatus('caching', '正在写入浏览器本地缓存，下次打开免下载...', 99);
      await saveCachedModel(buf);
    }

    onStatus && onStatus('initializing', '正在初始化本地 AI 引擎...', 100);
    // Prefer WASM SIMD for broad compatibility (LaMa uses FFT/ops that are rock-solid on WASM)
    ortSession = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    onStatus && onStatus('ready', '浏览器本地 AI 引擎已就绪 (100% 离线推理)', 100);
    return ortSession;
  })().catch((err) => {
    sessionPromise = null;
    throw err;
  });

  return sessionPromise;
}

/**
 * Find tight bounding box of non-transparent pixels in maskCanvas
 */
function getMaskBounds(maskCanvas) {
  const w = maskCanvas.width, h = maskCanvas.height;
  const data = maskCanvas.getContext('2d').getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 10) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY };
}

/**
 * Run LaMa ONNX inpainting in the browser using smart crop-around-mask
 * Preserves 100% original resolution outside the watermark!
 */
export async function inpaintInBrowser(imgCanvas, maskCanvas, onStatus) {
  const session = await ensureModelLoaded(onStatus);
  const W = imgCanvas.width;
  const H = imgCanvas.height;
  const bounds = getMaskBounds(maskCanvas);
  if (!bounds) throw new Error('未检测到涂抹区域');

  // Compute generous square crop around mask (at least 512px or mask + 140px margin)
  const mw = bounds.maxX - bounds.minX + 1;
  const mh = bounds.maxY - bounds.minY + 1;
  const margin = Math.max(128, Math.round(Math.max(mw, mh) * 0.45));
  let cropSize = Math.max(512, Math.max(mw, mh) + margin * 2);
  cropSize = Math.min(cropSize, Math.max(W, H));

  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  let sx = Math.round(cx - cropSize / 2);
  let sy = Math.round(cy - cropSize / 2);
  sx = Math.max(0, Math.min(W - cropSize, sx));
  sy = Math.max(0, Math.min(H - cropSize, sy));
  const sw = Math.min(cropSize, W - sx);
  const sh = Math.min(cropSize, H - sy);

  // Prepare 512x512 RGB & binary mask inputs
  const S = 512;
  const cImg = document.createElement('canvas');
  cImg.width = S; cImg.height = S;
  const cImgCtx = cImg.getContext('2d');
  cImgCtx.drawImage(imgCanvas, sx, sy, sw, sh, 0, 0, S, S);
  const imgData = cImgCtx.getImageData(0, 0, S, S).data;

  const cMask = document.createElement('canvas');
  cMask.width = S; cMask.height = S;
  const cMaskCtx = cMask.getContext('2d');
  cMaskCtx.drawImage(maskCanvas, sx, sy, sw, sh, 0, 0, S, S);
  const maskData = cMaskCtx.getImageData(0, 0, S, S).data;

  const imgTensorData = new Float32Array(1 * 3 * S * S);
  const maskTensorData = new Float32Array(1 * 1 * S * S);
  const plane = S * S;

  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    imgTensorData[i] = imgData[p] / 255.0;
    imgTensorData[plane + i] = imgData[p + 1] / 255.0;
    imgTensorData[2 * plane + i] = imgData[p + 2] / 255.0;
    maskTensorData[i] = maskData[p + 3] > 10 ? 1.0 : 0.0;
  }

  const inputNames = session.inputNames;
  const imgName = inputNames.find(n => n.toLowerCase().includes('image')) || inputNames[0];
  const maskName = inputNames.find(n => n.toLowerCase().includes('mask')) || inputNames[1];

  const feeds = {
    [imgName]: new ort.Tensor('float32', imgTensorData, [1, 3, S, S]),
    [maskName]: new ort.Tensor('float32', maskTensorData, [1, 1, S, S]),
  };

  const results = await session.run(feeds);
  const outTensor = results[session.outputNames[0]];
  const out = outTensor.data;

  // Auto-detect whether output is [0..1] or [0..255]
  let maxVal = 0;
  for (let i = 0; i < out.length; i += 97) {
    if (out[i] > maxVal) maxVal = out[i];
  }
  const scale = maxVal <= 1.5 ? 255.0 : 1.0;

  // Render 512x512 inpainted patch
  const patchCanvas = document.createElement('canvas');
  patchCanvas.width = S; patchCanvas.height = S;
  const pCtx = patchCanvas.getContext('2d');
  const pImg = pCtx.createImageData(S, S);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    pImg.data[p] = Math.max(0, Math.min(255, Math.round(out[i] * scale)));
    pImg.data[p + 1] = Math.max(0, Math.min(255, Math.round(out[plane + i] * scale)));
    pImg.data[p + 2] = Math.max(0, Math.min(255, Math.round(out[2 * plane + i] * scale)));
    pImg.data[p + 3] = 255;
  }
  pCtx.putImageData(pImg, 0, 0);

  // Composite ONLY the masked region (with soft feathered edge) onto full-res original
  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = W; finalCanvas.height = H;
  const fCtx = finalCanvas.getContext('2d');
  fCtx.drawImage(imgCanvas, 0, 0);

  // Build feathered clip mask for the crop area
  const blendCanvas = document.createElement('canvas');
  blendCanvas.width = sw; blendCanvas.height = sh;
  const bCtx = blendCanvas.getContext('2d');
  bCtx.drawImage(patchCanvas, 0, 0, S, S, 0, 0, sw, sh);
  bCtx.globalCompositeOperation = 'destination-in';
  bCtx.filter = 'blur(2px)';
  bCtx.drawImage(maskCanvas, sx, sy, sw, sh, 0, 0, sw, sh);

  fCtx.drawImage(blendCanvas, sx, sy);
  return finalCanvas.toDataURL('image/png');
}

/**
 * Pure-JS Multi-Image Auto Watermark Detection (Cross-Image Sobel Gradient Consistency)
 * Runs 100% locally in the browser in ~150ms!
 */
export function detectCommonWatermarkInBrowser(imgCanvases) {
  const N = Math.min(imgCanvases.length, 12);
  if (N < 2) return { found: false, coverage_pct: 0 };

  const S = 384;
  const gxSum = new Float32Array(S * S);
  const gySum = new Float32Array(S * S);
  const gxSq = new Float32Array(S * S);
  const gySq = new Float32Array(S * S);

  const tmp = document.createElement('canvas');
  tmp.width = S; tmp.height = S;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });

  for (let k = 0; k < N; k++) {
    tctx.drawImage(imgCanvases[k], 0, 0, S, S);
    const rgba = tctx.getImageData(0, 0, S, S).data;
    const gray = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) {
      const p = i * 4;
      gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    }
    for (let y = 1; y < S - 1; y++) {
      for (let x = 1; x < S - 1; x++) {
        const idx = y * S + x;
        const gx =
          -gray[idx - S - 1] + gray[idx - S + 1]
          - 2 * gray[idx - 1] + 2 * gray[idx + 1]
          - gray[idx + S - 1] + gray[idx + S + 1];
        const gy =
          -gray[idx - S - 1] - 2 * gray[idx - S] - gray[idx - S + 1]
          + gray[idx + S - 1] + 2 * gray[idx + S] + gray[idx + S + 1];
        gxSum[idx] += gx;
        gySum[idx] += gy;
        gxSq[idx] += gx * gx;
        gySq[idx] += gy * gy;
      }
    }
  }

  const score = new Float32Array(S * S);
  const meanMagArr = new Float32Array(S * S);
  let maxScore = 0;
  const nonZeroScores = [];

  for (let i = 0; i < S * S; i++) {
    const mgx = gxSum[i] / N;
    const mgy = gySum[i] / N;
    const vgx = Math.max(0, gxSq[i] / N - mgx * mgx);
    const vgy = Math.max(0, gySq[i] / N - mgy * mgy);
    const mm = Math.hypot(mgx, mgy);
    const sm = Math.hypot(Math.sqrt(vgx), Math.sqrt(vgy));
    const sc = mm / (sm * 0.65 + 14.0);
    meanMagArr[i] = mm;
    score[i] = sc;
    if (sc > maxScore) maxScore = sc;
    if (sc > 0.5) nonZeroScores.push(sc);
  }

  if (maxScore < 2.2 || nonZeroScores.length < 10) {
    return { found: false, coverage_pct: 0 };
  }

  nonZeroScores.sort((a, b) => a - b);
  const p99 = nonZeroScores[Math.min(nonZeroScores.length - 1, Math.floor(nonZeroScores.length * 0.99))];
  const thresh = Math.max(2.2, p99 * 0.55);

  const rawBin = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) {
    if (score[i] > thresh && meanMagArr[i] > 24.0) rawBin[i] = 1;
  }

  // Simple 3x3 morphological open (remove isolated speckles)
  const opened = new Uint8Array(S * S);
  for (let y = 1; y < S - 1; y++) {
    for (let x = 1; x < S - 1; x++) {
      const idx = y * S + x;
      if (!rawBin[idx]) continue;
      let nb = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          nb += rawBin[(y + dy) * S + (x + dx)];
        }
      }
      if (nb >= 3) opened[idx] = 1;
    }
  }

  // Dilate by radius 7 to fully cover watermark text strokes
  const R = 7;
  const dilated = new Uint8Array(S * S);
  let count = 0;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (!opened[y * S + x]) continue;
      const y0 = Math.max(0, y - R), y1 = Math.min(S - 1, y + R);
      const x0 = Math.max(0, x - R), x1 = Math.min(S - 1, x + R);
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          if ((yy - y) * (yy - y) + (xx - x) * (xx - x) <= R * R) {
            const j = yy * S + xx;
            if (!dilated[j]) { dilated[j] = 1; count++; }
          }
        }
      }
    }
  }

  const coverage = Number(((count / (S * S)) * 100).toFixed(2));
  if (count < 40 || coverage > 40) {
    return { found: false, coverage_pct: coverage };
  }

  // Render red transparent preview mask PNG
  const outCanvas = document.createElement('canvas');
  outCanvas.width = S; outCanvas.height = S;
  const octx = outCanvas.getContext('2d');
  const oimg = octx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    if (dilated[i]) {
      const p = i * 4;
      oimg.data[p] = 255;
      oimg.data[p + 1] = 42;
      oimg.data[p + 2] = 85;
      oimg.data[p + 3] = 255;
    }
  }
  octx.putImageData(oimg, 0, 0);
  return { found: true, coverage_pct: coverage, mask: outCanvas.toDataURL('image/png') };
}
