/**
 * RealtimeOscilloscope - Real-Time Frequency Spectrum Analyzer
 *
 * Shows overlaid frequency spectra of original and watermarked audio:
 * - X-axis: ~31 Hz (left) → 4000 Hz (right), 8kHz sampling
 * - Original audio spectrum (cyan fill + line)
 * - Watermarked output spectrum (purple line overlay)
 * - Difference highlighting (yellow = watermark perturbation)
 *
 * Optimized: FFT cached (only recomputed on buffer change), 30fps render.
 */

import { useRef, useEffect, useCallback, useMemo } from 'react';

const FFT_SIZE = 1024;
const NUM_BINS = FFT_SIZE / 2;       // 512 bins, 0–4 kHz
const SMOOTHING = 0.5;               // Moderate smoothing, smooth at 30fps
const DC_SKIP_BINS = 4;              // Skip 0-31 Hz (DC + sub-bass noise)
const DISPLAY_BINS = NUM_BINS - DC_SKIP_BINS;  // 508 bins to display
const DB_MIN = -80;
const DB_MAX = 0;
const DB_RANGE = DB_MAX - DB_MIN;
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;  // ~33ms

export function RealtimeOscilloscope({
  inputBuffer,
  outputBuffer,
  isActive = false,
  width = 640,
  height = 340
}) {
  const canvasRef = useRef(null);
  const diffCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const lastFrameTimeRef = useRef(0);

  // Buffer refs — animation loop reads from these without re-triggering effects
  const inputBufferRef = useRef(inputBuffer);
  const outputBufferRef = useRef(outputBuffer);
  inputBufferRef.current = inputBuffer;
  outputBufferRef.current = outputBuffer;

  // FFT cache: only recompute when buffer identity changes
  const lastInputIdRef = useRef(null);
  const lastOutputIdRef = useRef(null);
  const cachedInSpecRef = useRef(new Float32Array(NUM_BINS).fill(DB_MIN));
  const cachedOutSpecRef = useRef(new Float32Array(NUM_BINS).fill(DB_MIN));

  // Smoothed spectrum arrays
  const smoothedInRef = useRef(new Float32Array(NUM_BINS).fill(DB_MIN));
  const smoothedOutRef = useRef(new Float32Array(NUM_BINS).fill(DB_MIN));

  // Pre-compute Hanning window once
  const hannWindow = useMemo(() => {
    const w = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) {
      w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
    }
    return w;
  }, []);

  // Radix-2 Cooley-Tukey FFT (in-place)
  const fft = useCallback((re, im) => {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cRe = 1, cIm = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tRe = re[b] * cRe - im[b] * cIm;
          const tIm = re[b] * cIm + im[b] * cRe;
          re[b] = re[a] - tRe; im[b] = im[a] - tIm;
          re[a] += tRe;        im[a] += tIm;
          const tmp = cRe * wRe - cIm * wIm;
          cIm = cRe * wIm + cIm * wRe;
          cRe = tmp;
        }
      }
    }
  }, []);

  // Compute magnitude spectrum (dB) with DC removal
  const computeSpectrum = useCallback((buffer) => {
    const mags = new Float32Array(NUM_BINS);
    if (!buffer || buffer.length < FFT_SIZE) { mags.fill(DB_MIN); return mags; }

    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);
    const off = Math.max(0, buffer.length - FFT_SIZE);

    // DC removal
    let mean = 0;
    for (let i = 0; i < FFT_SIZE; i++) mean += (buffer[off + i] || 0);
    mean /= FFT_SIZE;

    for (let i = 0; i < FFT_SIZE; i++) {
      re[i] = ((buffer[off + i] || 0) - mean) * hannWindow[i];
    }

    fft(re, im);

    for (let i = 0; i < NUM_BINS; i++) {
      const mag = Math.sqrt(re[i] * re[i] + im[i] * im[i]) / FFT_SIZE;
      mags[i] = 20 * Math.log10(Math.max(mag, 1e-10));
    }
    return mags;
  }, [fft, hannWindow]);

  // ── Render loop ──
  useEffect(() => {
    if (!isActive) {
      if (animationRef.current) { cancelAnimationFrame(animationRef.current); animationRef.current = null; }
      return;
    }

    const canvas = canvasRef.current;
    const diffCanvas = diffCanvasRef.current;
    if (!canvas || !diffCanvas) return;

    const ctx = canvas.getContext('2d');
    const dCtx = diffCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const MAIN_H = Math.round(height * 0.68);
    const DIFF_H = Math.round(height * 0.32);

    canvas.width = width * dpr;   canvas.height = MAIN_H * dpr; ctx.scale(dpr, dpr);
    diffCanvas.width = width * dpr; diffCanvas.height = DIFF_H * dpr; dCtx.scale(dpr, dpr);

    const PL = 48, PR = 16, PT = 28, PB = 30;
    const pW = width - PL - PR, pH = MAIN_H - PT - PB;

    const dPT = 22, dPB = 26;
    const dpH = DIFF_H - dPT - dPB;
    const DIFF_MAX = 6, DIFF_MIN = -6, DIFF_R = DIFF_MAX - DIFF_MIN;

    const binX = (bin) => PL + ((bin - DC_SKIP_BINS) / DISPLAY_BINS) * pW;
    const dbY  = (db) => PT + pH * (1 - (Math.max(DB_MIN, Math.min(DB_MAX, db)) - DB_MIN) / DB_RANGE);
    const dffY = (d)  => dPT + dpH * (1 - (Math.max(DIFF_MIN, Math.min(DIFF_MAX, d)) - DIFF_MIN) / DIFF_R);
    const fqX  = (hz) => { const bin = hz / (8000 / FFT_SIZE); return PL + (Math.max(0, bin - DC_SKIP_BINS) / DISPLAY_BINS) * pW; };

    const freqs = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000];
    const binW = pW / DISPLAY_BINS;  // Width per bin (~1.1px)

    // Pre-create gradient objects (reused every frame instead of recreated)
    const gradientIn = ctx.createLinearGradient(0, PT, 0, PT + pH);
    gradientIn.addColorStop(0, 'rgba(56,189,248,0.35)');
    gradientIn.addColorStop(1, 'rgba(56,189,248,0.03)');

    const render = (now) => {
      animationRef.current = requestAnimationFrame(render);

      // Throttle to TARGET_FPS
      if (now - lastFrameTimeRef.current < FRAME_INTERVAL) return;
      lastFrameTimeRef.current = now;

      // ── Recompute FFT only when buffer has changed ──
      const curIn = inputBufferRef.current;
      const curOut = outputBufferRef.current;
      if (curIn !== lastInputIdRef.current)  { cachedInSpecRef.current  = computeSpectrum(curIn);  lastInputIdRef.current  = curIn; }
      if (curOut !== lastOutputIdRef.current) { cachedOutSpecRef.current = computeSpectrum(curOut); lastOutputIdRef.current = curOut; }

      const inSpec = cachedInSpecRef.current;
      const outSpec = cachedOutSpecRef.current;

      // Smoothing
      const sIn = smoothedInRef.current, sOut = smoothedOutRef.current;
      for (let i = 0; i < NUM_BINS; i++) {
        sIn[i]  = SMOOTHING * sIn[i]  + (1 - SMOOTHING) * inSpec[i];
        sOut[i] = SMOOTHING * sOut[i] + (1 - SMOOTHING) * outSpec[i];
      }

      // ═══════ MAIN CANVAS ═══════
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, width, MAIN_H);

      // Grid
      ctx.strokeStyle = '#1a2540'; ctx.lineWidth = 0.5;
      for (let db = DB_MIN; db <= DB_MAX; db += 10) {
        const y = dbY(db);
        ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(PL + pW, y); ctx.stroke();
        ctx.fillStyle = '#475569'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
        ctx.fillText(`${db}`, PL - 5, y + 3);
      }
      for (const f of freqs) {
        const x = fqX(f); if (x < PL) continue;
        ctx.beginPath(); ctx.moveTo(x, PT); ctx.lineTo(x, PT + pH); ctx.stroke();
        ctx.fillStyle = '#475569'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
        ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x, PT + pH + 14);
      }
      ctx.fillStyle = '#64748b'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Frequency (Hz)', PL + pW / 2, PT + pH + 26);
      ctx.save(); ctx.translate(11, PT + pH / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = '#64748b'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Magnitude (dB)', 0, 0); ctx.restore();

      // Diff highlight bars (508 individual bins)
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const d = sOut[i] - sIn[i]; if (Math.abs(d) < 0.3) continue;
        const x = binX(i), y1 = dbY(sIn[i]), y2 = dbY(sOut[i]);
        const a = Math.min(0.75, Math.min(Math.abs(d), 20) / 12);
        ctx.fillStyle = d > 0 ? `rgba(250,204,21,${a})` : `rgba(248,113,113,${a})`;
        ctx.fillRect(x, Math.min(y1, y2), Math.max(binW, 1), Math.abs(y2 - y1) || 1);
      }

      // Input fill (uses pre-created gradient)
      ctx.beginPath(); ctx.moveTo(binX(DC_SKIP_BINS), dbY(DB_MIN));
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) ctx.lineTo(binX(i) + binW / 2, dbY(sIn[i]));
      ctx.lineTo(binX(NUM_BINS - 1) + binW, dbY(DB_MIN)); ctx.closePath();
      ctx.fillStyle = gradientIn; ctx.fill();

      // Input line (508 points)
      ctx.beginPath();
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const x = binX(i) + binW / 2, y = dbY(sIn[i]);
        i === DC_SKIP_BINS ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(56,189,248,0.85)'; ctx.lineWidth = 1.5; ctx.stroke();

      // Output line (508 points)
      ctx.beginPath();
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const x = binX(i) + binW / 2, y = dbY(sOut[i]);
        i === DC_SKIP_BINS ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(192,132,252,0.95)'; ctx.lineWidth = 2; ctx.stroke();

      // Legend
      ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('Real-Time Frequency Spectrum', PL, 16);
      const lx = PL + pW - 250;
      ctx.fillStyle = 'rgba(56,189,248,0.9)';  ctx.fillRect(lx, 7, 14, 4);
      ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif'; ctx.fillText('Original', lx + 18, 13);
      ctx.fillStyle = 'rgba(192,132,252,0.9)'; ctx.fillRect(lx + 80, 7, 14, 4);
      ctx.fillStyle = '#94a3b8'; ctx.fillText('Watermarked', lx + 98, 13);
      ctx.fillStyle = 'rgba(250,204,21,0.7)';  ctx.fillRect(lx + 180, 7, 14, 4);
      ctx.fillStyle = '#94a3b8'; ctx.fillText('Diff', lx + 198, 13);

      // ═══════ DIFF CANVAS ═══════
      dCtx.fillStyle = '#0b1120'; dCtx.fillRect(0, 0, width, DIFF_H);

      // Zero line
      const zy = dffY(0);
      dCtx.strokeStyle = '#334155'; dCtx.lineWidth = 1;
      dCtx.setLineDash([4, 3]);
      dCtx.beginPath(); dCtx.moveTo(PL, zy); dCtx.lineTo(PL + pW, zy); dCtx.stroke();
      dCtx.setLineDash([]);

      // +/- grid
      for (const d of [-4, -2, 2, 4]) {
        const y = dffY(d);
        dCtx.strokeStyle = '#1a2540'; dCtx.lineWidth = 0.5;
        dCtx.beginPath(); dCtx.moveTo(PL, y); dCtx.lineTo(PL + pW, y); dCtx.stroke();
        dCtx.fillStyle = '#475569'; dCtx.font = '8px monospace'; dCtx.textAlign = 'right';
        dCtx.fillText(`${d > 0 ? '+' : ''}${d}`, PL - 4, y + 3);
      }
      for (const f of freqs) {
        const x = fqX(f); if (x < PL) continue;
        dCtx.strokeStyle = '#1a2540'; dCtx.lineWidth = 0.5;
        dCtx.beginPath(); dCtx.moveTo(x, dPT); dCtx.lineTo(x, dPT + dpH); dCtx.stroke();
        dCtx.fillStyle = '#475569'; dCtx.font = '8px monospace'; dCtx.textAlign = 'center';
        dCtx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x, dPT + dpH + 12);
      }
      dCtx.fillStyle = '#64748b'; dCtx.font = '9px sans-serif'; dCtx.textAlign = 'center';
      dCtx.fillText('Hz', PL + pW / 2, dPT + dpH + 22);

      // Diff bars (508 individual bins)
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const d = sOut[i] - sIn[i]; if (Math.abs(d) < 0.15) continue;
        const x = binX(i), y = dffY(d);
        const a = 0.3 + (Math.min(Math.abs(d), 6) / 6) * 0.65;
        dCtx.fillStyle = d > 0 ? `rgba(250,204,21,${a})` : `rgba(248,113,113,${a})`;
        dCtx.fillRect(x, Math.min(zy, y), Math.max(binW, 1), Math.abs(zy - y) || 1);
      }

      // Diff line (508 points)
      dCtx.beginPath();
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const d = sOut[i] - sIn[i];
        const x = binX(i) + binW / 2, y = dffY(d);
        i === DC_SKIP_BINS ? dCtx.moveTo(x, y) : dCtx.lineTo(x, y);
      }
      dCtx.strokeStyle = 'rgba(250,204,21,0.9)'; dCtx.lineWidth = 1.5; dCtx.stroke();

      // Title + stats
      dCtx.fillStyle = '#e2e8f0'; dCtx.font = 'bold 10px sans-serif'; dCtx.textAlign = 'left';
      dCtx.fillText('Watermark Perturbation: encoder(audio, msg) \u2212 audio', PL, 14);

      let sumD = 0, maxD = 0, peakBin = DC_SKIP_BINS;
      for (let i = DC_SKIP_BINS; i < NUM_BINS; i++) {
        const d = Math.abs(sOut[i] - sIn[i]); sumD += d;
        if (d > maxD) { maxD = d; peakBin = i; }
      }
      const peakHz = Math.round((peakBin / NUM_BINS) * 4000);
      dCtx.fillStyle = '#94a3b8'; dCtx.font = '9px monospace'; dCtx.textAlign = 'right';
      dCtx.fillText(
        `avg \u0394${(sumD / DISPLAY_BINS).toFixed(1)}dB  peak \u0394${maxD.toFixed(1)}dB @ ${peakHz}Hz`,
        PL + pW, 14
      );
    };

    animationRef.current = requestAnimationFrame(render);
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
  }, [isActive, width, height, computeSpectrum]);

  // Reset on streaming start
  useEffect(() => {
    if (isActive) {
      smoothedInRef.current.fill(DB_MIN);
      smoothedOutRef.current.fill(DB_MIN);
      lastInputIdRef.current = null;
      lastOutputIdRef.current = null;
      lastFrameTimeRef.current = 0;
    }
  }, [isActive]);

  const MAIN_H = Math.round(height * 0.68);
  const DIFF_H = Math.round(height * 0.32);

  return (
    <div className="space-y-1">
      <div className="relative">
        <canvas ref={canvasRef}
          style={{ width: `${width}px`, height: `${MAIN_H}px` }}
          className="w-full rounded-t-lg border border-gray-700/40" />
        {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/70 rounded-t-lg backdrop-blur-sm">
            <div className="text-center">
              <svg className="w-10 h-10 mx-auto text-gray-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
              <p className="text-gray-500 text-sm">스트리밍 시작 시 실시간 주파수 스펙트럼이 표시됩니다</p>
              <p className="text-gray-600 text-xs mt-1">~31 Hz (좌) → 4,000 Hz (우) | 8 kHz Sampling</p>
            </div>
          </div>
        )}
      </div>
      <div className="relative">
        <canvas ref={diffCanvasRef}
          style={{ width: `${width}px`, height: `${DIFF_H}px` }}
          className="w-full rounded-b-lg border border-t-0 border-gray-700/40" />
        {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/70 rounded-b-lg backdrop-blur-sm">
            <p className="text-gray-600 text-xs">워터마크에 의한 주파수 변화량 (Output - Input)</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default RealtimeOscilloscope;
