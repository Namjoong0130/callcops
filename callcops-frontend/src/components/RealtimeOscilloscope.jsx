/**
 * Real-Time Oscilloscope Component
 * 
 * Displays scrolling waveform and spectrogram in real-time:
 * - Top: Input waveform (blue)
 * - Middle: Output waveform (purple/watermarked)
 * - Bottom: Spectrogram showing frequency differences
 */

import { useRef, useEffect, useCallback } from 'react';

const SAMPLE_RATE = 8000;
const FFT_SIZE = 256;
const SCROLL_SPEED = 2;  // pixels per frame

export function RealtimeOscilloscope({ 
  inputBuffer,      // Float32Array of recent input samples
  outputBuffer,     // Float32Array of recent output samples
  isActive = false,
  width = 400,
  height = 200
}) {
  const canvasRef = useRef(null);
  const spectrogramCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const inputHistoryRef = useRef([]);
  const outputHistoryRef = useRef([]);
  const spectrogramHistoryRef = useRef([]);
  
  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current;
    const spectroCanvas = spectrogramCanvasRef.current;
    if (!canvas || !spectroCanvas) return;
    
    const ctx = canvas.getContext('2d');
    const spectroCtx = spectroCanvas.getContext('2d');
    
    // Clear with dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    spectroCtx.fillStyle = '#0f172a';
    spectroCtx.fillRect(0, 0, spectroCanvas.width, spectroCanvas.height);
    
    // Initialize history arrays
    inputHistoryRef.current = new Array(Math.ceil(width / SCROLL_SPEED)).fill(0);
    outputHistoryRef.current = new Array(Math.ceil(width / SCROLL_SPEED)).fill(0);
    spectrogramHistoryRef.current = new Array(Math.ceil(width / SCROLL_SPEED)).fill(null);
    
  }, [width, height]);
  
  // Calculate FFT for spectrogram
  const calculateFFT = useCallback((samples) => {
    if (!samples || samples.length < FFT_SIZE) return null;
    
    // Simple DFT for visualization (not optimized, but works for demo)
    const magnitudes = new Float32Array(FFT_SIZE / 2);
    
    for (let k = 0; k < FFT_SIZE / 2; k++) {
      let real = 0, imag = 0;
      for (let n = 0; n < FFT_SIZE; n++) {
        const sample = samples[samples.length - FFT_SIZE + n] || 0;
        const angle = (2 * Math.PI * k * n) / FFT_SIZE;
        real += sample * Math.cos(angle);
        imag -= sample * Math.sin(angle);
      }
      magnitudes[k] = Math.sqrt(real * real + imag * imag) / FFT_SIZE;
    }
    
    return magnitudes;
  }, []);
  
  // Get color for spectrogram value
  const getSpectrogramColor = (value, isOutput = false) => {
    const intensity = Math.min(1, value * 50);  // Scale up for visibility
    
    if (isOutput) {
      // Purple gradient for output
      const r = Math.floor(intensity * 147);
      const g = Math.floor(intensity * 51);
      const b = Math.floor(intensity * 234);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Blue gradient for input
      const r = Math.floor(intensity * 59);
      const g = Math.floor(intensity * 130);
      const b = Math.floor(intensity * 246);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };
  
  // Animation loop
  useEffect(() => {
    if (!isActive) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }
    
    const canvas = canvasRef.current;
    const spectroCanvas = spectrogramCanvasRef.current;
    if (!canvas || !spectroCanvas) return;
    
    const ctx = canvas.getContext('2d');
    const spectroCtx = spectroCanvas.getContext('2d');
    const waveHeight = height / 2;
    
    const render = () => {
      // Scroll existing content left
      const imageData = ctx.getImageData(SCROLL_SPEED, 0, canvas.width - SCROLL_SPEED, canvas.height);
      ctx.putImageData(imageData, 0, 0);
      
      const spectroImageData = spectroCtx.getImageData(SCROLL_SPEED, 0, spectroCanvas.width - SCROLL_SPEED, spectroCanvas.height);
      spectroCtx.putImageData(spectroImageData, 0, 0);
      
      // Clear right edge
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(canvas.width - SCROLL_SPEED, 0, SCROLL_SPEED, canvas.height);
      
      spectroCtx.fillStyle = '#0f172a';
      spectroCtx.fillRect(spectroCanvas.width - SCROLL_SPEED, 0, SCROLL_SPEED, spectroCanvas.height);
      
      // Get current samples
      const inputSample = inputBuffer && inputBuffer.length > 0 
        ? inputBuffer[inputBuffer.length - 1] || 0 
        : 0;
      const outputSample = outputBuffer && outputBuffer.length > 0 
        ? outputBuffer[outputBuffer.length - 1] || 0 
        : 0;
      
      // Draw input waveform (top half, blue)
      const inputY = waveHeight / 2 + inputSample * (waveHeight / 2 - 10);
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(canvas.width - SCROLL_SPEED, inputY - 1, SCROLL_SPEED, 3);
      
      // Center line for input
      ctx.fillStyle = '#374151';
      ctx.fillRect(canvas.width - SCROLL_SPEED, waveHeight / 2, SCROLL_SPEED, 1);
      
      // Draw output waveform (bottom half, purple)
      const outputY = waveHeight + waveHeight / 2 + outputSample * (waveHeight / 2 - 10);
      ctx.fillStyle = '#a855f7';
      ctx.fillRect(canvas.width - SCROLL_SPEED, outputY - 1, SCROLL_SPEED, 3);
      
      // Center line for output
      ctx.fillStyle = '#374151';
      ctx.fillRect(canvas.width - SCROLL_SPEED, waveHeight + waveHeight / 2, SCROLL_SPEED, 1);
      
      // Separator line
      ctx.fillStyle = '#4b5563';
      ctx.fillRect(canvas.width - SCROLL_SPEED, waveHeight - 1, SCROLL_SPEED, 2);
      
      // Calculate and draw spectrogram
      if (inputBuffer && inputBuffer.length >= FFT_SIZE) {
        const inputFFT = calculateFFT(Array.from(inputBuffer.slice(-FFT_SIZE)));
        const outputFFT = outputBuffer && outputBuffer.length >= FFT_SIZE 
          ? calculateFFT(Array.from(outputBuffer.slice(-FFT_SIZE)))
          : null;
        
        if (inputFFT) {
          const spectroHeight = spectroCanvas.height / 2;
          const binHeight = spectroHeight / (FFT_SIZE / 2);
          
          // Draw input spectrogram (top)
          for (let i = 0; i < FFT_SIZE / 2; i++) {
            spectroCtx.fillStyle = getSpectrogramColor(inputFFT[i], false);
            spectroCtx.fillRect(
              spectroCanvas.width - SCROLL_SPEED,
              spectroHeight - (i + 1) * binHeight,
              SCROLL_SPEED,
              binHeight
            );
          }
          
          // Draw output spectrogram (bottom)
          if (outputFFT) {
            for (let i = 0; i < FFT_SIZE / 2; i++) {
              spectroCtx.fillStyle = getSpectrogramColor(outputFFT[i], true);
              spectroCtx.fillRect(
                spectroCanvas.width - SCROLL_SPEED,
                spectroHeight + spectroHeight - (i + 1) * binHeight,
                SCROLL_SPEED,
                binHeight
              );
            }
          }
          
          // Separator
          spectroCtx.fillStyle = '#4b5563';
          spectroCtx.fillRect(spectroCanvas.width - SCROLL_SPEED, spectroHeight - 1, SCROLL_SPEED, 2);
        }
      }
      
      animationRef.current = requestAnimationFrame(render);
    };
    
    animationRef.current = requestAnimationFrame(render);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, inputBuffer, outputBuffer, height, calculateFFT]);
  
  return (
    <div className="space-y-2">
      {/* Waveform Oscilloscope */}
      <div className="relative">
        <div className="absolute top-1 left-2 flex gap-3 text-[10px] z-10">
          <span className="text-blue-400">● Input</span>
          <span className="text-purple-400">● Output</span>
        </div>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="w-full rounded-lg border border-gray-700/50"
          style={{ imageRendering: 'pixelated' }}
        />
        {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/50 rounded-lg">
            <p className="text-gray-500 text-sm">스트리밍을 시작하면 파형이 표시됩니다</p>
          </div>
        )}
      </div>
      
      {/* Spectrogram */}
      <div className="relative">
        <div className="absolute top-1 left-2 flex gap-3 text-[10px] z-10">
          <span className="text-blue-400">● Input FFT</span>
          <span className="text-purple-400">● Output FFT</span>
        </div>
        <canvas
          ref={spectrogramCanvasRef}
          width={width}
          height={height / 2}
          className="w-full rounded-lg border border-gray-700/50"
          style={{ imageRendering: 'pixelated' }}
        />
        <div className="absolute right-2 top-1 bottom-1 flex flex-col justify-between text-[8px] text-gray-500">
          <span>4kHz</span>
          <span>0Hz</span>
        </div>
      </div>
      
      {/* Legend */}
      <div className="flex justify-between text-[10px] text-gray-500 px-1">
        <span>← Time</span>
        <span>스펙트럼 변화로 워터마크 확인</span>
        <span>Now →</span>
      </div>
    </div>
  );
}

export default RealtimeOscilloscope;
