/**
 * Progressive Detection Component
 * 
 * Animates the detection process to look real-time:
 * - Processes audio in small chunks
 * - Reveals bits progressively as "playback" advances
 * - Shows a scanning animation effect
 */

import { useState, useCallback, useRef, useEffect } from 'react';

const FRAME_MS = 40;  // 40ms per frame
const PAYLOAD_LENGTH = 128;
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;  // 40ms @ 8kHz

export function ProgressiveDetection({
  audioData,
  onRunDecoder,
  onProgressUpdate,
  onComplete
}) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [revealedBits, setRevealedBits] = useState(new Float32Array(128).fill(0.5));
  const [currentFrame, setCurrentFrame] = useState(0);
  const [stats, setStats] = useState({ framesProcessed: 0, elapsed: 0 });

  const animationRef = useRef(null);
  const startTimeRef = useRef(null);
  const bitsAccumulatorRef = useRef(null);

  // Total frames in audio
  const totalFrames = audioData ? Math.floor(audioData.length / FRAME_SAMPLES) : 0;
  const audioDuration = audioData ? audioData.length / SAMPLE_RATE : 0;

  // Start progressive detection
  const startDetection = useCallback(async () => {
    if (!audioData || !onRunDecoder || isRunning) return;

    setIsRunning(true);
    setProgress(0);
    setCurrentFrame(0);
    setRevealedBits(new Float32Array(128).fill(0.5));

    // Initialize accumulator for progressive averaging
    bitsAccumulatorRef.current = {
      sums: new Float32Array(128).fill(0),
      counts: new Float32Array(128).fill(0)
    };

    startTimeRef.current = performance.now();

    // Process in chunks, with animation delay
    const chunkSize = Math.max(1, Math.floor(totalFrames / 50));  // ~50 updates
    let processedFrames = 0;

    const processNextChunk = async () => {
      if (processedFrames >= totalFrames) {
        // Final result
        const finalBits = new Float32Array(128);
        const acc = bitsAccumulatorRef.current;
        for (let i = 0; i < 128; i++) {
          finalBits[i] = acc.counts[i] > 0 ? acc.sums[i] / acc.counts[i] : 0.5;
        }

        setRevealedBits(finalBits);
        setProgress(100);
        setIsRunning(false);

        const elapsed = performance.now() - startTimeRef.current;
        setStats({ framesProcessed: processedFrames, elapsed });

        onComplete?.(finalBits);
        return;
      }

      // Get chunk of audio
      const startSample = processedFrames * FRAME_SAMPLES;
      const endSample = Math.min((processedFrames + chunkSize) * FRAME_SAMPLES, audioData.length);
      const chunk = audioData.slice(startSample, endSample);

      try {
        // Run decoder on chunk
        const result = await onRunDecoder(chunk);

        if (result && result.frameProbs) {
          // Update accumulator with new frame results
          const acc = bitsAccumulatorRef.current;
          for (let f = 0; f < result.frameProbs.length; f++) {
            const bitIdx = (processedFrames + f) % PAYLOAD_LENGTH;
            acc.sums[bitIdx] += result.frameProbs[f];
            acc.counts[bitIdx] += 1;
          }

          // Calculate current revealed bits
          const revealed = new Float32Array(128);
          for (let i = 0; i < 128; i++) {
            revealed[i] = acc.counts[i] > 0 ? acc.sums[i] / acc.counts[i] : 0.5;
          }

          setRevealedBits(revealed);
          onProgressUpdate?.(revealed, processedFrames);
        }

        processedFrames += chunkSize;
        setCurrentFrame(processedFrames);
        setProgress(Math.min(100, (processedFrames / totalFrames) * 100));

        // Schedule next chunk with animation delay
        const targetTime = (processedFrames / totalFrames) * audioDuration * 1000;
        const actualElapsed = performance.now() - startTimeRef.current;
        const delay = Math.max(10, targetTime * 0.5 - actualElapsed);  // 2x speed

        animationRef.current = setTimeout(processNextChunk, delay);

      } catch (err) {
        console.error('Detection error:', err);
        setIsRunning(false);
      }
    };

    processNextChunk();

  }, [audioData, onRunDecoder, isRunning, totalFrames, audioDuration, onProgressUpdate, onComplete]);

  // Stop detection
  const stopDetection = useCallback(() => {
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
    setIsRunning(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, []);

  // Get bit style
  const getBitStyle = (prob, revealed) => {
    if (!revealed) return { backgroundColor: '#374151', color: '#9CA3AF' }; // Gray-700 bg, Gray-400 text

    // Gradient from Black (0) to White (1)
    // Prob 0 -> lightness 0% (Black)
    // Prob 1 -> lightness 100% (White)
    // We clamp slightly to avoid pure pitch black if desired, but user asked for black/white.

    // Visualize uncertainty with grayscale
    // Confidence = distance from 0.5
    // However, user specifically asked for 1=White, 0=Black.
    // Let's map prob directly to lightness.
    const lightness = Math.floor(prob * 100);
    const backgroundColor = `hsl(0, 0%, ${lightness}%)`;

    // Text color: White for dark backgrounds (low prob), Black for light backgrounds (high prob)
    const color = prob < 0.5 ? '#FFFFFF' : '#000000';

    return { backgroundColor, color };
  };

  // Calculate how many bits are "revealed" based on progress
  const revealedCount = Math.min(128, Math.floor((progress / 100) * 128) + 1);

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <span className="text-lg">🔍</span>
          Progressive Detection
        </h3>

        {isRunning && (
          <span className="flex items-center gap-1 text-xs text-green-400">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            Analyzing...
          </span>
        )}
      </div>

      {/* Progress Bar */}
      <div>
        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
          <span>Frame {currentFrame} / {totalFrames}</span>
          <span>{progress.toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-surface rounded-full overflow-hidden relative">
          <div
            className="h-full bg-gradient-to-r from-gray-600 to-gray-200 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
          {isRunning && (
            <div
              className="absolute top-0 bottom-0 w-1 bg-white shadow-lg animate-pulse"
              style={{ left: `${progress}%` }}
            />
          )}
        </div>
      </div>

      {/* Bit Matrix with progressive reveal */}
      <div className="grid grid-cols-16 gap-0.5">
        {Array.from({ length: 128 }).map((_, i) => {
          const isRevealed = i < revealedCount;
          const prob = revealedBits[i];
          const style = getBitStyle(prob, isRevealed);

          return (
            <div
              key={i}
              className={`aspect-square rounded-sm flex items-center justify-center text-[8px] font-mono font-bold
                transition-all duration-200
                ${isRevealed ? '' : 'opacity-30'}
                ${i === revealedCount - 1 && isRunning ? 'ring-1 ring-white animate-pulse' : ''}`}
              style={style}
            >
              {isRevealed ? (prob > 0.5 ? '1' : '0') : '?'}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#ffffff' }} />
          1 (White)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#808080' }} />
          Uncertain (Gray)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm border border-gray-600" style={{ backgroundColor: '#000000' }} />
          0 (Black)
        </span>
      </div>

      {/* Control Button */}
      {!isRunning ? (
        <button
          onClick={startDetection}
          disabled={!audioData}
          className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
            ${audioData
              ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:opacity-90 glow'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
          실시간 탐지 시작
        </button>
      ) : (
        <button
          onClick={stopDetection}
          className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 
                   rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          탐지 중지
        </button>
      )}

      {/* Stats */}
      {stats.elapsed > 0 && !isRunning && (
        <div className="flex justify-center gap-4 text-[10px] text-gray-500">
          <span>{stats.framesProcessed} 프레임 처리</span>
          <span>|</span>
          <span>{(stats.elapsed / 1000).toFixed(2)}초 소요</span>
          <span>|</span>
          <span>{(stats.framesProcessed / (stats.elapsed / 1000)).toFixed(0)} fps</span>
        </div>
      )}
    </div>
  );
}

export default ProgressiveDetection;
