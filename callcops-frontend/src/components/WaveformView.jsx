/**
 * WaveformView Component
 * 
 * Displays audio waveform using wavesurfer.js with:
 * - Playback controls (play/pause)
 * - Real-time playhead cursor
 * - Position-based detection callback for synced bit matrix updates
 */

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import WaveSurfer from 'wavesurfer.js';

// Frame-wise constants
const FRAME_SAMPLES = 320;  // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;

export const WaveformView = forwardRef(function WaveformView({
  audioData,
  bitProbs,       // [128] aggregated bit probabilities
  frameProbs,     // [num_frames] raw frame probabilities (optional)
  sampleRate = 8000,
  onTimeUpdate,
  onPlayStateChange,
}, ref) {
  const containerRef = useRef(null);
  const wavesurferRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    play: () => wavesurferRef.current?.play(),
    pause: () => wavesurferRef.current?.pause(),
    stop: () => {
      wavesurferRef.current?.pause();
      wavesurferRef.current?.seekTo(0);
    },
    seekTo: (progress) => wavesurferRef.current?.seekTo(progress),
    getCurrentTime: () => wavesurferRef.current?.getCurrentTime() || 0,
    getDuration: () => wavesurferRef.current?.getDuration() || 0,
  }));

  // Store callbacks in refs to avoid re-initialization
  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onPlayStateChangeRef = useRef(onPlayStateChange);

  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdate;
    onPlayStateChangeRef.current = onPlayStateChange;
  }, [onTimeUpdate, onPlayStateChange]);

  // Initialize WaveSurfer (only once on mount)
  useEffect(() => {
    if (!containerRef.current) return;

    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#6366f1',
      progressColor: '#22c55e',
      cursorColor: '#ffffff',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 120,
      normalize: true,
    });

    wavesurferRef.current = wavesurfer;

    wavesurfer.on('ready', () => {
      setIsReady(true);
      setDuration(wavesurfer.getDuration());
    });

    wavesurfer.on('play', () => {
      setIsPlaying(true);
      onPlayStateChangeRef.current?.(true);
    });

    wavesurfer.on('pause', () => {
      setIsPlaying(false);
      onPlayStateChangeRef.current?.(false);
    });

    wavesurfer.on('finish', () => {
      setIsPlaying(false);
      onPlayStateChangeRef.current?.(false);
    });

    wavesurfer.on('audioprocess', (time) => {
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
    });

    wavesurfer.on('seeking', (progress) => {
      const time = progress * wavesurfer.getDuration();
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
    });

    return () => {
      wavesurfer.destroy();
    };
  }, []); // Empty dependency - initialize only once!

  // Load audio data
  useEffect(() => {
    if (!wavesurferRef.current || !audioData) return;

    // Create AudioBuffer from Float32Array
    const wavBlob = new Blob([encodeWav(audioData, sampleRate)], { type: 'audio/wav' });
    wavesurferRef.current.loadBlob(wavBlob);
    setIsPlaying(false);
    setCurrentTime(0);
  }, [audioData, sampleRate]);

  // Handle play/pause toggle
  const togglePlayPause = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  }, []);

  // Format time display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
  };

  // Calculate actual number of frames based on audio length
  const numFrames = audioData ? Math.floor(audioData.length / FRAME_SAMPLES) : 0;

  // Calculate current frame index based on playback position (40ms per frame)
  const getCurrentFrameIndex = () => {
    if (!duration || duration === 0) return -1;
    const currentSample = Math.floor(currentTime * sampleRate);
    return Math.floor(currentSample / FRAME_SAMPLES);
  };

  const currentFrameIndex = getCurrentFrameIndex();

  // Get cyclic bit index for a frame
  const getCyclicBitIndex = (frameIndex) => frameIndex % PAYLOAD_LENGTH;

  // Calculate zone colors based on frame probabilities
  const getZoneColors = () => {
    // Use frameProbs if available, otherwise fall back to bitProbs with cyclic mapping
    if (frameProbs && frameProbs.length > 0) {
      // Frame-based display (actual 40ms frames)
      return Array.from(frameProbs).map((prob, index) => {
        const confidence = Math.abs(prob - 0.5) * 2;
        const isActive = index <= currentFrameIndex;
        const isCurrent = index === currentFrameIndex;

        let color;
        if (prob > 0.5) {
          color = `rgba(34, 197, 94, ${isActive ? 0.2 + confidence * 0.4 : 0.05})`;
        } else {
          color = `rgba(239, 68, 68, ${isActive ? 0.2 + confidence * 0.4 : 0.05})`;
        }

        return { bg: color, active: isActive, current: isCurrent };
      });
    }

    // Fallback: Use bitProbs with cyclic mapping to frames
    if (!bitProbs || bitProbs.length !== 128 || numFrames === 0) {
      // Default: show actual frame count as neutral zones
      const count = numFrames > 0 ? Math.min(numFrames, 200) : 128;  // Limit for performance
      return Array(count).fill({ bg: 'rgba(99, 102, 241, 0.1)', active: false });
    }

    // Map frames to cyclic bit indices
    const displayFrames = Math.min(numFrames, 200);  // Limit for performance
    return Array(displayFrames).fill(0).map((_, frameIdx) => {
      const bitIdx = getCyclicBitIndex(frameIdx);
      const prob = bitProbs[bitIdx];
      const confidence = Math.abs(prob - 0.5) * 2;
      const isActive = frameIdx <= currentFrameIndex;
      const isCurrent = frameIdx === currentFrameIndex;

      let color;
      if (prob > 0.5) {
        color = `rgba(34, 197, 94, ${isActive ? 0.2 + confidence * 0.4 : 0.05})`;
      } else {
        color = `rgba(239, 68, 68, ${isActive ? 0.2 + confidence * 0.4 : 0.05})`;
      }

      return { bg: color, active: isActive, current: isCurrent };
    });
  };

  const zoneData = getZoneColors();

  return (
    <div className="relative w-full space-y-3">
      {/* Waveform container */}
      <div className="relative">
        <div
          ref={containerRef}
          className="w-full bg-surface/50 rounded-lg overflow-hidden cursor-pointer"
          onClick={togglePlayPause}
        />

        {/* Bit zones overlay */}
        {audioData && (
          <div className="absolute inset-0 flex pointer-events-none rounded-lg overflow-hidden">
            {zoneData.map((zone, index) => (
              <div
                key={index}
                className={`flex-1 border-r border-white/5 transition-all duration-100
                  ${zone.current ? 'ring-1 ring-white/50' : ''}`}
                style={{ backgroundColor: zone.bg }}
              />
            ))}
          </div>
        )}

        {/* Empty state */}
        {!audioData && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <div className="flex flex-col items-center justify-center gap-2">
              <svg
                className="w-10 h-10 opacity-50"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                />
              </svg>
              <div className="text-center">
                <p className="text-sm font-medium">No audio loaded</p>
                <p className="text-xs opacity-70">Record or upload to visualize</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Playback controls */}
      {audioData && isReady && (
        <div className="flex items-center gap-4">
          {/* Play/Pause button */}
          <button
            onClick={togglePlayPause}
            className="w-10 h-10 rounded-full bg-primary hover:bg-primary/80 
                     flex items-center justify-center transition-colors glow"
          >
            {isPlaying ? (
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          {/* Time display */}
          <div className="flex-1">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>

            {/* Progress bar */}
            <div className="h-1.5 bg-surface rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-green-500 transition-all duration-100"
                style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
              />
            </div>
          </div>

          {/* Current frame indicator */}
          <div className="text-right">
            <p className="text-xs text-gray-500">Current Frame</p>
            <p className="text-lg font-mono font-bold text-primary">
              {currentFrameIndex >= 0 ? currentFrameIndex + 1 : '—'}
              <span className="text-xs text-gray-500 font-normal"> / {numFrames || '—'}</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
});

/**
 * Encode Float32Array to WAV format
 */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Audio data
  const offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

export default WaveformView;
