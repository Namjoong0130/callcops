/**
 * RealtimeEmbedDemo Component
 *
 * Real-time watermark embedding demo with frequency spectrum visualization.
 * Shows live input vs watermarked output spectra with difference highlighting.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { RealtimeOscilloscope } from './RealtimeOscilloscope';

// Constants matching the encoder
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;  // 40ms @ 8kHz
const CHUNK_SIZE = FRAME_SAMPLES * 4;  // 160ms chunks for low latency

export function RealtimeEmbedDemo({ onEmbed, isModelReady = false }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [message, setMessage] = useState(null);
  const [stats, setStats] = useState({
    latencyMs: 0,
    chunksProcessed: 0,
    totalSamples: 0,
    avgLatency: 0
  });

  // Buffers for frequency spectrum visualization
  const [inputBuffer, setInputBuffer] = useState(new Float32Array(2048));
  const [outputBuffer, setOutputBuffer] = useState(new Float32Array(2048));

  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const latenciesRef = useRef([]);
  const isStreamingRef = useRef(false);
  const isProcessingRef = useRef(false);  // Lock to prevent overlapping inference

  // Generate random 128-bit message
  const generateMessage = useCallback(() => {
    const msg = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      msg[i] = Math.random() > 0.5 ? 1 : 0;
    }
    setMessage(msg);
    return msg;
  }, []);

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!onEmbed || !isModelReady) return;

    const currentMessage = message || generateMessage();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      mediaStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      // bufferSize=2048 → 256ms at 8kHz. Good balance of latency vs callback overhead.
      const bufferSize = 2048;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      let accumulatedSamples = new Float32Array(0);

      // NOT async — fire-and-forget inference behind a lock
      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const outputData = e.outputBuffer.getChannelData(0);

        // Always pass through audio (low-latency monitoring)
        outputData.set(inputData.slice(0, outputData.length));

        // Accumulate samples
        const newAccumulated = new Float32Array(accumulatedSamples.length + inputData.length);
        newAccumulated.set(accumulatedSamples);
        newAccumulated.set(inputData, accumulatedSamples.length);
        accumulatedSamples = newAccumulated;

        // Only start inference if not already processing (prevents overlap)
        if (accumulatedSamples.length >= CHUNK_SIZE && !isProcessingRef.current) {
          const chunk = accumulatedSamples.slice(0, CHUNK_SIZE);
          accumulatedSamples = accumulatedSamples.slice(CHUNK_SIZE);

          isProcessingRef.current = true;
          const startTime = performance.now();

          onEmbed(chunk, currentMessage).then(watermarkedChunk => {
            const latency = performance.now() - startTime;

            latenciesRef.current.push(latency);
            if (latenciesRef.current.length > 50) latenciesRef.current.shift();
            const avgLatency = latenciesRef.current.reduce((a, b) => a + b, 0) / latenciesRef.current.length;

            setStats(prev => ({
              latencyMs: latency.toFixed(1),
              chunksProcessed: prev.chunksProcessed + 1,
              totalSamples: prev.totalSamples + chunk.length,
              avgLatency: avgLatency.toFixed(1)
            }));

            // Update rolling buffers for spectrum visualization
            setInputBuffer(prev => {
              const newBuf = new Float32Array(2048);
              const shift = Math.min(chunk.length, 2048);
              newBuf.set(prev.slice(shift));
              newBuf.set(chunk.slice(0, shift), 2048 - shift);
              return newBuf;
            });

            setOutputBuffer(prev => {
              const newBuf = new Float32Array(2048);
              const shift = Math.min(watermarkedChunk.length, 2048);
              newBuf.set(prev.slice(shift));
              newBuf.set(watermarkedChunk.slice(0, shift), 2048 - shift);
              return newBuf;
            });

            isProcessingRef.current = false;
          }).catch(err => {
            console.error('Chunk processing error:', err);
            isProcessingRef.current = false;
          });
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      isStreamingRef.current = true;
      isProcessingRef.current = false;
      setIsStreaming(true);
      setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0 });
      latenciesRef.current = [];

    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  }, [onEmbed, isModelReady, message, generateMessage]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    isStreamingRef.current = false;
    isProcessingRef.current = false;
    setIsStreaming(false);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isStreamingRef.current) stopStreaming();
    };
  }, [stopStreaming]);

  const streamDuration = stats.totalSamples / SAMPLE_RATE;

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          {isStreaming && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
          Real-Time Streaming Demo
        </h3>
        <span className="text-xs text-gray-500">
          {isStreaming ? `${streamDuration.toFixed(1)}s` : '8 kHz | 160ms Chunks'}
        </span>
      </div>

      {isStreaming ? (
        <div className="space-y-3">
          {/* Pipeline indicator */}
          <div className="flex items-center justify-center gap-1 text-[10px] text-gray-400">
            <span className="px-2 py-0.5 bg-cyan-500/15 text-cyan-400 rounded">MIC</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-blue-500/15 text-blue-400 rounded">8kHz Resample</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-purple-500/15 text-purple-400 rounded">Encoder (ONNX)</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-yellow-500/15 text-yellow-400 rounded">Watermarked Output</span>
          </div>

          {/* Real-Time Frequency Spectrum – main visualization */}
          <RealtimeOscilloscope
            inputBuffer={inputBuffer}
            outputBuffer={outputBuffer}
            isActive={isStreaming}
            width={640}
            height={340}
          />

          {/* Compact stats row */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-surface/50 rounded-lg px-3 py-2 flex items-baseline justify-between">
              <span className="text-[10px] text-gray-500">Latency</span>
              <span className="text-sm font-mono font-bold text-green-400">{stats.latencyMs}<span className="text-[9px] text-gray-500 ml-0.5">ms</span></span>
            </div>
            <div className="bg-surface/50 rounded-lg px-3 py-2 flex items-baseline justify-between">
              <span className="text-[10px] text-gray-500">Avg</span>
              <span className="text-sm font-mono font-bold text-blue-400">{stats.avgLatency}<span className="text-[9px] text-gray-500 ml-0.5">ms</span></span>
            </div>
            <div className="bg-surface/50 rounded-lg px-3 py-2 flex items-baseline justify-between">
              <span className="text-[10px] text-gray-500">Chunks</span>
              <span className="text-sm font-mono font-bold text-yellow-400">{stats.chunksProcessed}</span>
            </div>
            <div className="bg-surface/50 rounded-lg px-3 py-2 flex items-baseline justify-between">
              <span className="text-[10px] text-gray-500">Duration</span>
              <span className="text-sm font-mono font-bold text-purple-400">{streamDuration.toFixed(1)}<span className="text-[9px] text-gray-500 ml-0.5">s</span></span>
            </div>
          </div>

          {/* Stop Button */}
          <button
            onClick={stopStreaming}
            className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400
                     rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
            실시간 스트리밍 중지
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Idle – spectrum placeholder + start button */}
          <RealtimeOscilloscope
            inputBuffer={inputBuffer}
            outputBuffer={outputBuffer}
            isActive={false}
            width={640}
            height={340}
          />

          <button
            onClick={startStreaming}
            disabled={!isModelReady}
            className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
              ${isModelReady
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:opacity-90 glow'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            실시간 스트리밍 시작
          </button>

          <div className="text-center text-[10px] text-gray-500 space-y-0.5">
            <p>160ms 청크 단위 저지연 처리 | WebAudio + ONNX Runtime Web</p>
            <p>Original(파란색) vs Watermarked(보라색) 주파수 스펙트럼을 실시간 비교합니다</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default RealtimeEmbedDemo;
