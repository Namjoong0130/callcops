/**
 * RealtimeEmbedDemo Component
 *
 * Real-time watermark embedding using StreamingEncoderWrapper.
 *
 * Architecture: Decoupled recording + visualization pipelines
 * ─────────────────────────────────────────────────────────────────────────
 * - ScriptProcessor captures audio chunks synchronously and feeds BOTH:
 *     1. visInputRef (oscilloscope input) — updated directly, always real-time
 *     2. inputQueue  (ONNX processing)   — best-effort, may lag behind
 * - processLoop watermarks frames and updates visOutputRef (output waveform)
 * - A separate rAF-based vis loop pushes vis buffers to React state (~12fps)
 * - Oscilloscope is always active and independent of ONNX processing speed
 * - Raw audio is ALWAYS saved (rawChunksRef) — zero-drop guarantee
 * - On stop, any unprocessed raw audio is watermarked during finalization
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { RealtimeOscilloscope } from './RealtimeOscilloscope';

// Constants matching the encoder model
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;       // 40ms @ 8kHz — model's atomic unit (1 bit per frame)
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload
const VIS_BUFFER_SIZE = 2048;    // Visualization buffer size
const VIS_THROTTLE_MS = 80;     // ~12fps for visualization React state updates
const YIELD_EVERY_N_FRAMES = 4; // Yield to browser every N frames (reduces setTimeout overhead)
const PENDING_BUFFER_INITIAL = 8192; // Pre-allocated pending buffer capacity

export function RealtimeEmbedDemo({ onEmbed, createStreamingEncoder, isModelReady = false, externalMessage = null, onVerify }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [finalizeProgress, setFinalizeProgress] = useState('');
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState(null);
  const [stats, setStats] = useState({
    latencyMs: 0,
    chunksProcessed: 0,
    totalSamples: 0,
    avgLatency: 0,
    queueDepth: 0
  });

  // Buffers for frequency spectrum visualization
  const [inputBuffer, setInputBuffer] = useState(new Float32Array(VIS_BUFFER_SIZE));
  const [outputBuffer, setOutputBuffer] = useState(new Float32Array(VIS_BUFFER_SIZE));

  // Audio capture refs
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);

  // Processing state refs
  const isStreamingRef = useRef(false);
  const isProcessingRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const latenciesRef = useRef([]);
  const visLoopRef = useRef(null); // rAF handle for independent visualization loop

  // Queue-based buffer management
  const inputQueueRef = useRef([]);
  const pendingDataRef = useRef(new Float32Array(PENDING_BUFFER_INITIAL));
  const pendingLenRef = useRef(0);
  const watermarkedChunksRef = useRef([]);
  const messageRef = useRef(null);

  // Raw audio accumulation — ALWAYS saved, never dropped
  const rawChunksRef = useRef([]);
  const rawTotalSamplesRef = useRef(0);

  // Rolling visualization buffers
  const visInputRef = useRef(new Float32Array(VIS_BUFFER_SIZE));
  const visOutputRef = useRef(new Float32Array(VIS_BUFFER_SIZE));

  // Streaming wrapper ref
  const streamingWrapperRef = useRef(null);

  // Visualization throttle
  const lastVisTimeRef = useRef(0);

  /**
   * Independent rAF-based visualization loop.
   * Pushes vis buffers to React state at ~12fps, completely decoupled
   * from the ONNX processLoop. This ensures the oscilloscope always
   * reflects real-time input audio even when ONNX is lagging.
   */
  const startVisLoop = useCallback(() => {
    const loop = () => {
      if (!visLoopRef.current) return; // Cancelled

      const now = performance.now();
      if (now - lastVisTimeRef.current >= VIS_THROTTLE_MS) {
        setInputBuffer(visInputRef.current.slice());
        setOutputBuffer(visOutputRef.current.slice());
        lastVisTimeRef.current = now;
      }

      visLoopRef.current = requestAnimationFrame(loop);
    };
    visLoopRef.current = requestAnimationFrame(loop);
  }, []);

  const stopVisLoop = useCallback(() => {
    if (visLoopRef.current) {
      cancelAnimationFrame(visLoopRef.current);
      visLoopRef.current = null;
    }
    // Push final snapshot
    setInputBuffer(visInputRef.current.slice());
    setOutputBuffer(visOutputRef.current.slice());
  }, []);

  /**
   * Append data to the pre-allocated pending buffer, growing if needed.
   */
  const appendToPending = useCallback((data) => {
    const needed = pendingLenRef.current + data.length;
    let buf = pendingDataRef.current;
    if (needed > buf.length) {
      const newBuf = new Float32Array(Math.max(buf.length * 2, needed));
      newBuf.set(buf.subarray(0, pendingLenRef.current));
      pendingDataRef.current = newBuf;
      buf = newBuf;
    }
    buf.set(data, pendingLenRef.current);
    pendingLenRef.current += data.length;
  }, []);

  /**
   * Compact the pending buffer: shift consumed data to the front.
   */
  const compactPending = useCallback((consumed) => {
    if (consumed <= 0) return;
    const remaining = pendingLenRef.current - consumed;
    if (remaining > 0) {
      pendingDataRef.current.copyWithin(0, consumed, pendingLenRef.current);
    }
    pendingLenRef.current = remaining;
  }, []);

  const processLoop = useCallback(async () => {
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;

    const wrapper = streamingWrapperRef.current;
    const msg = messageRef.current;
    const frameBuf = new Float32Array(FRAME_SAMPLES);

    try {
      while (!stopRequestedRef.current) {
        // 1. Drain queue into pending buffer
        const queueLen = inputQueueRef.current.length;
        if (queueLen > 0) {
          const chunks = inputQueueRef.current.splice(0);
          for (const c of chunks) {
            appendToPending(c);
          }
        }

        // 2. Not enough data — yield and wait
        if (pendingLenRef.current < FRAME_SAMPLES) {
          await new Promise(r => setTimeout(r, 5));
          continue;
        }

        // 3. Process all complete frames (recording only — vis is handled separately)
        let readOffset = 0;
        let framesProcessed = 0;

        while (readOffset + FRAME_SAMPLES <= pendingLenRef.current) {
          const srcOffset = readOffset;
          for (let i = 0; i < FRAME_SAMPLES; i++) {
            frameBuf[i] = pendingDataRef.current[srcOffset + i];
          }
          readOffset += FRAME_SAMPLES;

          const startTime = performance.now();
          const watermarkedFrame = await wrapper.processFrame(frameBuf, msg);
          const latency = performance.now() - startTime;

          watermarkedChunksRef.current.push(watermarkedFrame);

          latenciesRef.current.push(latency);
          if (latenciesRef.current.length > 50) latenciesRef.current.shift();

          // Update output visualization buffer (input vis is updated in audio callback)
          visOutputRef.current.copyWithin(0, FRAME_SAMPLES);
          visOutputRef.current.set(watermarkedFrame, VIS_BUFFER_SIZE - FRAME_SAMPLES);

          framesProcessed++;

          if (framesProcessed % YIELD_EVERY_N_FRAMES === 0) {
            await new Promise(r => setTimeout(r, 0));
          }
        }

        compactPending(readOffset);

        // 4. Update stats ONCE per batch (vis updates handled by rAF loop)
        if (framesProcessed > 0) {
          const avgLatency = latenciesRef.current.reduce((a, b) => a + b, 0) / latenciesRef.current.length;

          setStats(prev => ({
            latencyMs: latenciesRef.current[latenciesRef.current.length - 1]?.toFixed(1) || '0',
            chunksProcessed: prev.chunksProcessed + framesProcessed,
            totalSamples: prev.totalSamples + framesProcessed * FRAME_SAMPLES,
            avgLatency: avgLatency.toFixed(1),
            queueDepth: pendingLenRef.current
          }));
        }

        await new Promise(r => setTimeout(r, 0));
      }
    } catch (err) {
      console.error('Processing loop error:', err);
    } finally {
      isProcessingRef.current = false;
    }
  }, [appendToPending, compactPending]);

  // WAV encoding helper
  const encodeWav = useCallback((audioData) => {
    const sampleRate = SAMPLE_RATE;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = audioData.length * 2;
    const bufferSize = 44 + dataSize;

    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    for (let i = 0; i < audioData.length; i++) {
      const sample = Math.max(-1, Math.min(1, audioData[i]));
      view.setInt16(44 + i * 2, sample * 32767, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }, []);

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!createStreamingEncoder || !isModelReady) return;

    if (!externalMessage) {
      alert('워터마크 메시지를 먼저 설정해주세요.\n아래 "Set Watermark Message" 섹션에서 Timestamp와 Auth Data를 입력하세요.');
      return;
    }

    const currentMessage = externalMessage;
    messageRef.current = currentMessage;

    const wrapper = await createStreamingEncoder();
    wrapper.reset();
    streamingWrapperRef.current = wrapper;

    // Reset all buffers and queues
    inputQueueRef.current = [];
    pendingDataRef.current = new Float32Array(PENDING_BUFFER_INITIAL);
    pendingLenRef.current = 0;
    watermarkedChunksRef.current = [];
    rawChunksRef.current = [];
    rawTotalSamplesRef.current = 0;
    visInputRef.current.fill(0);
    visOutputRef.current.fill(0);
    stopRequestedRef.current = false;
    setFinalizeProgress('');

    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
    setRecordedAudio(null);
    setRecordedAudioUrl(null);

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

      const actualRate = audioContext.sampleRate;
      const needsResample = actualRate !== SAMPLE_RATE;
      if (needsResample) {
        console.warn(`AudioContext sampleRate: ${actualRate}Hz (requested ${SAMPLE_RATE}Hz). Will downsample.`);
      }
      const resampleRatio = needsResample ? SAMPLE_RATE / actualRate : 1;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const bufferSize = 2048;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const outputData = e.outputBuffer.getChannelData(0);

        // Pass through audio for monitoring
        outputData.set(inputData);

        // Downsample if needed
        let audioChunk;
        if (needsResample) {
          const outLen = Math.floor(inputData.length * resampleRatio);
          audioChunk = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            audioChunk[i] = inputData[Math.floor(i / resampleRatio)] || 0;
          }
        } else {
          audioChunk = new Float32Array(inputData);
        }

        // ALWAYS save raw audio (zero-drop guarantee)
        rawChunksRef.current.push(audioChunk);
        rawTotalSamplesRef.current += audioChunk.length;

        // Push to processing queue (best-effort real-time watermarking)
        inputQueueRef.current.push(audioChunk);

        // Update oscilloscope input buffer directly (real-time, no ONNX dependency)
        const visBuf = visInputRef.current;
        const chunkLen = audioChunk.length;
        if (chunkLen < VIS_BUFFER_SIZE) {
          visBuf.copyWithin(0, chunkLen);
          visBuf.set(audioChunk, VIS_BUFFER_SIZE - chunkLen);
        } else {
          visBuf.set(audioChunk.subarray(chunkLen - VIS_BUFFER_SIZE));
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      isStreamingRef.current = true;
      isProcessingRef.current = false;
      setIsStreaming(true);
      setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0, queueDepth: 0 });
      latenciesRef.current = [];

      processLoop();
      startVisLoop();

    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  }, [createStreamingEncoder, isModelReady, externalMessage, recordedAudioUrl, processLoop, startVisLoop]);

  // Stop streaming — drain ALL remaining frames (zero-drop)
  const stopStreaming = useCallback(async () => {
    isStreamingRef.current = false;
    stopVisLoop();
    setIsStreaming(false);
    setIsFinalizing(true);

    // Disconnect audio devices immediately
    if (sourceRef.current) { sourceRef.current.disconnect(); sourceRef.current = null; }
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null; }
    if (mediaStreamRef.current) { mediaStreamRef.current.getTracks().forEach(t => t.stop()); mediaStreamRef.current = null; }

    // Signal the processing loop to stop
    stopRequestedRef.current = true;

    // Wait for in-flight processing to complete
    await new Promise((resolve) => {
      const check = () => isProcessingRef.current ? setTimeout(check, 10) : resolve();
      check();
    });

    const wrapper = streamingWrapperRef.current;
    const msg = messageRef.current;

    // Count how many frames were already watermarked in real-time
    const watermarkedFrameCount = watermarkedChunksRef.current.length;

    // Reconstruct the FULL raw audio from rawChunksRef (guaranteed complete)
    const rawTotal = rawTotalSamplesRef.current;
    const fullRawAudio = new Float32Array(rawTotal);
    let writePos = 0;
    for (const chunk of rawChunksRef.current) {
      fullRawAudio.set(chunk, writePos);
      writePos += chunk.length;
    }
    rawChunksRef.current = []; // Free memory

    // Calculate how many frames we need total and how many are remaining
    const totalFrames = Math.floor(rawTotal / FRAME_SAMPLES);
    const remainingFrames = totalFrames - watermarkedFrameCount;

    if (remainingFrames > 0 && wrapper && msg) {
      setFinalizeProgress(`${watermarkedFrameCount}/${totalFrames} frames (processing ${remainingFrames} remaining...)`);

      // Process remaining frames from the raw audio
      const drainFrame = new Float32Array(FRAME_SAMPLES);
      let drainedCount = 0;

      for (let frameIdx = watermarkedFrameCount; frameIdx < totalFrames; frameIdx++) {
        const srcStart = frameIdx * FRAME_SAMPLES;
        for (let i = 0; i < FRAME_SAMPLES; i++) {
          drainFrame[i] = fullRawAudio[srcStart + i];
        }

        try {
          const watermarkedFrame = await wrapper.processFrame(drainFrame, msg);
          watermarkedChunksRef.current.push(watermarkedFrame);
          drainedCount++;

          if (drainedCount % 10 === 0) {
            setFinalizeProgress(`${watermarkedFrameCount + drainedCount}/${totalFrames} frames`);
            await new Promise(r => setTimeout(r, 0));
          }
        } catch (err) {
          console.error('Drain frame error:', err);
          break;
        }
      }

      setFinalizeProgress(`${watermarkedFrameCount + drainedCount}/${totalFrames} frames (done)`);
    }

    // Concatenate all watermarked frames
    const chunks = watermarkedChunksRef.current;
    if (chunks.length > 0) {
      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const finalAudio = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        finalAudio.set(chunk, offset);
        offset += chunk.length;
      }

      setRecordedAudio(finalAudio);
      const blob = encodeWav(finalAudio);
      setRecordedAudioUrl(URL.createObjectURL(blob));
    }

    setIsFinalizing(false);
  }, [encodeWav, stopVisLoop]);

  // Download
  const handleDownload = useCallback(() => {
    if (!recordedAudio) return;
    const blob = encodeWav(recordedAudio);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `watermarked_recording_${Date.now()}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }, [recordedAudio, encodeWav]);

  // Verify
  const handleVerify = useCallback(() => {
    if (recordedAudio && onVerify) {
      onVerify(recordedAudio, messageRef.current);
    }
  }, [recordedAudio, onVerify]);

  // New recording
  const handleNewRecording = useCallback(() => {
    if (recordedAudioUrl) URL.revokeObjectURL(recordedAudioUrl);
    setRecordedAudio(null);
    setRecordedAudioUrl(null);
    setInputBuffer(new Float32Array(VIS_BUFFER_SIZE));
    setOutputBuffer(new Float32Array(VIS_BUFFER_SIZE));
    setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0, queueDepth: 0 });
    streamingWrapperRef.current = null;
    inputQueueRef.current = [];
    pendingDataRef.current = new Float32Array(PENDING_BUFFER_INITIAL);
    pendingLenRef.current = 0;
    watermarkedChunksRef.current = [];
    rawChunksRef.current = [];
    rawTotalSamplesRef.current = 0;
  }, [recordedAudioUrl]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRequestedRef.current = true;
      if (visLoopRef.current) {
        cancelAnimationFrame(visLoopRef.current);
        visLoopRef.current = null;
      }
      if (isStreamingRef.current) {
        isStreamingRef.current = false;
        if (processorRef.current) processorRef.current.disconnect();
        if (audioContextRef.current) audioContextRef.current.close();
        if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const streamDuration = stats.totalSamples / SAMPLE_RATE;
  const rawDuration = rawTotalSamplesRef.current / SAMPLE_RATE;
  const recordedDuration = recordedAudio ? (recordedAudio.length / SAMPLE_RATE).toFixed(1) : 0;
  const wrapperFrameIndex = streamingWrapperRef.current?.frameIndex ?? 0;

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          {isStreaming && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
          {isFinalizing && <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />}
          Real-Time Streaming Demo
        </h3>
        <span className="text-xs text-gray-500">
          {isStreaming ? `${streamDuration.toFixed(1)}s | Frame #${wrapperFrameIndex}` : '8 kHz | 40ms Frames | 25 FPS'}
        </span>
      </div>

      {isStreaming ? (
        <div className="space-y-3">
          <div className="flex items-center justify-center gap-1 text-[10px] text-gray-400">
            <span className="px-2 py-0.5 bg-cyan-500/15 text-cyan-400 rounded">MIC</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-blue-500/15 text-blue-400 rounded">8kHz Resample</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-purple-500/15 text-purple-400 rounded">Encoder (1920→320)</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="px-2 py-0.5 bg-yellow-500/15 text-yellow-400 rounded">Watermarked Output</span>
          </div>

          {/* Oscilloscope — always active, decoupled from recording pipeline */}
          <RealtimeOscilloscope inputBuffer={inputBuffer} outputBuffer={outputBuffer} isActive={isStreaming} width={640} height={340} />

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
              <span className="text-[10px] text-gray-500">Frames</span>
              <span className="text-sm font-mono font-bold text-yellow-400">{stats.chunksProcessed}</span>
            </div>
            <div className="bg-surface/50 rounded-lg px-3 py-2 flex items-baseline justify-between">
              <span className="text-[10px] text-gray-500">Duration</span>
              <span className="text-sm font-mono font-bold text-purple-400">{streamDuration.toFixed(1)}<span className="text-[9px] text-gray-500 ml-0.5">s</span></span>
            </div>
          </div>

          <button onClick={stopStreaming}
            className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl font-medium transition-colors flex items-center justify-center gap-2">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
            실시간 스트리밍 중지
          </button>
        </div>
      ) : isFinalizing ? (
        <div className="space-y-3">
          <RealtimeOscilloscope inputBuffer={inputBuffer} outputBuffer={outputBuffer} isActive={false} width={640} height={340} />
          <div className="flex flex-col items-center justify-center gap-2 py-6">
            <div className="flex items-center gap-3">
              <svg className="w-6 h-6 animate-spin text-yellow-400" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              <span className="text-sm text-yellow-400 font-medium">Finalizing...</span>
            </div>
            {finalizeProgress && (
              <span className="text-xs text-gray-400 font-mono">{finalizeProgress}</span>
            )}
            <span className="text-[10px] text-gray-500">
              Remaining frames are being watermarked. No audio will be dropped.
            </span>
          </div>
        </div>
      ) : recordedAudio ? (
        <div className="space-y-3">
          <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-xl space-y-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-green-500/20 text-green-400 text-xs font-bold flex items-center justify-center">✓</span>
              <h4 className="text-sm font-semibold text-green-400">Recording Complete</h4>
              <span className="text-xs text-gray-400 ml-auto">{recordedDuration}s @ {SAMPLE_RATE / 1000}kHz</span>
            </div>
            <audio src={recordedAudioUrl} controls className="w-full h-10" />
            <div className="flex gap-2">
              <button onClick={handleDownload}
                className="flex-1 py-2.5 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download WAV
              </button>
              {onVerify && (
                <button onClick={handleVerify}
                  className="flex-1 py-2.5 bg-primary/20 hover:bg-primary/30 text-primary rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Verify with Decoder
                </button>
              )}
            </div>
            <button onClick={handleNewRecording}
              className="w-full py-2 bg-surface/50 hover:bg-surface/80 text-gray-400 hover:text-gray-200 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              New Recording
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <RealtimeOscilloscope inputBuffer={inputBuffer} outputBuffer={outputBuffer} isActive={false} width={640} height={340} />
          <button onClick={startStreaming} disabled={!isModelReady}
            className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
              ${isModelReady ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:opacity-90 glow' : 'bg-gray-700 text-gray-500 cursor-not-allowed'}`}>
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            실시간 스트리밍 시작
          </button>
          <div className="text-center text-[10px] text-gray-500 space-y-0.5">
            <p>40ms frame processing (25 FPS) | Zero-drop recording | ONNX Runtime Web</p>
            <p>Toggle oscilloscope during streaming for Original vs Watermarked spectrum comparison</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default RealtimeEmbedDemo;
