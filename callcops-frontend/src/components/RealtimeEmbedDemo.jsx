/**
 * RealtimeEmbedDemo Component
 *
 * Real-time watermark embedding using StreamingEncoderWrapper.
 *
 * Architecture: Queue-based sequential processing
 * ─────────────────────────────────────────────────
 * - ScriptProcessor captures audio chunks synchronously
 * - Chunks are pushed to a queue (no data loss)
 * - A single processing loop drains the queue sequentially
 * - No race conditions: only one frame processes at a time
 *
 * Processing pipeline:
 * - ScriptProcessor captures 2048-sample buffers @ 8kHz
 * - Audio samples are pushed to inputQueue
 * - processLoop() consumes 320-sample frames from the queue
 * - The wrapper handles history buffer (1920 samples → 320 output)
 * - Watermarked frames are stored for final export
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { RealtimeOscilloscope } from './RealtimeOscilloscope';

// Constants matching the encoder model
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;       // 40ms @ 8kHz — model's atomic unit (1 bit per frame)
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload
const VIS_BUFFER_SIZE = 2048;    // Visualization buffer size

export function RealtimeEmbedDemo({ onEmbed, createStreamingEncoder, isModelReady = false, externalMessage = null, onVerify }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState(null);
  const [message, setMessage] = useState(null);
  const [stats, setStats] = useState({
    latencyMs: 0,
    chunksProcessed: 0,
    totalSamples: 0,
    avgLatency: 0
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
  const isProcessingRef = useRef(false);  // Mutex for processing loop
  const stopRequestedRef = useRef(false); // Signal to stop processing loop
  const latenciesRef = useRef([]);

  // Queue-based buffer management (prevents race conditions)
  const inputQueueRef = useRef([]);             // Queue of raw audio chunks
  const pendingBufferRef = useRef(new Float32Array(0));  // Partial frame buffer
  const watermarkedChunksRef = useRef([]);      // Output chunks for final export
  const messageRef = useRef(null);

  // Rolling visualization buffers (continuous update)
  const visInputRef = useRef(new Float32Array(VIS_BUFFER_SIZE));
  const visOutputRef = useRef(new Float32Array(VIS_BUFFER_SIZE));

  // Streaming wrapper ref — holds the StreamingEncoderWrapper instance
  const streamingWrapperRef = useRef(null);

  // Generate random 128-bit message
  const generateMessage = useCallback(() => {
    const msg = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      msg[i] = Math.random() > 0.5 ? 1 : 0;
    }
    setMessage(msg);
    return msg;
  }, []);

  // Update visualization buffers (rolling window)
  const updateVisualization = useCallback((inputFrame, outputFrame) => {
    // Shift left and append new samples
    const shiftAmount = inputFrame.length;
    
    // Input visualization
    visInputRef.current.copyWithin(0, shiftAmount);
    visInputRef.current.set(inputFrame, VIS_BUFFER_SIZE - shiftAmount);
    
    // Output visualization  
    visOutputRef.current.copyWithin(0, shiftAmount);
    visOutputRef.current.set(outputFrame, VIS_BUFFER_SIZE - shiftAmount);
    
    // Batch update React state (creates new array refs for FFT cache invalidation)
    setInputBuffer(visInputRef.current.slice());
    setOutputBuffer(visOutputRef.current.slice());
  }, []);

  /**
   * Sequential processing loop.
   * 
   * This function processes frames one at a time from the queue.
   * It uses a mutex (isProcessingRef) to ensure only one instance runs.
   * New audio chunks are pushed to inputQueueRef by onaudioprocess.
   */
  const processLoop = useCallback(async () => {
    // Mutex: only one loop instance at a time
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;

    const wrapper = streamingWrapperRef.current;
    const msg = messageRef.current;

    try {
      while (!stopRequestedRef.current) {
        // 1. Drain queue into pending buffer
        while (inputQueueRef.current.length > 0) {
          const chunk = inputQueueRef.current.shift();
          const prev = pendingBufferRef.current;
          const combined = new Float32Array(prev.length + chunk.length);
          combined.set(prev);
          combined.set(chunk, prev.length);
          pendingBufferRef.current = combined;
        }

        // 2. Process complete frames
        if (pendingBufferRef.current.length < FRAME_SAMPLES) {
          // Not enough samples yet — yield and wait
          await new Promise(r => setTimeout(r, 5));
          continue;
        }

        // Extract one frame
        const frame = pendingBufferRef.current.slice(0, FRAME_SAMPLES);
        pendingBufferRef.current = pendingBufferRef.current.slice(FRAME_SAMPLES);

        // 3. Process frame through encoder
        const startTime = performance.now();
        const watermarkedFrame = await wrapper.processFrame(frame, msg);
        const latency = performance.now() - startTime;

        // 4. Store watermarked output
        watermarkedChunksRef.current.push(watermarkedFrame);

        // 5. Update latency stats
        latenciesRef.current.push(latency);
        if (latenciesRef.current.length > 50) latenciesRef.current.shift();
        const avgLatency = latenciesRef.current.reduce((a, b) => a + b, 0) / latenciesRef.current.length;

        setStats(prev => ({
          latencyMs: latency.toFixed(1),
          chunksProcessed: prev.chunksProcessed + 1,
          totalSamples: prev.totalSamples + FRAME_SAMPLES,
          avgLatency: avgLatency.toFixed(1)
        }));

        // 6. Update visualization
        updateVisualization(frame, watermarkedFrame);
      }
    } catch (err) {
      console.error('Processing loop error:', err);
    } finally {
      isProcessingRef.current = false;
    }
  }, [updateVisualization]);

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

    const currentMessage = externalMessage || message || generateMessage();
    messageRef.current = currentMessage;

    // Create and reset the streaming wrapper
    const wrapper = await createStreamingEncoder();
    wrapper.reset();
    streamingWrapperRef.current = wrapper;

    // Reset all buffers and queues
    inputQueueRef.current = [];
    pendingBufferRef.current = new Float32Array(0);
    watermarkedChunksRef.current = [];
    visInputRef.current.fill(0);
    visOutputRef.current.fill(0);
    stopRequestedRef.current = false;

    // Clear previous recording
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

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const bufferSize = 2048;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      // Synchronous callback: just push to queue, no async processing here
      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const outputData = e.outputBuffer.getChannelData(0);

        // Pass through audio for monitoring (original audio)
        outputData.set(inputData);

        // Push a copy to the queue (avoid sharing ArrayBuffer)
        inputQueueRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      isStreamingRef.current = true;
      isProcessingRef.current = false;
      setIsStreaming(true);
      setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0 });
      latenciesRef.current = [];

      // Start the processing loop (runs independently)
      processLoop();

    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  }, [createStreamingEncoder, isModelReady, message, externalMessage, generateMessage, recordedAudioUrl, processLoop]);

  // Stop streaming — signal the processing loop to stop, then drain remaining samples
  const stopStreaming = useCallback(async () => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    setIsFinalizing(true);

    // Disconnect audio devices immediately (stop capturing new audio)
    if (sourceRef.current) { sourceRef.current.disconnect(); sourceRef.current = null; }
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null; }
    if (mediaStreamRef.current) { mediaStreamRef.current.getTracks().forEach(t => t.stop()); mediaStreamRef.current = null; }

    // Signal the processing loop to stop
    stopRequestedRef.current = true;

    // Wait for any in-flight processing to complete
    await new Promise((resolve) => {
      const check = () => isProcessingRef.current ? setTimeout(check, 10) : resolve();
      check();
    });

    // Drain remaining queue and pending buffer
    const wrapper = streamingWrapperRef.current;
    const msg = messageRef.current;

    // First, drain the queue into pending buffer
    while (inputQueueRef.current.length > 0) {
      const chunk = inputQueueRef.current.shift();
      const prev = pendingBufferRef.current;
      const combined = new Float32Array(prev.length + chunk.length);
      combined.set(prev);
      combined.set(chunk, prev.length);
      pendingBufferRef.current = combined;
    }

    // Then process all complete frames
    while (pendingBufferRef.current.length >= FRAME_SAMPLES && wrapper && msg) {
      const frame = pendingBufferRef.current.slice(0, FRAME_SAMPLES);
      pendingBufferRef.current = pendingBufferRef.current.slice(FRAME_SAMPLES);

      try {
        const watermarkedFrame = await wrapper.processFrame(frame, msg);
        watermarkedChunksRef.current.push(watermarkedFrame);
      } catch (err) {
        console.error('Drain frame error:', err);
        break;
      }
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
  }, [encodeWav]);

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

  // Verify — pass original (unrotated) message for correct comparison
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
    setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0 });
    streamingWrapperRef.current = null;
    inputQueueRef.current = [];
    pendingBufferRef.current = new Float32Array(0);
    watermarkedChunksRef.current = [];
  }, [recordedAudioUrl]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRequestedRef.current = true;
      if (isStreamingRef.current) {
        isStreamingRef.current = false;
        if (processorRef.current) processorRef.current.disconnect();
        if (audioContextRef.current) audioContextRef.current.close();
        if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const streamDuration = stats.totalSamples / SAMPLE_RATE;
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
          <div className="flex items-center justify-center gap-3 py-6">
            <svg className="w-6 h-6 animate-spin text-yellow-400" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <span className="text-sm text-yellow-400 font-medium">Finalizing...</span>
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
            <p>40ms 프레임 단위 처리 (25 FPS) | History Buffer 1920 samples | ONNX Runtime Web</p>
            <p>Original(파란색) vs Watermarked(보라색) 주파수 스펙트럼을 실시간 비교합니다</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default RealtimeEmbedDemo;
