/**
 * useInference Hook v2.0
 * 
 * Frame-Wise ONNX Runtime Web integration for CallCops.
 * 
 * 설계 철학:
 * - 40ms 프레임 단위 처리 (320 samples @ 8kHz)
 * - 128-bit Cyclic Payload (5.12초 사이클)
 * - 실시간 스트리밍 지원
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { StreamingEncoderWrapper } from '../utils/StreamingEncoderWrapper';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

// 모델 경로 맵 (FP32 / INT8)
const MODEL_PATHS = {
  fp32: {
    decoder: '/models/decoder.onnx',
    encoder: '/models/encoder.onnx',
  },
  int8: {
    decoder: '/models/decoder_int8.onnx',
    encoder: '/models/encoder_int8.onnx',
  },
};

// Frame-Wise Constants (match Python model)
const FRAME_SAMPLES = 320;      // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;     // 128-bit cyclic payload
const CYCLE_SAMPLES = FRAME_SAMPLES * PAYLOAD_LENGTH;  // 40,960 = 5.12s
const SAMPLE_RATE = 8000;

export function useInference() {
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);
  const [frameProbs, setFrameProbs] = useState(null);  // [num_frames] 프레임별 확률
  const [bitProbs, setBitProbs] = useState(null);      // [128] 복원된 128비트
  const [lastInferenceTime, setLastInferenceTime] = useState(0);
  const [modelPrecision, setModelPrecisionState] = useState('fp32'); // 'fp32' | 'int8'

  const decoderSessionRef = useRef(null);
  const encoderSessionRef = useRef(null);
  const currentPrecisionRef = useRef('fp32');

  /**
   * 모델 정밀도 전환 (FP32 ↔ INT8)
   * 기존 세션을 해제하고 다음 로드 시 새 정밀도 모델을 사용.
   */
  const setModelPrecision = useCallback(async (precision) => {
    if (precision !== 'fp32' && precision !== 'int8') return;
    if (precision === currentPrecisionRef.current) return;

    console.log(`Switching model precision: ${currentPrecisionRef.current} → ${precision}`);

    // 기존 세션 해제
    if (decoderSessionRef.current) {
      decoderSessionRef.current.release();
      decoderSessionRef.current = null;
    }
    if (encoderSessionRef.current) {
      encoderSessionRef.current.release();
      encoderSessionRef.current = null;
    }

    currentPrecisionRef.current = precision;
    setModelPrecisionState(precision);
    setIsReady(false);

    // Decoder 즉시 재로드
    try {
      setIsLoading(true);
      const paths = MODEL_PATHS[precision];
      const session = await ort.InferenceSession.create(paths.decoder, {
        executionProviders: ['webgpu', 'webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });
      decoderSessionRef.current = session;
      setIsReady(true);
      console.log(`Decoder (${precision}) loaded`);
    } catch (err) {
      setError(`Failed to load ${precision} decoder: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Load decoder model
   */
  const loadDecoder = useCallback(async () => {
    if (decoderSessionRef.current) return decoderSessionRef.current;

    const precision = currentPrecisionRef.current;
    const modelPath = MODEL_PATHS[precision].decoder;

    try {
      setIsLoading(true);
      setError(null);

      console.log(`Loading decoder model (${precision})...`);

      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu', 'webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });

      console.log('Decoder using backend:', session.handler?._ep?.name || 'unknown');

      decoderSessionRef.current = session;
      setIsReady(true);
      console.log(`Decoder (${precision}) loaded: ${modelPath}`);

      return session;
    } catch (err) {
      const errorMsg = `Failed to load decoder (${precision}): ${err.message}`;
      console.error(errorMsg);
      setError(errorMsg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Load encoder model (optional, for embedding)
   */
  const loadEncoder = useCallback(async () => {
    if (encoderSessionRef.current) return encoderSessionRef.current;

    const precision = currentPrecisionRef.current;
    const modelPath = MODEL_PATHS[precision].encoder;

    try {
      setIsLoading(true);
      setError(null);

      console.log(`Loading encoder model (${precision})...`);

      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu', 'webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });

      console.log('Encoder using backend:', session.handler?._ep?.name || 'unknown');

      encoderSessionRef.current = session;
      console.log(`Encoder (${precision}) loaded: ${modelPath}`);

      return session;
    } catch (err) {
      const errorMsg = `Failed to load encoder (${precision}): ${err.message}`;
      console.error(errorMsg);
      setError(errorMsg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Align audio to frame boundary (pad to multiple of FRAME_SAMPLES)
   */
  const alignToFrames = useCallback((audioData) => {
    const T = audioData.length;
    if (T % FRAME_SAMPLES === 0) return audioData;

    const padSize = FRAME_SAMPLES - (T % FRAME_SAMPLES);
    const aligned = new Float32Array(T + padSize);
    aligned.set(audioData);
    return aligned;
  }, []);

  /**
   * Get cyclic bit index for a frame
   */
  const getCyclicBitIndex = useCallback((frameIndex) => {
    return frameIndex % PAYLOAD_LENGTH;
  }, []);

  /**
   * Aggregate frame probabilities to 128-bit payload (Cyclic averaging)
   */
  const aggregateTo128Bits = useCallback((frameProbs) => {
    const bits128 = new Float32Array(PAYLOAD_LENGTH).fill(0);
    const counts = new Float32Array(PAYLOAD_LENGTH).fill(0);

    for (let i = 0; i < frameProbs.length; i++) {
      const bitIdx = getCyclicBitIndex(i);
      bits128[bitIdx] += frameProbs[i];
      counts[bitIdx] += 1;
    }

    // Average
    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
      if (counts[i] > 0) {
        bits128[i] /= counts[i];
      } else {
        bits128[i] = 0.5;  // No data, neutral
      }
    }

    return bits128;
  }, [getCyclicBitIndex]);

  /**
   * Run decoder inference (Frame-Wise)
   * @param {Float32Array} audioData - 8kHz audio data (any length)
   * @returns {Object} - { frameProbs: Float32Array, bits128: Float32Array }
   */
  const runDecoder = useCallback(async (audioData) => {
    try {
      // Ensure model is loaded
      const session = decoderSessionRef.current || await loadDecoder();

      const startTime = performance.now();

      // Align to frame boundary
      const alignedAudio = alignToFrames(audioData);
      const numFrames = Math.floor(alignedAudio.length / FRAME_SAMPLES);

      console.log(`Processing ${alignedAudio.length} samples (${numFrames} frames)`);

      // Create tensor [1, 1, T]
      const audioTensor = new ort.Tensor('float32', alignedAudio, [1, 1, alignedAudio.length]);

      // Run inference
      const results = await session.run({ audio: audioTensor });

      // Get frame-wise probabilities [1, num_frames]
      const probs = new Float32Array(results.bit_probs.data);

      // Aggregate to 128-bit payload
      const bits128 = aggregateTo128Bits(probs);

      const endTime = performance.now();
      setLastInferenceTime(endTime - startTime);

      // Update state
      setFrameProbs(probs);
      setBitProbs(bits128);

      return {
        frameProbs: probs,
        bits128: bits128,
        numFrames: numFrames,
        cycleCoverage: numFrames / PAYLOAD_LENGTH  // How many full cycles
      };
    } catch (err) {
      console.error('Decoder inference error:', err);
      setError(`Inference failed: ${err.message}`);
      throw err;
    }
  }, [loadDecoder, alignToFrames, aggregateTo128Bits]);

  /**
   * Run decoder on a specific time segment
   * @param {Float32Array} audioData - Full audio data
   * @param {number} startSec - Start time in seconds
   * @param {number} durationSec - Duration in seconds
   */
  const runDecoderAtPosition = useCallback(async (audioData, startSec, durationSec = 1.0) => {
    const startSample = Math.floor(startSec * SAMPLE_RATE);
    const numSamples = Math.floor(durationSec * SAMPLE_RATE);
    const endSample = Math.min(startSample + numSamples, audioData.length);

    if (startSample >= audioData.length) {
      throw new Error('Start position beyond audio length');
    }

    const segment = audioData.slice(startSample, endSample);
    return runDecoder(segment);
  }, [runDecoder]);

  /**
   * Run encoder inference (Frame-Wise embedding with chunked processing)
   * Processes large files in chunks to avoid WASM memory limits
   * @param {Float32Array} audioData - 8kHz audio data (any length)
   * @param {Float32Array} message - 128-bit message (0/1)
   * @param {Function} onProgress - Optional progress callback (0-100)
   * @returns {Float32Array} - Watermarked audio (same length as input)
   */
  const runEncoder = useCallback(async (audioData, message, onProgress) => {
    try {
      // Ensure model is loaded
      const session = encoderSessionRef.current || await loadEncoder();

      const startTime = performance.now();
      const originalLength = audioData.length;

      // Process in chunks to avoid WASM memory overflow
      // Each chunk = 1 cycle (5.12 seconds = 40,960 samples)
      const CHUNK_SIZE = CYCLE_SAMPLES;  // 40,960 samples = 5.12s
      const MAX_CHUNK_SIZE = CHUNK_SIZE * 2;  // 10.24s max per inference for safety

      // For small files, process all at once
      if (audioData.length <= MAX_CHUNK_SIZE) {
        console.log(`Processing ${audioData.length} samples in single batch`);

        const alignedAudio = alignToFrames(audioData);
        const audioTensor = new ort.Tensor('float32', alignedAudio, [1, 1, alignedAudio.length]);
        const messageTensor = new ort.Tensor('float32', message, [1, PAYLOAD_LENGTH]);

        const results = await session.run({
          audio: audioTensor,
          message: messageTensor,
        });

        const watermarked = new Float32Array(results.watermarked.data);
        onProgress?.(100);

        const endTime = performance.now();
        setLastInferenceTime(endTime - startTime);

        return watermarked.slice(0, originalLength);
      }

      // For large files, process in chunks
      console.log(`Processing ${audioData.length} samples in chunks (${Math.ceil(audioData.length / CHUNK_SIZE)} chunks)`);

      const watermarkedChunks = [];
      const numChunks = Math.ceil(audioData.length / CHUNK_SIZE);

      for (let i = 0; i < numChunks; i++) {
        const chunkStart = i * CHUNK_SIZE;
        const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, audioData.length);
        const chunk = audioData.slice(chunkStart, chunkEnd);

        // Align chunk to frame boundary
        const alignedChunk = alignToFrames(chunk);

        // Create tensors
        const audioTensor = new ort.Tensor('float32', alignedChunk, [1, 1, alignedChunk.length]);
        const messageTensor = new ort.Tensor('float32', message, [1, PAYLOAD_LENGTH]);

        // Run inference on chunk
        const results = await session.run({
          audio: audioTensor,
          message: messageTensor,
        });

        // Store watermarked chunk (trim to original chunk size)
        const watermarkedChunk = new Float32Array(results.watermarked.data);
        watermarkedChunks.push(watermarkedChunk.slice(0, chunkEnd - chunkStart));

        // Report progress
        const progress = Math.round(((i + 1) / numChunks) * 100);
        onProgress?.(progress);

        // Small delay to allow UI updates
        if (i < numChunks - 1) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }

      // Concatenate all chunks
      const totalLength = watermarkedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const watermarked = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of watermarkedChunks) {
        watermarked.set(chunk, offset);
        offset += chunk.length;
      }

      const endTime = performance.now();
      setLastInferenceTime(endTime - startTime);

      console.log(`Processed ${numChunks} chunks in ${(endTime - startTime).toFixed(0)}ms`);

      return watermarked.slice(0, originalLength);
    } catch (err) {
      console.error('Encoder inference error:', err);
      setError(`Encoder failed: ${err.message}`);
      throw err;
    }
  }, [loadEncoder, alignToFrames]);

  /**
   * Create a StreamingEncoderWrapper for real-time frame-by-frame encoding.
   * Loads the encoder model if not already loaded.
   * @returns {Promise<StreamingEncoderWrapper>} - Ready-to-use streaming wrapper
   */
  const createStreamingEncoder = useCallback(async () => {
    const session = encoderSessionRef.current || await loadEncoder();
    return new StreamingEncoderWrapper(session);
  }, [loadEncoder]);

  /**
   * Calculate confidence from bit probabilities
   * Confidence = average of max(p, 1-p) for all bits
   */
  const calculateConfidence = useCallback((probs) => {
    if (!probs || probs.length === 0) return 0;

    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
      sum += Math.max(probs[i], 1 - probs[i]);
    }
    return (sum / probs.length) * 100;
  }, []);

  /**
   * Calculate frame-wise confidence (how confident each frame's bit is)
   */
  const calculateFrameConfidence = useCallback((frameProbs) => {
    if (!frameProbs) return null;
    return frameProbs.map(p => Math.max(p, 1 - p) * 100);
  }, []);

  /**
   * Convert probabilities to binary bits
   */
  const probsToBits = useCallback((probs) => {
    if (!probs) return null;
    return new Uint8Array(probs.map(p => p > 0.5 ? 1 : 0));
  }, []);

  /**
   * Get bit index for current playback time
   */
  const getPlaybackBitIndex = useCallback((currentTimeSec) => {
    const frameIndex = Math.floor((currentTimeSec * SAMPLE_RATE) / FRAME_SAMPLES);
    return getCyclicBitIndex(frameIndex);
  }, [getCyclicBitIndex]);

  /**
   * Get frame index for current playback time
   */
  const getPlaybackFrameIndex = useCallback((currentTimeSec) => {
    return Math.floor((currentTimeSec * SAMPLE_RATE) / FRAME_SAMPLES);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (decoderSessionRef.current) {
        decoderSessionRef.current.release();
      }
      if (encoderSessionRef.current) {
        encoderSessionRef.current.release();
      }
    };
  }, []);

  return {
    // State
    isLoading,
    isReady,
    error,
    frameProbs,      // NEW: Frame-wise probabilities
    bitProbs,        // Aggregated 128-bit payload
    lastInferenceTime,
    modelPrecision,  // 현재 정밀도 ('fp32' | 'int8')

    // Model names (현재 정밀도 반영)
    decoderModelName: MODEL_PATHS[modelPrecision].decoder.split('/').pop(),
    encoderModelName: MODEL_PATHS[modelPrecision].encoder.split('/').pop(),

    // 정밀도 전환
    setModelPrecision,

    // Core functions
    loadDecoder,
    loadEncoder,
    runDecoder,
    runDecoderAtPosition,  // NEW: Detect at specific position
    runEncoder,
    createStreamingEncoder,  // NEW: For real-time 40ms frame streaming

    // Analysis functions
    calculateConfidence,
    calculateFrameConfidence,  // NEW
    probsToBits,
    aggregateTo128Bits,        // NEW

    // Playback helpers
    getPlaybackBitIndex,       // NEW
    getPlaybackFrameIndex,     // NEW
    getCyclicBitIndex,         // NEW

    // Constants (exported for UI)
    FRAME_SAMPLES,
    PAYLOAD_LENGTH,
    CYCLE_SAMPLES,
    SAMPLE_RATE,
  };
}

export default useInference;

// Export constants for use in other modules
export { FRAME_SAMPLES, PAYLOAD_LENGTH, CYCLE_SAMPLES, SAMPLE_RATE };
