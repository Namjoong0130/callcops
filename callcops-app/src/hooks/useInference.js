/**
 * useInference Hook for React Native
 * Uses onnxruntime-react-native for ONNX model inference
 *
 * Fixed: Cold start bug where ONNX fails on first launch
 * - Dynamic ONNX availability check (not just at require time)
 * - Asset loading with retry and verification
 * - Proper initialization state management
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import * as FileSystem from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';
import { resampleLinear } from '../utils/AudioResampler';

// Frame-Wise Constants (match Python model)
const FRAME_SAMPLES = 320;      // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;     // 128-bit cyclic payload
const SAMPLE_RATE = 8000;

// Lazy-load ONNX runtime to handle cold start
let _ort = null;
let _InferenceSession = null;
let _Tensor = null;

/**
 * Attempt to load ONNX runtime (can be called multiple times safely)
 * @returns {boolean} Whether ONNX is available
 */
const tryLoadOnnxRuntime = () => {
  if (_InferenceSession && _Tensor) {
    return true;
  }

  try {
    _ort = require('onnxruntime-react-native');
    _InferenceSession = _ort.InferenceSession;
    _Tensor = _ort.Tensor;
    console.log('ONNX Runtime loaded successfully');
    return true;
  } catch (e) {
    console.warn('ONNX Runtime not available:', e.message);
    return false;
  }
};

// Initial attempt
tryLoadOnnxRuntime();

export function useInference() {
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);
  const [onnxAvailable, setOnnxAvailable] = useState(false);
  const [initAttempted, setInitAttempted] = useState(false);

  const decoderSessionRef = useRef(null);
  const encoderSessionRef = useRef(null);

  // Dynamic ONNX availability check on mount and periodically
  useEffect(() => {
    const checkOnnx = () => {
      const available = tryLoadOnnxRuntime();
      setOnnxAvailable(available);
      if (!available) {
        setError('ONNX Runtime not available on this device');
      }
      setInitAttempted(true);
    };

    // Check immediately
    checkOnnx();

    // Retry after a delay in case native module is still initializing
    const retryTimeout = setTimeout(() => {
      if (!onnxAvailable) {
        console.log('Retrying ONNX initialization...');
        checkOnnx();
      }
    }, 500);

    // Another retry after longer delay
    const retryTimeout2 = setTimeout(() => {
      if (!onnxAvailable) {
        console.log('Final ONNX initialization attempt...');
        checkOnnx();
      }
    }, 1500);

    return () => {
      clearTimeout(retryTimeout);
      clearTimeout(retryTimeout2);
    };
  }, []);

  /**
   * Get local file path for an asset with verification
   * Handles cold start race conditions with retry logic
   */
  const getAssetLocalUri = useCallback(async (assetModule, maxRetries = 3) => {
    let lastError = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const asset = Asset.fromModule(assetModule);

        // Ensure asset is downloaded
        if (!asset.localUri) {
          await asset.downloadAsync();
        }

        // Check if localUri is now available
        if (asset.localUri) {
          // Verify the file actually exists
          const info = await FileSystem.getInfoAsync(asset.localUri);
          if (info.exists && info.size > 0) {
            console.log(`Asset loaded (attempt ${attempt}):`, asset.localUri);
            return asset.localUri;
          }
        }

        // Fallback: copy to document directory
        const fileName = `${asset.name || 'model'}.${asset.type || 'onnx'}`;
        const destPath = FileSystem.documentDirectory + fileName;

        // Check if already copied and valid
        const destInfo = await FileSystem.getInfoAsync(destPath);
        if (destInfo.exists && destInfo.size > 0) {
          console.log(`Using cached asset (attempt ${attempt}):`, destPath);
          return destPath;
        }

        // Download from asset URI
        if (asset.uri) {
          console.log(`Downloading asset to cache (attempt ${attempt})...`);
          await FileSystem.downloadAsync(asset.uri, destPath);

          // Verify download
          const verifyInfo = await FileSystem.getInfoAsync(destPath);
          if (verifyInfo.exists && verifyInfo.size > 0) {
            console.log('Asset downloaded and verified:', destPath);
            return destPath;
          }
        }

        throw new Error('Asset not available after download');
      } catch (err) {
        lastError = err;
        console.warn(`Asset loading attempt ${attempt} failed:`, err.message);

        if (attempt < maxRetries) {
          // Wait before retry (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 200 * attempt));
        }
      }
    }

    throw new Error(`Failed to load asset after ${maxRetries} attempts: ${lastError?.message}`);
  }, []);

  /**
   * Ensure ONNX runtime is available (with retry)
   */
  const ensureOnnxAvailable = useCallback(async () => {
    // First check
    if (_InferenceSession && _Tensor) {
      return true;
    }

    // Try loading
    if (tryLoadOnnxRuntime()) {
      setOnnxAvailable(true);
      return true;
    }

    // Wait and retry
    await new Promise(resolve => setTimeout(resolve, 300));
    if (tryLoadOnnxRuntime()) {
      setOnnxAvailable(true);
      return true;
    }

    return false;
  }, []);

  /**
   * Load decoder model with robust initialization
   */
  const loadDecoder = useCallback(async () => {
    // Ensure ONNX is available first
    const onnxReady = await ensureOnnxAvailable();
    if (!onnxReady) {
      throw new Error('ONNX Runtime not available after retries');
    }

    // Return existing session if already loaded
    if (decoderSessionRef.current) {
      setIsReady(true);
      return decoderSessionRef.current;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Get asset path with retry logic
      const modelAsset = require('../../assets/models/decoder.onnx');
      const modelPath = await getAssetLocalUri(modelAsset);

      console.log('Loading decoder from:', modelPath);

      // Verify file exists before creating session
      const fileInfo = await FileSystem.getInfoAsync(modelPath);
      if (!fileInfo.exists) {
        throw new Error(`Model file not found at: ${modelPath}`);
      }
      console.log(`Decoder model size: ${fileInfo.size} bytes`);

      // Create inference session
      const session = await _InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
      });

      decoderSessionRef.current = session;
      setIsReady(true);
      setOnnxAvailable(true);
      console.log('Decoder loaded successfully');
      console.log('Input names:', session.inputNames);
      console.log('Output names:', session.outputNames);

      return session;
    } catch (err) {
      console.error('Failed to load decoder:', err);
      setError(`Failed to load decoder: ${err.message}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [getAssetLocalUri, ensureOnnxAvailable]);

  /**
   * Load encoder model with robust initialization
   * @returns {InferenceSession} The loaded ONNX session
   */
  const loadEncoder = useCallback(async () => {
    // Ensure ONNX is available first
    const onnxReady = await ensureOnnxAvailable();
    if (!onnxReady) {
      throw new Error('ONNX Runtime not available after retries');
    }

    // Return existing session if already loaded
    if (encoderSessionRef.current) {
      return encoderSessionRef.current;
    }

    setIsLoading(true);
    setError(null);

    try {
      const modelAsset = require('../../assets/models/encoder.onnx');
      const modelPath = await getAssetLocalUri(modelAsset);

      console.log('Loading encoder from:', modelPath);

      // Verify file exists
      const fileInfo = await FileSystem.getInfoAsync(modelPath);
      if (!fileInfo.exists) {
        throw new Error(`Model file not found at: ${modelPath}`);
      }
      console.log(`Encoder model size: ${fileInfo.size} bytes`);

      const session = await _InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
      });

      encoderSessionRef.current = session;
      setOnnxAvailable(true);
      console.log('Encoder loaded successfully');
      console.log('Input names:', session.inputNames);
      console.log('Output names:', session.outputNames);

      return session;
    } catch (err) {
      console.error('Failed to load encoder:', err);
      setError(`Failed to load encoder: ${err.message}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [getAssetLocalUri, ensureOnnxAvailable]);

  /**
   * Resample audio to 8kHz mono
   */
  /**
   * Resample audio to 8kHz mono
   * Reverted to simple Nearest Neighbor to match upgrade branch performance
   */
  const resampleTo8kHz = useCallback((audioData, originalSampleRate) => {
    if (originalSampleRate === SAMPLE_RATE) {
      return audioData;
    }

    const ratio = originalSampleRate / SAMPLE_RATE;
    const newLength = Math.floor(audioData.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = Math.floor(i * ratio);
      resampled[i] = audioData[srcIndex];
    }

    return resampled;
  }, []);

  /**
   * Run decoder on audio data (Batch Processing like Web)
   * @param {Float32Array} audioData - Audio samples at 8kHz
   * @returns {Object} { bits128, confidence, numFrames }
   */
  const runDecoder = useCallback(async (audioData) => {
    if (!decoderSessionRef.current) {
      throw new Error('Decoder not loaded');
    }
    if (!_Tensor) {
      throw new Error('ONNX Tensor not available');
    }

    try {
      const session = decoderSessionRef.current;

      // Align to frame boundary
      const remainder = audioData.length % FRAME_SAMPLES;
      let alignedAudio = audioData;
      if (remainder !== 0) {
        const pad = FRAME_SAMPLES - remainder;
        const padded = new Float32Array(audioData.length + pad);
        padded.set(audioData);
        alignedAudio = padded;
      }

      const numFrames = alignedAudio.length / FRAME_SAMPLES;
      if (numFrames === 0) throw new Error('Audio too short');

      // Create tensor [1, 1, T] - Entire audio at once
      // Match upgrade branch: Use plain Array for compatibility
      const audioArray = Array.from(alignedAudio);
      const inputTensor = new _Tensor('float32', audioArray, [1, 1, alignedAudio.length]);

      // Run inference
      const feeds = {};
      feeds[session.inputNames[0]] = inputTensor;
      const results = await session.run(feeds);

      // Get bit_probs output (Expected shape: [1, num_frames])
      const outputName = session.outputNames[0];
      const outputData = results[outputName].data;

      // Aggregate probabilities using CYCLIC method (like frontend)
      const bitAccumulators = new Float32Array(PAYLOAD_LENGTH).fill(0);
      const bitCounts = new Float32Array(PAYLOAD_LENGTH).fill(0);

      for (let i = 0; i < outputData.length; i++) {
        const bitIdx = i % PAYLOAD_LENGTH;
        bitAccumulators[bitIdx] += outputData[i];
        bitCounts[bitIdx] += 1;
      }

      // Calculate averaged probabilities
      const bits128 = new Float32Array(PAYLOAD_LENGTH);
      for (let i = 0; i < PAYLOAD_LENGTH; i++) {
        bits128[i] = bitCounts[i] > 0 ? bitAccumulators[i] / bitCounts[i] : 0.5;
      }

      // Calculate confidence
      const confidences = bits128.map(p => Math.abs(p - 0.5) * 2);
      const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;

      return {
        bits128,
        confidence: avgConfidence,
        numFrames,
        rawScores: outputData,
      };
    } catch (err) {
      console.error('Decoder inference error:', err);
      throw err;
    }
  }, []);

  /**
   * Run encoder on audio data with message (Batch layout)
   */
  const runEncoder = useCallback(async (audioData, messageBits) => {
    if (!encoderSessionRef.current) {
      throw new Error('Encoder not loaded');
    }
    if (!_Tensor) {
      throw new Error('ONNX Tensor not available');
    }

    try {
      const session = encoderSessionRef.current;

      // Align to frame boundary
      const remainder = audioData.length % FRAME_SAMPLES;
      let alignedAudio = audioData;
      if (remainder !== 0) {
        const pad = FRAME_SAMPLES - remainder;
        const padded = new Float32Array(audioData.length + pad);
        padded.set(audioData);
        alignedAudio = padded;
      }

      const numFrames = alignedAudio.length / FRAME_SAMPLES;
      if (numFrames === 0) throw new Error('Audio too short');

      // Prepare tensors
      // Match upgrade branch: Use plain Array
      const audioArray = Array.from(alignedAudio);
      const audioTensor = new _Tensor('float32', audioArray, [1, 1, alignedAudio.length]);

      const messageArray = new Float32Array(PAYLOAD_LENGTH);
      for (let i = 0; i < PAYLOAD_LENGTH; i++) {
        messageArray[i] = messageBits[i] || 0;
      }
      const messageTensor = new _Tensor('float32', messageArray, [1, PAYLOAD_LENGTH]);

      // Run inference
      const feeds = {};
      feeds[session.inputNames[0]] = audioTensor;
      if (session.inputNames.length > 1) {
        feeds[session.inputNames[1]] = messageTensor;
      }

      const results = await session.run(feeds);

      // Get encoded output
      const outputName = session.outputNames[0];
      const outputData = results[outputName].data;

      const encoded = new Float32Array(outputData);
      return { encoded };
    } catch (err) {
      console.error('Encoder inference error:', err);
      throw err;
    }
  }, []);

  /**
   * Check if ONNX is truly available (dynamic check)
   * Use this instead of just reading onnxAvailable state
   */
  const checkOnnxAvailable = useCallback(() => {
    const available = tryLoadOnnxRuntime();
    if (available !== onnxAvailable) {
      setOnnxAvailable(available);
    }
    return available;
  }, [onnxAvailable]);

  return {
    isLoading,
    isReady,
    error,
    onnxAvailable,
    initAttempted,
    loadDecoder,
    loadEncoder,
    runDecoder,
    runEncoder,
    resampleTo8kHz,
    checkOnnxAvailable,
    ensureOnnxAvailable,
    encoderSession: encoderSessionRef.current,
    decoderSession: decoderSessionRef.current,
    Tensor: _Tensor,
  };
}

export default useInference;
