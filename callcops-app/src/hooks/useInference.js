/**
 * useInference Hook for React Native
 * Uses onnxruntime-react-native for ONNX model inference
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import * as FileSystem from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';

// Import ONNX runtime (will be null if not available)
let InferenceSession = null;
let Tensor = null;

try {
    const ort = require('onnxruntime-react-native');
    InferenceSession = ort.InferenceSession;
    Tensor = ort.Tensor;
} catch (e) {
    console.warn('onnxruntime-react-native not available:', e);
}

// Frame-Wise Constants (match Python model)
const FRAME_SAMPLES = 320;      // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;     // 128-bit cyclic payload
const SAMPLE_RATE = 8000;

export function useInference() {
    const [isLoading, setIsLoading] = useState(false);
    const [isReady, setIsReady] = useState(false);
    const [error, setError] = useState(null);
    const [onnxAvailable, setOnnxAvailable] = useState(!!InferenceSession);

    const decoderSessionRef = useRef(null);
    const encoderSessionRef = useRef(null);

    useEffect(() => {
        if (!InferenceSession) {
            setError('ONNX Runtime not available on this device');
        }
    }, []);

    /**
     * Get local file path for an asset with verification and retry logic
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
     * Load decoder model
     */
    const loadDecoder = useCallback(async () => {
        if (!InferenceSession) {
            throw new Error('ONNX Runtime not available');
        }

        if (decoderSessionRef.current) {
            setIsReady(true);
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            // Get asset path
            const modelAsset = require('../../assets/models/decoder.onnx');
            const modelPath = await getAssetLocalUri(modelAsset);

            console.log('Loading decoder from:', modelPath);

            // Create inference session
            const session = await InferenceSession.create(modelPath, {
                executionProviders: ['cpu'],
            });

            decoderSessionRef.current = session;
            setIsReady(true);
            console.log('Decoder loaded successfully');
            console.log('Input names:', session.inputNames);
            console.log('Output names:', session.outputNames);
        } catch (err) {
            console.error('Failed to load decoder:', err);
            setError(`Failed to load decoder: ${err.message}`);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [getAssetLocalUri]);

    /**
     * Load encoder model
     */
    const loadEncoder = useCallback(async () => {
        if (!InferenceSession) {
            throw new Error('ONNX Runtime not available');
        }

        if (encoderSessionRef.current) {
            return encoderSessionRef.current;
        }

        setIsLoading(true);
        setError(null);

        try {
            const modelAsset = require('../../assets/models/encoder.onnx');
            const modelPath = await getAssetLocalUri(modelAsset);

            console.log('Loading encoder from:', modelPath);

            const session = await InferenceSession.create(modelPath, {
                executionProviders: ['cpu'],
            });

            encoderSessionRef.current = session;
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
    }, [getAssetLocalUri]);

    /**
     * Resample audio to 8kHz mono
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
            // CRITICAL: Pass Float32Array directly, NOT Array.from() conversion
            const inputTensor = new Tensor('float32', alignedAudio, [1, 1, alignedAudio.length]);

            // Run inference with explicit input name matching model schema
            const feeds = { audio: inputTensor };
            const results = await session.run(feeds);

            // Get bit_probs output with explicit name (not dynamic outputNames[0])
            const outputData = results.bit_probs.data; // Float32Array

            // Aggregate probabilities using CYCLIC method (like frontend)
            // Each frame probability maps to bitIdx = frameIndex % 128
            const bitAccumulators = new Float32Array(PAYLOAD_LENGTH).fill(0);
            const bitCounts = new Float32Array(PAYLOAD_LENGTH).fill(0);

            for (let i = 0; i < outputData.length; i++) {
                const bitIdx = i % PAYLOAD_LENGTH;  // Cyclic mapping!
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
                rawScores: outputData, // Return raw frame probabilities for streaming viz
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

            // Prepare tensors - CRITICAL: Use Float32Array directly, no Array.from()
            // 1. Audio: [1, 1, T]
            const audioTensor = new Tensor('float32', alignedAudio, [1, 1, alignedAudio.length]);

            // 2. Message: [1, 128]
            const messageArray = new Float32Array(PAYLOAD_LENGTH);
            for (let i = 0; i < PAYLOAD_LENGTH; i++) {
                messageArray[i] = messageBits[i] || 0;
            }
            const messageTensor = new Tensor('float32', messageArray, [1, PAYLOAD_LENGTH]);

            // Run inference with explicit input names matching model schema
            const feeds = {
                audio: audioTensor,
                message: messageTensor,
            };

            const results = await session.run(feeds);

            // Get encoded output with explicit name
            const outputData = results.watermarked.data; // Float32Array

            const encoded = new Float32Array(outputData);
            return { encoded };
        } catch (err) {
            console.error('Encoder inference error:', err);
            throw err;
        }
    }, []);

    return {
        isLoading,
        isReady,
        error,
        onnxAvailable,
        loadDecoder,
        loadEncoder,
        runDecoder,
        runEncoder,
        resampleTo8kHz,
        encoderSession: encoderSessionRef.current,
        decoderSession: decoderSessionRef.current,
        Tensor,
    };
}

export default useInference;
