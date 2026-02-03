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
     * Get local file path for an asset
     */
    const getAssetLocalUri = useCallback(async (assetModule) => {
        try {
            const asset = Asset.fromModule(assetModule);
            await asset.downloadAsync();

            // For Android, we need the local file URI
            if (asset.localUri) {
                return asset.localUri;
            }

            // Fallback: copy to document directory
            const fileName = asset.name + (asset.type ? '.' + asset.type : '');
            const destPath = FileSystem.documentDirectory + fileName;

            // Check if already copied
            const fileInfo = await FileSystem.getInfoAsync(destPath);
            if (!fileInfo.exists) {
                await FileSystem.downloadAsync(asset.uri, destPath);
            }

            return destPath;
        } catch (err) {
            console.error('Asset loading error:', err);
            throw err;
        }
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
            return;
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
     * Run decoder on audio data
     * @param {Float32Array} audioData - Audio samples at 8kHz
     * @returns {Object} { bits128, confidence, frameProbs }
     */
    const runDecoder = useCallback(async (audioData) => {
        if (!decoderSessionRef.current) {
            throw new Error('Decoder not loaded');
        }

        try {
            const session = decoderSessionRef.current;

            // Calculate number of complete frames
            const numFrames = Math.floor(audioData.length / FRAME_SAMPLES);
            if (numFrames === 0) {
                throw new Error('Audio too short');
            }

            // Process frames and aggregate probabilities
            const bitAccumulators = new Float32Array(PAYLOAD_LENGTH).fill(0);
            const bitCounts = new Float32Array(PAYLOAD_LENGTH).fill(0);

            for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
                const start = frameIdx * FRAME_SAMPLES;
                const frameData = audioData.slice(start, start + FRAME_SAMPLES);

                // Create input tensor [1, 1, 320] - [batch, channel, samples]
                const inputTensor = new Tensor('float32', Array.from(frameData), [1, 1, FRAME_SAMPLES]);

                // Run inference
                const feeds = {};
                feeds[session.inputNames[0]] = inputTensor;
                const results = await session.run(feeds);

                // Get output probability (single value per frame)
                const outputName = session.outputNames[0];
                const prob = results[outputName].data[0];

                // Map frame to bit position (cyclic)
                const bitIdx = frameIdx % PAYLOAD_LENGTH;
                bitAccumulators[bitIdx] += prob;
                bitCounts[bitIdx] += 1;
            }

            // Calculate averaged probabilities for each bit
            const bits128 = new Float32Array(PAYLOAD_LENGTH);
            for (let i = 0; i < PAYLOAD_LENGTH; i++) {
                bits128[i] = bitCounts[i] > 0 ? bitAccumulators[i] / bitCounts[i] : 0.5;
            }

            // Calculate overall confidence
            const confidences = bits128.map(p => Math.abs(p - 0.5) * 2);
            const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;

            return {
                bits128,
                confidence: avgConfidence,
                numFrames,
            };
        } catch (err) {
            console.error('Decoder inference error:', err);
            throw err;
        }
    }, []);

    /**
     * Run encoder on audio data with message
     * @param {Float32Array} audioData - Audio samples at 8kHz
     * @param {number[]} messageBits - 128-bit message
     * @returns {Object} { encoded }
     */
    const runEncoder = useCallback(async (audioData, messageBits) => {
        if (!encoderSessionRef.current) {
            throw new Error('Encoder not loaded');
        }

        try {
            const session = encoderSessionRef.current;

            // Calculate number of complete frames
            const numFrames = Math.floor(audioData.length / FRAME_SAMPLES);
            if (numFrames === 0) {
                throw new Error('Audio too short');
            }

            // Prepare full message tensor [1, 128]
            const messageArray = new Float32Array(PAYLOAD_LENGTH);
            for (let i = 0; i < PAYLOAD_LENGTH; i++) {
                messageArray[i] = messageBits[i] || 0;
            }
            const messageTensor = new Tensor('float32', messageArray, [1, PAYLOAD_LENGTH]);

            // Output encoded audio
            const encoded = new Float32Array(audioData.length);
            encoded.set(audioData); // Start with original

            for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
                const start = frameIdx * FRAME_SAMPLES;
                const frameData = audioData.slice(start, start + FRAME_SAMPLES);

                // Create audio tensor - [1, 1, 320]
                const audioTensor = new Tensor('float32', Array.from(frameData), [1, 1, FRAME_SAMPLES]);

                // Run inference with audio and full message
                const feeds = {};
                feeds[session.inputNames[0]] = audioTensor;
                if (session.inputNames.length > 1) {
                    feeds[session.inputNames[1]] = messageTensor;
                }
                const results = await session.run(feeds);

                // Get encoded frame
                const outputName = session.outputNames[0];
                const encodedFrame = results[outputName].data;

                // Copy to output
                for (let i = 0; i < FRAME_SAMPLES && start + i < encoded.length; i++) {
                    encoded[start + i] = encodedFrame[i];
                }
            }

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
    };
}

export default useInference;
