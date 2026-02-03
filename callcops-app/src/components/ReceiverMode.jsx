/**
 * ReceiverMode - File upload and real ONNX watermark detection
 * Uses onnxruntime for decoding with simple fallback
 */
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ScrollView, PanResponder, Animated, Dimensions } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { useInference } from '../hooks/useInference';
import { verifyCRC, attemptCorrection } from '../utils/crc';
import { Audio } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const SAMPLE_RATE = 8000;
const FRAME_SIZE = 320;
const PAYLOAD_LENGTH = 128;

export default function ReceiverMode({ onBack }) {
    const [state, setState] = useState('idle');
    const [fileName, setFileName] = useState(null);
    const [progress, setProgress] = useState(0);
    const [isValid, setIsValid] = useState(null);
    const [crcValid, setCrcValid] = useState(null);
    const [crcInfo, setCrcInfo] = useState({ expected: null, actual: null }); // CRC details
    const [confidence, setConfidence] = useState(0);
    const [bitProbs, setBitProbs] = useState(null);

    // Streaming UI State (New)
    const [currentChunkProbs, setCurrentChunkProbs] = useState(null);
    const [activeIndices, setActiveIndices] = useState(new Set());
    const [realtimeValid, setRealtimeValid] = useState(false);
    const [showDetails, setShowDetails] = useState(false); // Toggle for CRC details
    const [instantWaveform, setInstantWaveform] = useState(new Array(80).fill(5)); // Live EQ (80 bars)
    const [currentBitInfo, setCurrentBitInfo] = useState({ index: -1, prob: 0.5 }); // Live Bit Monitor
    const [callDuration, setCallDuration] = useState(0); // For iPhone style timer
    const [showSecurity, setShowSecurity] = useState(false); // Toggle for AI monitors

    // Animation states
    const [timeSeriesData, setTimeSeriesData] = useState([]);
    const [isAnimating, setIsAnimating] = useState(false);
    const timerRef = useRef(null);
    const [errorMessage, setErrorMessage] = useState(null);
    const [usedOnnx, setUsedOnnx] = useState(false);

    // Streaming state refs
    const accumulatorsRef = useRef(new Float32Array(PAYLOAD_LENGTH).fill(0));
    const countsRef = useRef(new Float32Array(PAYLOAD_LENGTH).fill(0));
    const latestProbsRef = useRef(new Float32Array(PAYLOAD_LENGTH).fill(0.5)); // Persist latest instant val
    const displayedBitProbsRef = useRef(new Float32Array(PAYLOAD_LENGTH).fill(0.5)); // For animated bottom grid
    const audioDataRef = useRef(null);
    const processingOffsetRef = useRef(0);
    const soundRef = useRef(null);

    // Slide to Answer State
    const pan = useRef(new Animated.ValueXY()).current;
    const [sliderWidth, setSliderWidth] = useState(0);
    const SCREEN_WIDTH = Dimensions.get('window').width;
    const SLIDER_WIDTH = SCREEN_WIDTH * 0.8;
    const KNOB_SIZE = 76;
    const MAX_SLIDE = SLIDER_WIDTH - KNOB_SIZE - 8; // 8 is padding

    // Ringtone Logic
    useEffect(() => {
        let ringtoneObject = null;

        const playRingtone = async () => {
            if (state === 'idle') {
                try {
                    // Try to play ringtone if file exists (commented out for now as file doesn't exist)
                    // const { sound } = await Audio.Sound.createAsync(require('../../assets/ringtone.mp3'), { isLooping: true });
                    // ringtoneObject = sound;
                    // await sound.playAsync();
                } catch (error) {
                    console.warn('Ringtone playback failed (file might be missing):', error);
                }
            }
        };

        playRingtone();

        return () => {
            if (ringtoneObject) {
                ringtoneObject.stopAsync();
                ringtoneObject.unloadAsync();
            }
        };
    }, [state]);

    const panResponder = useRef(
        PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onPanResponderMove: (_, gestureState) => {
                const newX = gestureState.dx;
                if (newX >= 0 && newX <= MAX_SLIDE) {
                    pan.setValue({ x: newX, y: 0 });
                }
            },
            onPanResponderRelease: (_, gestureState) => {
                if (gestureState.dx > MAX_SLIDE * 0.8) {
                    // Successful slide
                    Animated.timing(pan, {
                        toValue: { x: MAX_SLIDE, y: 0 },
                        duration: 200,
                        useNativeDriver: false,
                    }).start(() => {
                        handlePickFile();
                        // Reset slider after tiny delay/transition
                        setTimeout(() => pan.setValue({ x: 0, y: 0 }), 1000);
                    });
                } else {
                    // Snap back
                    Animated.spring(pan, {
                        toValue: { x: 0, y: 0 },
                        friction: 5,
                        useNativeDriver: false,
                    }).start();
                }
            },
        })
    ).current;

    // Cleanup timer and sound on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (soundRef.current) {
                soundRef.current.unloadAsync();
            }
        };
    }, []);

    const inference = useInference();

    // Call Timer Effect (iPhone Style)
    useEffect(() => {
        let interval;
        if (state === 'analyzing') {
            setCallDuration(0);
            interval = setInterval(() => {
                setCallDuration(prev => prev + 1);
            }, 1000);
        }
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [state]);

    // Format seconds to MM:SS
    const formatDuration = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // Read WAV file and get audio samples
    const readWavFile = async (uri) => {
        try {
            // Optimization: Increased to 50MB to support longer files (approx 5 mins HQ / 50 mins LowQ)
            // Was 2MB which cut off after ~10s of HQ audio
            const MAX_READ_SIZE = 50 * 1024 * 1024;

            const fileInfo = await FileSystem.getInfoAsync(uri);
            if (!fileInfo.exists) throw new Error('File not found');

            const readLength = Math.min(fileInfo.size, MAX_READ_SIZE);

            const base64 = await FileSystem.readAsStringAsync(uri, {
                encoding: 'base64',
                length: readLength, // Partial read
            });

            // Standard base64 decode
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }

            const view = new DataView(bytes.buffer);

            // Parse WAV header
            const sampleRate = view.getUint32(24, true);
            const bitsPerSample = view.getUint16(34, true);
            // const numChannels = view.getUint16(22, true);
            const bytesPerSample = bitsPerSample / 8;

            // Find data chunk
            let dataOffset = 12;
            let foundData = false;

            while (dataOffset < bytes.length - 8) {
                const chunkId = String.fromCharCode(bytes[dataOffset], bytes[dataOffset + 1], bytes[dataOffset + 2], bytes[dataOffset + 3]);
                const chunkSize = view.getUint32(dataOffset + 4, true);

                if (chunkId === 'data') {
                    dataOffset += 8;
                    foundData = true;
                    break;
                }
                dataOffset += 8 + chunkSize;
            }

            if (!foundData) {
                // If header is cut off or simple format, assume header 44
                console.warn('WAV data chunk not found in first 2MB, assuming standard offset 44');
                dataOffset = 44;
            }

            // Calculate samples available in this chunk
            const numSamples = Math.floor((bytes.length - dataOffset) / bytesPerSample);
            const samples = new Float32Array(numSamples);

            if (bitsPerSample === 16) {
                for (let i = 0; i < numSamples; i++) {
                    const offset = dataOffset + i * 2;
                    if (offset + 1 < bytes.length) {
                        const sample = view.getInt16(offset, true);
                        samples[i] = sample / 32768.0;
                    }
                }
            } else if (bitsPerSample === 32) {
                for (let i = 0; i < numSamples; i++) {
                    const offset = dataOffset + i * 4;
                    if (offset + 3 < bytes.length) {
                        samples[i] = view.getFloat32(offset, true);
                    }
                }
            }

            return { samples, sampleRate };
        } catch (err) {
            console.error('WAV parsing error:', err);
            throw new Error('WAV 파일을 읽을 수 없습니다');
        }
    };

    // Fallback: Simple watermark detection (matches embedding)
    const detectWatermarkSimple = (samples, globalOffset = 0) => {
        // Debug signal
        let maxAmp = 0;
        for (let k = 0; k < Math.min(samples.length, 100); k++) maxAmp = Math.max(maxAmp, Math.abs(samples[k]));
        console.log(`[Detector] Offset: ${globalOffset}, MaxAmp(first100): ${maxAmp}`);

        const numFrames = Math.floor(samples.length / FRAME_SIZE);

        const bitAccumulators = new Float32Array(PAYLOAD_LENGTH).fill(0);
        const bitCounts = new Float32Array(PAYLOAD_LENGTH).fill(0);

        for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            const start = frameIdx * FRAME_SIZE;
            // Correct global phase: (globalOffset + start)
            const globalPhase = globalOffset + start;

            // To find which bit this frame belongs to, we need global frame index
            // Global Sample Index / Frame Size
            const globalFrameIdx = Math.floor(globalPhase / FRAME_SIZE);
            const bitIdx = globalFrameIdx % PAYLOAD_LENGTH;

            let correlation = 0;
            for (let i = 0; i < FRAME_SIZE; i++) {
                const expected = Math.sin(2 * Math.PI * (globalPhase + i) / 10);
                correlation += samples[start + i] * expected;
            }

            correlation /= FRAME_SIZE;

            bitAccumulators[bitIdx] += correlation;
            bitCounts[bitIdx] += 1;
        }

        const bits128 = new Float32Array(PAYLOAD_LENGTH);

        for (let i = 0; i < PAYLOAD_LENGTH; i++) {
            if (bitCounts[i] > 0) {
                const avg = bitAccumulators[i] / bitCounts[i];
                bits128[i] = 1 / (1 + Math.exp(-avg * 1000));
            } else {
                bits128[i] = 0.5;
            }
        }

        const confidences = bits128.map(p => Math.abs(p - 0.5) * 2);
        const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;

        return { bits128, confidence: avgConfidence, numFrames };
    };

    const handlePickFile = async () => {
        setErrorMessage(null);

        try {
            const result = await DocumentPicker.getDocumentAsync({
                type: 'audio/*',
                copyToCacheDir: true,
            });

            if (!result.canceled && result.assets && result.assets[0]) {
                const file = result.assets[0];
                setFileName(file.name);
                await analyzeFile(file.uri);
            }
        } catch (err) {
            console.error('File picker error:', err);
            Alert.alert('오류', '파일을 선택할 수 없습니다.');
        }
    };

    const analyzeFile = async (fileUri) => {
        setState('analyzing');
        setProgress(0);
        setUsedOnnx(false);
        setErrorMessage(null);

        try {
            // 1. Read File (Quickly)
            const { samples, sampleRate } = await readWavFile(fileUri);
            console.log('Audio loaded:', samples.length);

            // 2. Resample
            const audio8k = inference.resampleTo8kHz(samples, sampleRate);
            audioDataRef.current = audio8k;
            processingOffsetRef.current = 0;

            // Calculate waveform bars (normalize relative to global max)
            const numBars = 48;
            const step = Math.floor(audio8k.length / numBars);
            const heights = [];
            let globalMax = 0;
            const rawMaxes = [];

            for (let i = 0; i < numBars; i++) {
                let max = 0;
                const start = i * step;
                const end = Math.min(start + step, audio8k.length);
                const localStep = Math.max(1, Math.floor((end - start) / 20)); // Check 20 points per bar

                for (let j = start; j < end; j += localStep) {
                    const val = Math.abs(audio8k[j]);
                    if (val > max) max = val;
                }
                rawMaxes.push(max);
                if (max > globalMax) globalMax = max;
            }

            if (globalMax < 0.01) globalMax = 0.01; // Avoid divide by zero

            for (let i = 0; i < numBars; i++) {
                // Min 5%, Max 100%
                const normalized = (rawMaxes[i] / globalMax) * 100;
                heights.push(Math.max(5, normalized));
            }


            // Reset accumulators and display refs
            accumulatorsRef.current.fill(0);
            countsRef.current.fill(0);
            latestProbsRef.current.fill(-1);  // -1 = not yet detected (will show dim)
            displayedBitProbsRef.current.fill(-1);
            setCurrentChunkProbs(null);  // Clear instant grid
            setBitProbs(null);  // Clear accumulated grid

            // Load Model with dynamic ONNX check (handles cold start)
            // Use checkOnnxAvailable() for fresh check instead of stale state
            const onnxReady = inference.checkOnnxAvailable
                ? inference.checkOnnxAvailable()
                : inference.onnxAvailable;

            if (onnxReady) {
                try {
                    await inference.loadDecoder();
                    console.log('ONNX decoder loaded successfully');
                } catch (e) {
                    console.warn('ONNX decoder load failed, will use fallback:', e.message);
                }
            }

            // Play Audio
            const { sound } = await Audio.Sound.createAsync({ uri: fileUri });
            soundRef.current = sound;
            await sound.playAsync();



            // Start Streaming Loop
            processNextChunk();

        } catch (err) {
            console.error('Analysis error:', err);
            setErrorMessage(`오류: ${err.message}`);
            setState('idle');
        }
    };

    const processNextChunk = async () => {
        if (!audioDataRef.current) return;

        const offset = processingOffsetRef.current;
        const totalLen = audioDataRef.current.length;

        // Chunk size: 3200 samples (0.4s at 8kHz) - 10 Frames (Accurate detection)
        const CHUNK_SIZE = 3200;

        if (offset >= totalLen) {
            finishAnalysis();
            return;
        }

        console.log(`[Flow] Offset: ${offset}, Len: ${totalLen}, ChunkSize: ${CHUNK_SIZE}`);

        const end = Math.min(offset + CHUNK_SIZE, totalLen);

        // CAUSAL ACCUMULATION: Process all audio from start to current point
        // This maintains context without using future data (real-time compatible)
        const audioSoFar = audioDataRef.current.slice(0, end);
        const chunk = audioDataRef.current.slice(offset, end); // For fallback

        try {
            let chunkResult;
            if (inference.onnxAvailable) {
                // Process ALL audio received so far (causal - no future data)
                chunkResult = await inference.runDecoder(audioSoFar);
                setUsedOnnx(true);
            } else {
                // Fallback still uses chunk for performance
                chunkResult = detectWatermarkSimple(chunk, offset);
            }

            const chunkProbs = chunkResult.bits128; // Already aggregated by ONNX

            // For animation: identify which frames were added in this chunk
            const startFrame = Math.floor(offset / FRAME_SIZE);
            const endFrame = Math.floor(end / FRAME_SIZE);
            const framesToAnimate = []; // { bitIdx, prob }

            if (inference.onnxAvailable) {
                // ONNX with causal accumulation: bits128 is already the full result
                // Just animate the NEW frames (from startFrame to endFrame)
                for (let f = startFrame; f < endFrame; f++) {
                    const bitIdx = f % PAYLOAD_LENGTH;
                    const prob = chunkProbs[bitIdx]; // Use aggregated result
                    framesToAnimate.push({ bitIdx, prob });
                }
                // Use bits128 directly as running probs (already averaged)
            } else {
                // Fallback: manual accumulation for chunk
                const numFramesInChunk = Math.floor(chunk.length / FRAME_SIZE);

                for (let j = 0; j < numFramesInChunk; j++) {
                    const bitIdx = (startFrame + j) % PAYLOAD_LENGTH;
                    const prob = chunkProbs[bitIdx];

                    accumulatorsRef.current[bitIdx] += prob;
                    countsRef.current[bitIdx] += 1;

                    framesToAnimate.push({ bitIdx, prob });
                }
            }

            // Get running probabilities
            let runningProbs;
            if (inference.onnxAvailable) {
                // ONNX already computed full aggregation
                runningProbs = chunkProbs;
            } else {
                // Fallback: calculate from accumulators
                runningProbs = new Float32Array(PAYLOAD_LENGTH);
                for (let i = 0; i < PAYLOAD_LENGTH; i++) {
                    const count = countsRef.current[i];
                    runningProbs[i] = count > 0 ? accumulatorsRef.current[i] / count : 0.5;
                }
            }

            // Real-time Validity Check
            const rtResult = verifyCRC(runningProbs);
            setRealtimeValid(rtResult.isValid);

            // Progress
            const progressRaw = (end / totalLen) * 100;
            setProgress(Math.floor(progressRaw));

            // Animated Bit Reveal (40ms per bit) - Color updates WITH highlight!
            let animIdx = 0;

            const animateBit = () => {
                if (animIdx >= framesToAnimate.length) {
                    // Animation done, update confidence and move to next chunk
                    let totalConf = 0;
                    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
                        totalConf += Math.abs(displayedBitProbsRef.current[i] - 0.5) * 2;
                    }
                    setConfidence(totalConf / PAYLOAD_LENGTH);

                    processingOffsetRef.current = end;
                    processNextChunk();
                    return;
                }

                // Get current frame data
                const { bitIdx, prob } = framesToAnimate[animIdx];

                // 1. Live Waveform Calculation (EQ Style)
                const currentSamplePos = offset + (animIdx * FRAME_SIZE);
                const windowSize = 640; // 2 frames context
                const start = Math.max(0, currentSamplePos - windowSize / 2);
                const wEnd = Math.min(totalLen, currentSamplePos + windowSize / 2);
                const bars = [];

                // 80 bars (High resolution)
                const step = Math.floor((wEnd - start) / 80) || 1;
                for (let i = 0; i < 80; i++) {
                    let max = 0;
                    const s = start + i * step;
                    const e = Math.min(wEnd, s + step);
                    for (let k = s; k < e; k += Math.max(1, Math.floor((e - s) / 3))) {
                        const val = Math.abs(audioDataRef.current[k]);
                        if (val > max) max = val;
                    }
                    bars.push(Math.min(100, max * 500));
                }
                setInstantWaveform(bars);
                setCurrentBitInfo({ index: bitIdx, prob });

                // Update TOP grid (Instant) for THIS bit NOW
                latestProbsRef.current[bitIdx] = prob;
                setCurrentChunkProbs(new Float32Array(latestProbsRef.current));

                // Update BOTTOM grid (Accumulated) for THIS bit NOW
                displayedBitProbsRef.current[bitIdx] = runningProbs[bitIdx];
                setBitProbs(new Float32Array(displayedBitProbsRef.current));

                // Highlight only the current bit
                setActiveIndices(new Set([bitIdx]));

                animIdx++;
                setTimeout(animateBit, 40);
            };

            animateBit();

        } catch (err) {
            console.error('Chunk processing error:', err);
            finishAnalysis();
        }
    };

    const finishAnalysis = () => {
        // Use Ref for final check to avoid stale state
        const finalProbs = displayedBitProbsRef.current ?
            new Float32Array(displayedBitProbsRef.current) :
            new Float32Array(128).fill(0.5);

        // First, check raw CRC
        const rawCrcResult = verifyCRC(finalProbs);
        console.log('Raw CRC:', rawCrcResult);

        // Attempt error correction if CRC fails
        let usedProbs = finalProbs;
        let correctionResult = null;

        if (!rawCrcResult.isValid) {
            correctionResult = attemptCorrection(finalProbs);
            console.log('Correction attempt:', correctionResult);

            if (correctionResult.success) {
                usedProbs = correctionResult.corrected;
                setBitProbs(usedProbs); // Update display with corrected bits
            }
        }

        // Final CRC Check (after correction)
        const crcResult = verifyCRC(usedProbs);
        console.log('Final CRC:', crcResult);

        const crcPassed = crcResult.isValid;

        setCrcValid(crcPassed);
        setCrcInfo({
            expected: crcResult.expectedCRC,
            actual: crcResult.actualCRC
        });

        // Logic Update: If CRC passes, it IS valid regardless of confidence
        // CRC is a strong mathematical proof of integrity
        setIsValid(crcPassed);
        setState('result');
    };

    const handleReset = async () => {
        if (soundRef.current) {
            await soundRef.current.stopAsync();
            await soundRef.current.unloadAsync();
            soundRef.current = null;
        }
        setState('idle');
        setFileName(null);
        setProgress(0);
        setIsValid(null);
        setCrcValid(null);
        setCrcInfo({ expected: null, actual: null });
        setConfidence(0);
        setBitProbs(null);
        setErrorMessage(null);
        setUsedOnnx(false);
    };

    // Render Dual Bit Grids
    const renderBitMatrix = (showInstant = true, isResult = false) => {
        const instantData = currentChunkProbs ? Array.from(currentChunkProbs) : new Array(128).fill(-1);
        const accumulatedData = bitProbs ? Array.from(bitProbs) : new Array(128).fill(-1);

        const renderGrid = (data, isInstant) => {
            return (
                <View style={styles.gridWrapper}>
                    <Text style={styles.gridLabel}>{isInstant ? 'Real-time Signal (Instant)' : 'Accumulated Result (Final)'}</Text>
                    <View style={styles.gridContainer}>
                        {data.map((prob, i) => {
                            let bgColor;
                            if (isInstant) {
                                if (prob === -1) bgColor = '#1f2937'; // Dim (Unknown)
                                else {
                                    // Grayscale mapping
                                    const val = Math.floor(prob * 255);
                                    bgColor = `rgb(${val}, ${val}, ${val})`;
                                }
                            } else {
                                // Accumulated: Binary Threshold (or dim if not yet detected)
                                if (prob === -1 || prob === null || prob === undefined) {
                                    bgColor = '#1f2937'; // Dim (Unknown)
                                } else {
                                    bgColor = prob > 0.5 ? '#ffffff' : '#000000';
                                }
                            }

                            const isActive = activeIndices.has(i);
                            // If it's result page, we DO NOT show the cyan active border
                            const showActiveBorder = isActive && !isResult;

                            return (
                                <View
                                    key={i}
                                    style={[
                                        styles.gridBit,
                                        {
                                            backgroundColor: bgColor,
                                            borderColor: showActiveBorder ? '#00ffff' : '#374151',
                                            borderWidth: showActiveBorder ? 2 : 0.5,
                                            zIndex: showActiveBorder ? 10 : 1
                                        }
                                    ]}
                                />
                            );
                        })}
                    </View>
                </View>
            );
        };

        return (
            <View style={styles.dualMatrixContainer}>
                {showInstant && renderGrid(instantData, true)}

                {showInstant && <View style={styles.spacer} />}

                {renderGrid(accumulatedData, false)}
            </View>
        );
    };

    // Miniaturized Bit Matrix for iPhone Call Mode (Supports Instant and Accumulated)
    const renderMiniBitMatrix = (isInstant = false) => {
        const data = isInstant
            ? (currentChunkProbs ? Array.from(currentChunkProbs) : new Array(128).fill(-1))
            : (bitProbs ? Array.from(bitProbs) : new Array(128).fill(-1));

        return (
            <View style={styles.miniMatrixGrid}>
                {data.map((prob, i) => {
                    let bgColor = '#1f2937'; // Unknown
                    if (prob !== -1 && prob !== null) {
                        if (isInstant) {
                            const val = Math.floor(prob * 255);
                            bgColor = `rgb(${val}, ${val}, ${val})`;
                        } else {
                            bgColor = prob > 0.5 ? '#ffffff' : '#000';
                        }
                    }
                    const isActive = activeIndices.has(i);
                    return (
                        <View
                            key={i}
                            style={[
                                styles.miniGridBit,
                                {
                                    backgroundColor: bgColor,
                                    borderColor: isActive ? '#00ffff' : 'rgba(255,255,255,0.05)',
                                    borderWidth: isActive ? 1 : 0.2,
                                }
                            ]}
                        />
                    );
                })}
            </View>
        );
    };

    // Waveform Visualization (Live Mode)
    const renderWaveform = () => {
        // Decide bit value: > 0.5 is 1 (Cyan), else 0 (Gray/Red)
        const bitVal = currentBitInfo.prob > 0.5 ? 1 : 0;
        const bitColor = bitVal === 1 ? '#00ffff' : '#ef4444'; // Cyan vs Red
        const confidencePct = Math.floor(Math.abs(currentBitInfo.prob - 0.5) * 200); // 0-100%

        return (
            <View style={styles.liveContainer}>
                {/* Left: EQ Visualizer */}
                <View style={styles.eqContainer}>
                    {instantWaveform.map((height, i) => (
                        <View
                            key={i}
                            style={[
                                styles.eqBar,
                                { height: `${Math.max(5, height)}%` }
                            ]}
                        />
                    ))}
                </View>

                {/* Right: Bit Monitor */}
                <View style={styles.monitorContainer}>
                    <Text style={styles.monitorLabel}>SCANNING BIT</Text>
                    <Text style={styles.bitIndex}>#{currentBitInfo.index}</Text>

                    <View style={styles.decisionBox}>
                        <Text style={[styles.bitValue, { color: bitColor }]}>
                            {currentBitInfo.index === -1 ? '-' : bitVal}
                        </Text>
                        <Text style={styles.confidenceMini}>{confidencePct}% Conf.</Text>
                    </View>
                </View>
            </View>
        );
    };

    const renderWaveformOld = () => {
        // If no audio data, return simple placeholder or nothing
        if (!audioDataRef.current) return null;

        const numBars = 48; // Number of bars to display
        const samplesPerBar = Math.floor(audioDataRef.current.length / numBars);
        const bars = [];

        // Use progress state to determine played portion
        // progress is 0-100
        const currentBarIndex = Math.floor((progress / 100) * numBars);

        for (let i = 0; i < numBars; i++) {
            // Find max amplitude in this segment
            let maxAmp = 0;
            const startIdx = i * samplesPerBar;
            const endIdx = Math.min(startIdx + samplesPerBar, audioDataRef.current.length);

            // Optimization: check step-wise to save CPU
            const step = Math.max(1, Math.floor((endIdx - startIdx) / 10)); // check 10 points per bar
            for (let j = startIdx; j < endIdx; j += step) {
                const amp = Math.abs(audioDataRef.current[j]);
                if (amp > maxAmp) maxAmp = amp;
            }

            // Normalize height (min 10%, max 100%)
            const heightPercent = Math.max(10, Math.min(100, maxAmp * 100));
            const isPlayed = i <= currentBarIndex;

            bars.push(
                <View
                    key={i}
                    style={[
                        styles.waveBar,
                        {
                            height: `${heightPercent}%`,
                            backgroundColor: isPlayed ? '#00ffff' : '#374151'
                        }
                    ]}
                />
            );
        }

        return (
            <View style={styles.waveformContainer}>
                {bars}
            </View>
        );
    };

    // Shimmer Text Component
    const ShimmerText = ({ text, style }) => {
        const animatedValues = useRef([]);
        const textLen = text.length;

        if (animatedValues.current.length !== textLen) {
            animatedValues.current = Array(textLen).fill(0).map(() => new Animated.Value(0));
        }

        useEffect(() => {
            const createAnimation = (index) => {
                return Animated.sequence([
                    Animated.timing(animatedValues.current[index], {
                        toValue: 1,
                        duration: 300,
                        useNativeDriver: true
                    }),
                    Animated.timing(animatedValues.current[index], {
                        toValue: 0,
                        duration: 1500, // Long fade out for trail effect
                        useNativeDriver: true
                    })
                ]);
            };

            const animations = text.split('').map((_, i) => createAnimation(i));

            // Stagger the animations to create a wave
            Animated.loop(
                Animated.stagger(100, animations) // 100ms delay between each letter
            ).start();
        }, [text]);

        return (
            <View style={{ flexDirection: 'row' }}>
                {text.split('').map((char, i) => {
                    const opacity = animatedValues.current[i].interpolate({
                        inputRange: [0, 1],
                        outputRange: [0.3, 1] // Gray (0.3) to White (1)
                    });
                    return (
                        <Animated.Text key={i} style={[style, { opacity }]}>
                            {char}
                        </Animated.Text>
                    );
                })}
            </View>
        );
    };

    // ... inside ReceiverMode ...

    // IDLE STATE
    if (state === 'idle') {
        return (
            <View style={{ flex: 1 }}>
                <LinearGradient
                    colors={['#7c2d12', '#831843', '#2e1065']} // Dark Orange -> Dark Pink -> Dark Purple
                    style={StyleSheet.absoluteFill}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                />

                <TouchableOpacity onPress={onBack} style={styles.backButtonTop}>
                    <Ionicons name="chevron-back" size={32} color="#fff" />
                </TouchableOpacity>

                <View style={[styles.centerContent, { justifyContent: 'flex-start', paddingTop: 130 }]}>
                    <Text style={styles.callerName}>070-7079-5431</Text>
                    <Text style={styles.callerNumber}>대한민국</Text>
                </View>

                {/* iPhone Slide to Answer UI */}
                <View style={styles.bottomControlArea}>

                    {/* Action Buttons Row - Narrowed and Moved Up */}
                    <View style={styles.actionButtonsRow}>
                        <TouchableOpacity style={styles.actionColumn}>
                            <Ionicons name="alarm" size={24} color="#fff" style={{ marginBottom: 8 }} />
                            <Text style={styles.iconLabel}>나중에 보기</Text>
                        </TouchableOpacity>

                        <TouchableOpacity style={styles.actionColumn}>
                            <Ionicons name="chatbubble" size={24} color="#fff" style={{ marginBottom: 8 }} />
                            <Text style={styles.iconLabel}>메시지</Text>
                        </TouchableOpacity>
                    </View>

                    {/* Slider */}
                    <View style={styles.sliderContainer}>
                        <View style={[styles.sliderTrack, { backgroundColor: 'rgba(255, 255, 255, 0.2)' }]}>
                            {/* Masked Text Container */}
                            <Animated.View
                                style={{
                                    position: 'absolute',
                                    left: pan.x, // Mask moves with knob
                                    right: 0,
                                    height: '100%',
                                    overflow: 'hidden', // Clips content to the left of pan.x
                                    justifyContent: 'center',
                                }}
                            >
                                <Animated.View
                                    style={{
                                        width: Dimensions.get('window').width * 0.85, // Full Text Width
                                        alignItems: 'center', // Center Text
                                        justifyContent: 'center',
                                        marginLeft: 20, // Shift slightly right
                                        transform: [{ translateX: Animated.multiply(pan.x, -1) }] // Counteract mask movement to keep text static
                                    }}
                                >
                                    <ShimmerText text="밀어서 통화하기" style={styles.shimmerText} />
                                </Animated.View>
                            </Animated.View>
                        </View>

                        <Animated.View
                            style={{
                                transform: [{ translateX: pan.x }],
                            }}
                            {...panResponder.panHandlers}
                        >
                            <View style={styles.sliderKnob}>
                                <Ionicons name="call" size={32} color="#22c55e" />
                            </View>
                        </Animated.View>
                    </View>
                </View>
            </View>
        );
    }

    // ANALYZING STATE (iPhone Call Style V2)
    if (state === 'analyzing') {
        const topButtons = [
            { iconName: 'mic-off', label: '소리 끔' },
            { iconName: 'keypad', label: '키패드' },
            { iconName: 'volume-high', label: '스피커' },
        ];
        const bottomButtons = [
            { iconName: 'shield-checkmark', label: 'Callcops', onPress: () => setShowSecurity(true), highlight: true },
            { iconName: 'videocam', label: 'FaceTime' },
            { iconName: 'person-circle', label: '연락처' },
        ];

        return (
            <View style={styles.iphoneContainer}>
                {typeof LinearGradient !== 'undefined' ? (
                    <LinearGradient
                        colors={['#7c2d12', '#831843', '#2e1065']} // Dark Orange -> Dark Pink -> Dark Purple
                        style={StyleSheet.absoluteFill}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                    />
                ) : (
                    <View style={[StyleSheet.absoluteFill, { backgroundColor: '#111b27' }]} />
                )}

                {/* Header Info */}
                <View style={styles.iphoneHeaderV2}>
                    <Text style={styles.iphoneCallerNameV2}>Jane님 및 Armando</Text>
                    <View style={[styles.persistentBadge, realtimeValid ? styles.badgeVerified : styles.badgeSpoofed]}>
                        <Text style={styles.persistentBadgeText}>
                            {realtimeValid ? 'AUTHENTICATED' : 'SPOOFING SUSPECTED'}
                        </Text>
                    </View>
                    <Text style={styles.iphoneTimerV2}>{formatDuration(callDuration)}</Text>
                </View>

                {/* Main Call UI Area */}
                <View style={styles.iphoneContentArea}>
                    {showSecurity ? (
                        /* AI Monitoring Overlay - Visible when toggled */
                        /* AI Monitoring Overlay - Evolved Tab Design */
                        <View style={styles.aiOverlayV2}>
                            <LinearGradient
                                colors={['rgba(71, 85, 105, 0.95)', 'rgba(30, 41, 59, 0.98)']} // Slate-600 -> Slate-800 (Lighter)
                                style={StyleSheet.absoluteFill}
                            />

                            <View style={styles.aiHeader}>

                                <TouchableOpacity
                                    style={styles.closeOverlayButton}
                                    onPress={() => setShowSecurity(false)}
                                >
                                    <Ionicons name="close" size={20} color="#9ca3af" />
                                </TouchableOpacity>
                            </View>

                            <ScrollView
                                style={styles.aiScrollArea}
                                contentContainerStyle={styles.aiScrollContent}
                                showsVerticalScrollIndicator={false}
                            >
                                <View style={styles.aiSection}>
                                    {renderWaveform()}
                                </View>

                                <View style={styles.miniMatrixSection}>
                                    <Text style={styles.miniLabel}>REAL-TIME SIGNAL</Text>
                                    {renderMiniBitMatrix(true)}
                                </View>

                                <View style={styles.miniMatrixSection}>
                                    <Text style={styles.miniLabel}>ACCUMULATED ANALYSIS</Text>
                                    {renderMiniBitMatrix(false)}
                                </View>
                            </ScrollView>
                        </View>
                    ) : (
                        /* Standard Icon Grid (3x2) */
                        <View style={styles.iphoneGridV2}>
                            <View style={styles.iphoneGridRow}>
                                {topButtons.map((btn, i) => (
                                    <View key={i} style={styles.iphoneGridItemV2}>
                                        <View style={[styles.iphoneCircleButtonV2, i === 2 && styles.activeSpeaker]}>
                                            <Ionicons name={btn.iconName} size={32} color={i === 2 ? '#000' : '#fff'} />
                                        </View>
                                        <Text style={styles.iphoneButtonLabelV2}>{btn.label}</Text>
                                    </View>
                                ))}
                            </View>
                            <View style={styles.iphoneGridRow}>
                                {bottomButtons.map((btn, i) => (
                                    <TouchableOpacity
                                        key={i}
                                        style={styles.iphoneGridItemV2}
                                        onPress={btn.onPress}
                                        disabled={!btn.onPress}
                                        activeOpacity={0.7}
                                    >
                                        <View style={[styles.iphoneCircleButtonV2, { opacity: btn.highlight ? 1.0 : 0.5 }]}>
                                            <Ionicons name={btn.iconName} size={32} color="#fff" />
                                        </View>
                                        <Text style={styles.iphoneButtonLabelV2}>{btn.label}</Text>
                                    </TouchableOpacity>
                                ))}
                            </View>

                        </View>
                    )}
                </View>

                {/* Bottom Control Area */}
                {!showSecurity && (
                    <View style={styles.iphoneFooterV2}>
                        <TouchableOpacity
                            style={styles.endCallButtonV2}
                            onPress={handleReset}
                        >
                            <Ionicons name="call" size={36} color="#ffffff" />
                        </TouchableOpacity>
                    </View>
                )}
            </View>
        );
    }

    // RESULT STATE
    if (state === 'result') {
        return (
            <View style={{ flex: 1 }}>
                <LinearGradient
                    colors={['#7c2d12', '#831843', '#2e1065']} // Dark Orange -> Dark Pink -> Dark Purple
                    style={StyleSheet.absoluteFill}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                />
                <View style={[styles.container, { backgroundColor: 'transparent' }]}>
                    <ScrollView
                        style={styles.scrollView}
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                    >
                        <View style={styles.centerContent}>
                            <View style={[styles.resultCircle, isValid ? styles.validCircle : styles.invalidCircle]}>
                                <Text style={styles.resultIcon}>{isValid ? '✓' : '✗'}</Text>
                            </View>

                            <Text style={[styles.resultTitle, isValid ? styles.validText : styles.invalidText]}>
                                {isValid ? 'Verified Caller' : 'Potential Spoofing'}
                            </Text>

                            <Text style={styles.subtitle}>
                                {isValid ? '인증된 발신자입니다' : '발신자 인증에 실패했습니다'}
                            </Text>

                            {/* Method Badge */}

                            {/* Main Status Summary */}
                            <View style={[styles.summaryBadge, isValid ? styles.validBadge : styles.invalidBadge]}>
                                <Text style={styles.summaryText}>
                                    {isValid
                                        ? '✓ CRC 일치 · 인증 완료'
                                        : '✗ CRC 불일치 · 인증 실패'}
                                </Text>
                            </View>

                            {/* Expandable Details Section */}
                            <TouchableOpacity
                                style={styles.detailsToggle}
                                onPress={() => setShowDetails(!showDetails)}
                            >
                                <Text style={styles.detailsToggleText}>
                                    {showDetails ? '▲ 상세정보 닫기' : '▼ 상세정보 보기'}
                                </Text>
                            </TouchableOpacity>

                            {showDetails && (
                                <View style={styles.detailsContainer}>
                                    {/* CRC Details */}
                                    <View style={styles.detailRow}>
                                        <Text style={styles.detailLabel}>Expected CRC:</Text>
                                        <Text style={styles.detailValue}>
                                            0x{crcInfo.expected?.toString(16).toUpperCase().padStart(4, '0') || '----'}
                                        </Text>
                                    </View>
                                    <View style={styles.detailRow}>
                                        <Text style={styles.detailLabel}>Actual CRC:</Text>
                                        <Text style={styles.detailValue}>
                                            0x{crcInfo.actual?.toString(16).toUpperCase().padStart(4, '0') || '----'}
                                        </Text>
                                    </View>
                                    <View style={styles.detailRow}>
                                        <Text style={styles.detailLabel}>CRC 상태:</Text>
                                        <Text style={[styles.detailValue, crcValid ? styles.successText : styles.failText]}>
                                            {crcValid ? '일치' : '불일치'}
                                        </Text>
                                    </View>
                                    <View style={styles.detailDivider} />
                                    <View style={styles.detailRow}>
                                        <Text style={styles.detailLabel}>Confidence:</Text>
                                        <Text style={styles.detailValue}>{(confidence * 100).toFixed(1)}%</Text>
                                    </View>
                                    <View style={styles.detailRow}>
                                        <Text style={styles.detailLabel}>Detection:</Text>
                                        <Text style={[styles.detailValue, confidence > 0.15 ? styles.successText : styles.failText]}>
                                            {confidence > 0.15 ? '워터마크 감지' : '감지 실패'}
                                        </Text>
                                    </View>
                                </View>
                            )}


                            {renderBitMatrix(false, true)}

                            {errorMessage && (
                                <View style={styles.errorBox}>
                                    <Text style={styles.errorText}>{errorMessage}</Text>
                                </View>
                            )}

                            {/* Exit Button (나가기) */}
                            <TouchableOpacity
                                style={styles.exitButton}
                                onPress={onBack}
                            >
                                <Text style={styles.exitButtonText}>나가기</Text>
                            </TouchableOpacity>
                        </View>
                    </ScrollView>
                </View>
            </View>
        );
    }

    return null;
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#111827',
        paddingHorizontal: 24,
        paddingVertical: 48,
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    validContainer: {
        backgroundColor: '#0c1a0f',
    },
    invalidContainer: {
        backgroundColor: '#1a0c0c',
    },
    backButton: {
        alignSelf: 'flex-start',
    },
    backText: {
        color: '#9ca3af',
        fontSize: 16,
    },
    centerContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    callerCircle: {
        width: 120,
        height: 120,
        borderRadius: 60,
        backgroundColor: '#374151',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    callerIcon: {
        fontSize: 48,
    },
    callerName: {
        fontSize: 34,
        fontWeight: '400',
        color: '#fff',
        marginBottom: 8,
    },
    callerNumber: {
        fontSize: 18,
        color: '#e5e5e5', // Brighter gray
        fontWeight: '400',
        marginBottom: 24,
    },
    incomingText: {
        fontSize: 14,
        color: '#6b7280',
    },
    warningBox: {
        backgroundColor: 'rgba(234, 179, 8, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(234, 179, 8, 0.3)',
        borderRadius: 12,
        padding: 12,
        marginTop: 16,
    },
    warningText: {
        color: '#fbbf24',
        fontSize: 12,
        textAlign: 'center',
    },
    errorBox: {
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(239, 68, 68, 0.3)',
        borderRadius: 12,
        padding: 12,
        marginTop: 16,
        maxWidth: '80%',
    },
    errorText: {
        color: '#f87171',
        fontSize: 12,
        textAlign: 'center',
    },
    callButtons: {
        flexDirection: 'row',
        gap: 80,
    },
    buttonLabels: {
        flexDirection: 'row',
        gap: 60,
        marginTop: 12,
    },
    declineButton: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: '#ef4444',
        alignItems: 'center',
        justifyContent: 'center',
    },
    answerButton: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: '#22c55e',
        alignItems: 'center',
        justifyContent: 'center',
    },
    buttonIcon: {
        fontSize: 24,
        color: '#fff',
    },
    declineLabel: {
        color: '#ef4444',
        fontSize: 14,
    },
    answerLabel: {
        color: '#22c55e',
        fontSize: 14,
    },
    analyzeCircle: {
        width: 128,
        height: 128,
        borderRadius: 64,
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 4,
        borderColor: 'rgba(59, 130, 246, 0.5)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    analyzeIcon: {
        fontSize: 48,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#fff',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 14,
        color: '#9ca3af',
        textAlign: 'center',
        marginBottom: 24,
    },
    progressContainer: {
        width: '80%',
        height: 8,
        backgroundColor: '#374151',
        borderRadius: 4,
        overflow: 'hidden',
    },
    progressBar: {
        height: '100%',
        backgroundColor: '#3b82f6',
    },
    progressText: {
        color: '#9ca3af',
        fontSize: 14,
        marginTop: 8,
        marginBottom: 16,
    },
    fileName: {
        color: '#6b7280',
        fontSize: 12,
        fontFamily: 'monospace',
    },
    bitMatrix: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        width: 196,
        gap: 2,
        padding: 12,
        backgroundColor: 'rgba(55, 65, 81, 0.3)',
        borderRadius: 12,
        marginTop: 24,
    },
    bit: {
        width: 10,
        height: 10,
        borderRadius: 2,
        backgroundColor: '#374151', // Fixed: Remove dynamic background here to prevent undefined error if not set? No, prob is passed.
    },
    resultCircle: {
        width: 96,
        height: 96,
        borderRadius: 48,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 4,
        marginBottom: 24,
    },
    validCircle: {
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderColor: '#22c55e',
    },
    invalidCircle: {
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderColor: '#ef4444',
    },
    resultIcon: {
        fontSize: 36,
        color: '#fff',
    },
    resultTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 8,
    },
    validText: {
        color: '#22c55e',
    },
    invalidText: {
        color: '#ef4444',
    },
    methodBadge: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        marginBottom: 16,
    },
    onnxBadge: {
        backgroundColor: 'rgba(139, 92, 246, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(139, 92, 246, 0.3)',
    },
    fallbackBadge: {
        backgroundColor: 'rgba(234, 179, 8, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(234, 179, 8, 0.3)',
    },
    methodText: {
        color: '#fff',
        fontSize: 14,
    },
    statusContainer: {
        gap: 8,
        marginTop: 8,
    },
    statusBadge: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 8,
        width: 240,
    },
    validBadge: {
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 1,
        borderColor: '#22c55e',
    },
    invalidBadge: {
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
    },
    statusLabel: {
        color: '#9ca3af',
        fontSize: 14,
    },
    statusValue: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
    endCallButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#ef4444',
        paddingHorizontal: 32,
        paddingVertical: 16,
        borderRadius: 32,
        gap: 12,
    },
    endCallIcon: {
        fontSize: 20,
    },
    endCallText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    // Dual Grid Styles
    dualMatrixContainer: {
        marginTop: 24,
        alignItems: 'center',
    },
    gridWrapper: {
        alignItems: 'center',
    },
    gridLabel: {
        color: '#9ca3af',
        fontSize: 12,
        marginBottom: 8,
        fontWeight: '600',
    },
    gridContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        width: 224, // 16 cols * (12px + 2px gap)
        gap: 2,
    },
    gridBit: {
        width: 12,
        height: 12,
        borderRadius: 2,
    },
    spacer: {
        height: 24,
    },
    rtBadge: {
        marginTop: 16,
        paddingHorizontal: 16,
        paddingVertical: 6,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.1)',
    },
    rtText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: 'bold',
    },
    // Summary Badge
    summaryBadge: {
        marginTop: 16,
        paddingHorizontal: 24,
        paddingVertical: 12,
        borderRadius: 12,
    },
    summaryText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
        textAlign: 'center',
    },
    // Expandable Details
    detailsToggle: {
        marginTop: 16,
        paddingVertical: 8,
    },
    detailsToggleText: {
        color: '#6b7280',
        fontSize: 14,
    },
    detailsContainer: {
        marginTop: 8,
        backgroundColor: 'rgba(255,255,255,0.05)',
        borderRadius: 12,
        padding: 16,
        width: '100%',
    },
    detailRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 6,
    },
    detailLabel: {
        color: '#9ca3af',
        fontSize: 13,
    },
    detailValue: {
        color: '#e5e7eb',
        fontSize: 13,
        fontFamily: 'monospace',
    },
    detailDivider: {
        height: 1,
        backgroundColor: 'rgba(255,255,255,0.1)',
        marginVertical: 8,
    },
    successText: {
        color: '#22c55e',
    },
    failText: {
        color: '#ef4444',
    },
    // ScrollView Styles
    scrollView: {
        width: '100%',
        flex: 1,
    },
    scrollContent: {
        alignItems: 'center',
        paddingBottom: 24,
        paddingTop: 80, // Increased to move content down further
    },
    // Waveform Styles
    // Waveform Styles
    liveContainer: {
        flexDirection: 'row',
        height: '100%', // Match parent height
        width: '100%',
        backgroundColor: '#111827',
        borderRadius: 8,
        padding: 8,
        borderWidth: 1,
        borderColor: '#374151',
    },
    eqContainer: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center', // Center vertical
        justifyContent: 'space-between',
        paddingRight: 10,
        gap: 0.5, // Even smaller gap
    },
    eqBar: {
        flex: 1,
        backgroundColor: '#00ffff',
        borderRadius: 0.5,
        minWidth: 1, // Ultra-thin bars
    },
    monitorContainer: {
        width: 100,
        justifyContent: 'center',
        alignItems: 'center',
        borderLeftWidth: 1,
        borderLeftColor: '#374151',
        paddingLeft: 10,
    },
    monitorLabel: {
        color: '#6b7280',
        fontSize: 10,
        fontWeight: 'bold',
        marginBottom: 2,
    },
    bitIndex: {
        color: '#ffffff',
        fontSize: 14,
        fontFamily: 'monospace',
        marginBottom: 4,
    },
    decisionBox: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#1f2937',
        borderRadius: 4,
        width: '100%',
        paddingVertical: 4,
    },
    bitValue: {
        fontSize: 24,
        fontWeight: 'bold',
        lineHeight: 28,
    },
    confidenceMini: {
        color: '#9ca3af',
        fontSize: 8,
    },
    // iPhone Style V2 Redesign
    iphoneContainer: {
        flex: 1,
        backgroundColor: '#111b27', // iOS dark blue-ish
        width: '100%',
        alignItems: 'center',
    },
    iphoneHeaderV2: {
        alignItems: 'center',
        marginTop: 120, // Increased to move header down
        marginBottom: 80, // Increased to push content down
        width: '100%',
    },
    persistentBadge: {
        paddingHorizontal: 12,
        paddingVertical: 4,
        borderRadius: 20,
        marginBottom: 10,
        borderWidth: 1,
    },
    badgeVerified: {
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
    },
    badgeSpoofed: {
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
    },
    persistentBadgeText: {
        color: '#fff',
        fontSize: 10,
        fontWeight: 'bold',
        letterSpacing: 1,
    },
    iphoneCallerNameV2: {
        color: '#fff',
        fontSize: 32,
        fontWeight: '300',
        marginBottom: 8,
    },
    iphoneTimerV2: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '300',
        opacity: 0.8,
    },
    iphoneContentArea: {
        flex: 1,
        width: '100%',
        justifyContent: 'flex-start',
        alignItems: 'center',
    },
    iphoneGridV2: {
        width: '100%',
        paddingHorizontal: 10,
        marginTop: 50, // Increased to push grid down
    },
    iphoneGridRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        paddingHorizontal: 40,
        marginBottom: 20,
    },
    iphoneGridItemV2: {
        alignItems: 'center',
        width: 80,
    },
    iphoneCircleButtonV2: {
        width: 75,
        height: 75,
        borderRadius: 37.5,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 8,
    },
    activeSpeaker: {
        backgroundColor: '#ffffff',
    },
    iphoneButtonIconV2: {
        fontSize: 32,
    },
    iphoneButtonLabelV2: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '400',
        marginTop: 4,
    },
    callcopsButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 24,
        paddingVertical: 14,
        borderRadius: 30,
        borderWidth: 1,
        borderColor: 'rgba(34, 211, 238, 0.4)', // Cyan-400
        width: 240,
        gap: 8,
    },
    callcopsButtonActive: {
        backgroundColor: '#00ffff',
        borderColor: '#00ffff',
    },
    callcopsButtonText: {
        color: '#22d3ee', // Cyan-400
        fontSize: 15,
        fontWeight: '600',
        letterSpacing: 0.5,
    },
    callcopsButtonIcon: {
        fontSize: 16,
    },
    callcopsButtonTextActive: {
        color: '#000',
    },
    aiOverlayV2: {
        width: '92%',
        height: '85%', // Increased Height
        borderRadius: 32,
        overflow: 'hidden',
        alignSelf: 'center',
        marginTop: 10, // Moved up slightly
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.2)', // Lighter border
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.5,
        shadowRadius: 20,
        elevation: 10,
        padding: 20, // Added padding as requested
    },
    aiHeader: {
        width: '100%',
        height: 50, // Spacer for the absolute button
    },
    closeOverlayButton: {
        position: 'absolute',
        top: 20,
        right: 20,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
    },
    aiScrollArea: {
        flex: 1,
        width: '100%',
    },
    aiScrollContent: {
        paddingBottom: 20,
    },
    aiSection: {
        height: 100, // Reduced height as requested
        marginBottom: 0, // Minimized gap
        justifyContent: 'center',
    },
    aiHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 15,
        paddingHorizontal: 5,
    },
    closeOverlayButton: {
        width: 30,
        height: 30,
        borderRadius: 15,
        backgroundColor: 'rgba(255,255,255,0.1)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    closeOverlayText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '200',
    },
    exitButton: {
        marginTop: 40,
        backgroundColor: '#ef4444',
        paddingHorizontal: 40,
        paddingVertical: 12,
        borderRadius: 25,
        width: 200,
        alignItems: 'center',
    },
    exitButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
    },
    // New Slide to Answer Styles
    backButtonTop: {
        position: 'absolute',
        top: 60,
        left: 24,
        zIndex: 10,
    },
    backTextTop: {
        color: '#fff',
        fontSize: 18,
    },
    bottomControlArea: {
        width: '100%',
        alignItems: 'center',
        paddingBottom: 80, // Moved up
    },
    actionButtonsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '70%', // Narrower width (was 80%)
        marginBottom: 50, // Increased to move buttons up
        marginTop: 20,
    },
    actionColumn: {
        alignItems: 'center',
    },
    iconCircle: {
        width: 24,
        height: 24,
        marginBottom: 8,
        alignItems: 'center',
    },
    smallIcon: {
        fontSize: 24,
        color: '#fff',
    },
    iconLabel: {
        color: '#fff',
        fontSize: 12,
    },
    sliderContainer: {
        width: '85%',
        height: 80, // slightly taller container
        justifyContent: 'center',
    },
    sliderTrack: {
        position: 'absolute',
        width: '100%',
        height: 76,
        borderRadius: 38, // Fully rounded
        backgroundColor: 'rgba(255, 255, 255, 0.25)', // Semi-transparent white like iOS
        justifyContent: 'center',
        alignItems: 'center',
    },
    shimmerText: {
        color: '#fff',
        fontSize: 17, // Increased size
        fontWeight: '500',
        letterSpacing: -0.5,
        opacity: 0.8,
    },
    sliderKnob: {
        width: 76,
        height: 76,
        borderRadius: 38,
        backgroundColor: '#fff',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: "#000",
        shadowOffset: {
            width: 0,
            height: 2,
        },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
        elevation: 5,
    },
    phoneIcon: {
        fontSize: 32,
        color: '#22c55e', // Green phone icon inside
    },
    statusIndicator: {
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 4,
        borderWidth: 1,
    },
    statusGreen: {
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
    },
    statusAmber: {
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
    },
    statusText: {
        fontSize: 9,
        fontWeight: 'bold',
        color: '#fff',
    },
    aiContent: {
        width: '100%',
    },
    miniMatrixSection: {
        marginTop: 10,
        paddingTop: 10,
        borderTopWidth: 1,
        borderTopColor: 'rgba(255, 255, 255, 0.05)',
    },
    miniLabel: {
        color: 'rgba(255, 255, 255, 0.4)',
        fontSize: 9,
        fontWeight: 'bold',
        marginBottom: 10,
        letterSpacing: 0.5,
    },
    miniMatrixGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        width: '100%',
        aspectRatio: 2, // 16x8 feel
    },
    miniGridBit: {
        width: '6.25%', // 100/16
        height: '12.5%', // 100/8
    },
    miniMatrixContainerV2: {
        marginTop: 20,
        height: 150,
    },
    iphoneFooterV2: {
        position: 'absolute',
        bottom: 60, // Adjusted for safe area
        width: '100%',
        alignItems: 'center',
    },
    endCallButtonV2: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#ff3b30', // Reverted to Red
        justifyContent: 'center',
        alignItems: 'center',
        transform: [{ rotate: '135deg' }],
    },
    endCallIconV2: {
        fontSize: 38,
        color: '#fff',
    },
});
