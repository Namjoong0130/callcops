/**
 * ReceiverMode - File upload and real ONNX watermark detection
 * Uses onnxruntime for decoding with simple fallback
 */
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ScrollView } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { useInference } from '../hooks/useInference';
import { verifyCRC, attemptCorrection } from '../utils/crc';

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

    // Cleanup timer on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    const inference = useInference();

    // Read WAV file and get audio samples
    const readWavFile = async (uri) => {
        try {
            // Optimization: Read only first 2MB (approx 10-20 seconds depending on quality)
            // This prevents blocking the UI for long files
            const MAX_READ_SIZE = 2 * 1024 * 1024;

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

            // Reset accumulators and display refs
            accumulatorsRef.current.fill(0);
            countsRef.current.fill(0);
            latestProbsRef.current.fill(-1);  // -1 = not yet detected (will show dim)
            displayedBitProbsRef.current.fill(-1);
            setCurrentChunkProbs(null);  // Clear instant grid
            setBitProbs(null);  // Clear accumulated grid

            // Load Model
            if (inference.onnxAvailable) {
                await inference.loadDecoder();
            }

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

    const handleReset = () => {
        setState('idle');
        setFileName(null);
        setProgress(0);
        setIsValid(null);
        setCrcValid(null);
        setConfidence(0);
        setBitProbs(null);
        setErrorMessage(null);
        setUsedOnnx(false);
    };

    // Render Dual Bit Grids
    const renderBitMatrix = () => {
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

                            return (
                                <View
                                    key={i}
                                    style={[
                                        styles.gridBit,
                                        {
                                            backgroundColor: bgColor,
                                            borderColor: isActive ? '#00ffff' : '#374151',
                                            borderWidth: isActive ? 2 : 0.5,
                                            zIndex: isActive ? 10 : 1
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
                {renderGrid(instantData, true)}

                <View style={styles.spacer} />

                {renderGrid(accumulatedData, false)}

                <View style={[styles.rtBadge, realtimeValid ? styles.validBadge : styles.invalidBadge]}>
                    <Text style={styles.rtText}>
                        {realtimeValid ? '✓ 인증된 사용자' : '⚠️ 스푸핑 의심'}
                    </Text>
                </View>
            </View>
        );
    };

    // IDLE STATE
    if (state === 'idle') {
        return (
            <View style={styles.container}>
                <TouchableOpacity onPress={onBack} style={styles.backButton}>
                    <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>

                <View style={styles.centerContent}>
                    <View style={styles.callerCircle}>
                        <Text style={styles.callerIcon}>📱</Text>
                    </View>
                    <Text style={styles.callerName}>Unknown Caller</Text>
                    <Text style={styles.callerNumber}>+82 10-****-****</Text>
                    <Text style={styles.incomingText}>Incoming Call...</Text>

                    {!inference.onnxAvailable && (
                        <View style={styles.warningBox}>
                            <Text style={styles.warningText}>⚠️ ONNX 사용 불가 - Fallback 모드</Text>
                        </View>
                    )}

                    {errorMessage && (
                        <View style={styles.errorBox}>
                            <Text style={styles.errorText}>{errorMessage}</Text>
                        </View>
                    )}
                </View>

                <View style={styles.callButtons}>
                    <TouchableOpacity onPress={onBack} style={styles.declineButton}>
                        <Text style={styles.buttonIcon}>✕</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={handlePickFile} style={styles.answerButton}>
                        <Text style={styles.buttonIcon}>✓</Text>
                    </TouchableOpacity>
                </View>
                <View style={styles.buttonLabels}>
                    <Text style={styles.declineLabel}>Decline</Text>
                    <Text style={styles.answerLabel}>Answer</Text>
                </View>
            </View>
        );
    }

    // ANALYZING STATE
    if (state === 'analyzing') {
        return (
            <View style={styles.container}>
                <View style={styles.centerContent}>
                    <View style={styles.analyzeCircle}>
                        <Text style={styles.analyzeIcon}>🎵</Text>
                    </View>
                    <Text style={styles.title}>Analyzing Call...</Text>
                    <Text style={styles.subtitle}>
                        {inference.onnxAvailable ? 'ONNX 모델로 분석 중...' : 'Fallback 모드로 분석 중...'}
                    </Text>

                    <View style={styles.progressContainer}>
                        <View style={[styles.progressBar, { width: `${progress}%` }]} />
                    </View>
                    <Text style={styles.progressText}>{progress}%</Text>

                    {fileName && (
                        <Text style={styles.fileName}>{fileName}</Text>
                    )}

                    {renderBitMatrix()}
                </View>
            </View>
        );
    }

    // RESULT STATE
    if (state === 'result') {
        return (
            <View style={[styles.container, isValid ? styles.validContainer : styles.invalidContainer]}>
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
                        <View style={[styles.methodBadge, usedOnnx ? styles.onnxBadge : styles.fallbackBadge]}>
                            <Text style={styles.methodText}>
                                {usedOnnx ? '🧠 ONNX Model' : '📊 Fallback'}
                            </Text>
                        </View>

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
                                        {crcValid ? '✓ 일치' : '✗ 불일치'}
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
                                        {confidence > 0.15 ? '✓ 워터마크 감지' : '✗ 감지 실패'}
                                    </Text>
                                </View>
                            </View>
                        )}

                        {renderBitMatrix()}

                        {errorMessage && (
                            <View style={styles.errorBox}>
                                <Text style={styles.errorText}>{errorMessage}</Text>
                            </View>
                        )}
                    </View>
                </ScrollView>

                <TouchableOpacity onPress={handleReset} style={styles.endCallButton}>
                    <Text style={styles.endCallIcon}>📞</Text>
                    <Text style={styles.endCallText}>End Call</Text>
                </TouchableOpacity>
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
        fontSize: 28,
        fontWeight: 'bold',
        color: '#fff',
        marginBottom: 8,
    },
    callerNumber: {
        fontSize: 16,
        color: '#9ca3af',
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
    },
});
