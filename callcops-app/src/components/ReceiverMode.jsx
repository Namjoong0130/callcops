/**
 * ReceiverMode - File upload and real ONNX watermark detection
 * Uses onnxruntime for decoding with simple fallback
 */
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { useInference } from '../hooks/useInference';
import { verifyCRC } from '../utils/crc';

const SAMPLE_RATE = 8000;
const FRAME_SIZE = 320;
const PAYLOAD_LENGTH = 128;

export default function ReceiverMode({ onBack }) {
    const [state, setState] = useState('idle');
    const [fileName, setFileName] = useState(null);
    const [progress, setProgress] = useState(0);
    const [isValid, setIsValid] = useState(null);
    const [crcValid, setCrcValid] = useState(null);
    const [confidence, setConfidence] = useState(0);
    const [bitProbs, setBitProbs] = useState(null);
    const [errorMessage, setErrorMessage] = useState(null);
    const [usedOnnx, setUsedOnnx] = useState(false);

    const inference = useInference();

    // Read WAV file and get audio samples
    const readWavFile = async (uri) => {
        try {
            const base64 = await FileSystem.readAsStringAsync(uri, {
                encoding: 'base64',
            });

            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }

            const view = new DataView(bytes.buffer);

            // Parse WAV header
            const sampleRate = view.getUint32(24, true);
            const bitsPerSample = view.getUint16(34, true);
            const bytesPerSample = bitsPerSample / 8;

            // Find data chunk
            let dataOffset = 12;
            while (dataOffset < bytes.length - 8) {
                const chunkId = String.fromCharCode(bytes[dataOffset], bytes[dataOffset + 1], bytes[dataOffset + 2], bytes[dataOffset + 3]);
                const chunkSize = view.getUint32(dataOffset + 4, true);
                if (chunkId === 'data') {
                    dataOffset += 8;
                    break;
                }
                dataOffset += 8 + chunkSize;
            }

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
    const detectWatermarkSimple = (samples) => {
        const numFrames = Math.floor(samples.length / FRAME_SIZE);

        const bitAccumulators = new Float32Array(PAYLOAD_LENGTH).fill(0);
        const bitCounts = new Float32Array(PAYLOAD_LENGTH).fill(0);

        for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            const start = frameIdx * FRAME_SIZE;
            const bitIdx = frameIdx % PAYLOAD_LENGTH;

            let correlation = 0;
            for (let i = 0; i < FRAME_SIZE; i++) {
                const expected = Math.sin(2 * Math.PI * (start + i) / 10);
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

        try {
            setProgress(10);

            // Read audio file
            const { samples, sampleRate } = await readWavFile(fileUri);
            console.log('Audio loaded:', samples.length, 'samples at', sampleRate, 'Hz');
            setProgress(30);

            // Resample to 8kHz
            const audio8k = inference.resampleTo8kHz(samples, sampleRate);
            console.log('Resampled to:', audio8k.length, 'samples');
            setProgress(40);

            let result;

            // Try ONNX decoder first
            if (inference.onnxAvailable) {
                try {
                    console.log('Attempting ONNX decoding...');
                    await inference.loadDecoder();
                    setProgress(60);

                    result = await inference.runDecoder(audio8k);
                    setUsedOnnx(true);
                    console.log('ONNX decoding successful');
                } catch (onnxErr) {
                    console.warn('ONNX decoding failed, using fallback:', onnxErr);
                    result = detectWatermarkSimple(audio8k);
                }
            } else {
                console.log('ONNX not available, using fallback');
                result = detectWatermarkSimple(audio8k);
            }

            setProgress(80);

            // Set bit probabilities for display
            setBitProbs(Array.from(result.bits128));
            setConfidence(result.confidence);

            // Verify CRC
            const crcResult = verifyCRC(result.bits128);
            console.log('CRC verification:', crcResult);
            setProgress(90);

            // Determine validity
            const hasWatermark = result.confidence > 0.15;
            const crcPassed = crcResult.isValid;

            setCrcValid(crcPassed);
            setIsValid(hasWatermark && crcPassed);

            setProgress(100);
            setState('result');

        } catch (err) {
            console.error('Analysis error:', err);
            setErrorMessage(`분석 오류: ${err.message}`);
            setIsValid(false);
            setCrcValid(false);
            setState('result');
        }
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

    // Render bit matrix
    const renderBitMatrix = () => (
        <View style={styles.bitMatrix}>
            {(bitProbs || Array(128).fill(0.5)).map((prob, i) => (
                <View
                    key={i}
                    style={[
                        styles.bit,
                        { backgroundColor: prob > 0.5 ? '#22c55e' : '#ef4444' }
                    ]}
                />
            ))}
        </View>
    );

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

                    {/* Status Badges */}
                    <View style={styles.statusContainer}>
                        <View style={[styles.statusBadge, crcValid ? styles.validBadge : styles.invalidBadge]}>
                            <Text style={styles.statusLabel}>CRC-16</Text>
                            <Text style={styles.statusValue}>{crcValid ? '✓ Valid' : '✗ Mismatch'}</Text>
                        </View>
                        <View style={[styles.statusBadge, confidence > 0.15 ? styles.validBadge : styles.invalidBadge]}>
                            <Text style={styles.statusLabel}>Confidence</Text>
                            <Text style={styles.statusValue}>{(confidence * 100).toFixed(1)}%</Text>
                        </View>
                    </View>

                    {renderBitMatrix()}

                    {errorMessage && (
                        <View style={styles.errorBox}>
                            <Text style={styles.errorText}>{errorMessage}</Text>
                        </View>
                    )}
                </View>

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
});
