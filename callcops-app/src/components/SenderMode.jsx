/**
 * SenderMode - Record audio and encode with ONNX watermark
 * Uses react-native-live-audio-stream to get REAL PCM data
 */
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, Platform, PermissionsAndroid } from 'react-native';
import LiveAudioStream from 'react-native-live-audio-stream';
import { Audio } from 'expo-av'; // For permission only
import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';
import { useInference } from '../hooks/useInference';
import { calculateCRC16 } from '../utils/crc';
import { Buffer } from 'buffer';

// ONNX model settings
const TARGET_SAMPLE_RATE = 16000;
const DOWNSAMPLE_RATE = 8000;

export default function SenderMode({ onBack }) {
    const [state, setState] = useState('idle');
    const [recordingTime, setRecordingTime] = useState(0);
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState('');
    const [errorMessage, setErrorMessage] = useState(null);
    const [bitProbs, setBitProbs] = useState(null);
    const [encodedUri, setEncodedUri] = useState(null);
    const [permissionGranted, setPermissionGranted] = useState(false);
    const [usedOnnx, setUsedOnnx] = useState(false);

    const inference = useInference();
    const timerRef = useRef(null);

    // Store raw PCM samples
    const pcmDataRef = useRef([]);

    // Configure LiveAudioStream
    useEffect(() => {
        const options = {
            sampleRate: TARGET_SAMPLE_RATE,
            channels: 1,
            bitsPerSample: 16,
            audioSource: 6, // Voice Recognition
            bufferSize: 4096,
        };

        LiveAudioStream.init(options);

        LiveAudioStream.on('data', data => {
            // 'data' is a base64 encoded string of PCM samples
            const chunk = Buffer.from(data, 'base64');
            const samples = new Int16Array(
                chunk.buffer,
                chunk.byteOffset,
                chunk.byteLength / 2
            );

            // Convert to float [-1, 1] and append
            const floatSamples = new Float32Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
                floatSamples[i] = samples[i] / 32768.0;
            }

            // Push individual samples to array (simple but works for demo length)
            // For production, use Float32Array concatenation helper
            for (let i = 0; i < floatSamples.length; i++) {
                pcmDataRef.current.push(floatSamples[i]);
            }
        });

        return () => {
            LiveAudioStream.stop();
        };
    }, []);

    // Request permissions
    useEffect(() => {
        (async () => {
            if (Platform.OS === 'android') {
                const granted = await PermissionsAndroid.request(
                    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO
                );
                setPermissionGranted(granted === PermissionsAndroid.RESULTS.GRANTED);
            } else {
                const { status } = await Audio.requestPermissionsAsync();
                setPermissionGranted(status === 'granted');
            }
        })();
    }, []);

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const generateMessage = () => {
        const now = Math.floor(Date.now() / 1000);
        const syncPattern = 0xAAAA;
        const timestamp = now & 0xFFFFFFFF;
        const authCode = BigInt(Math.floor(Math.random() * 0xFFFFFFFFFFFFFFFF));
        const messageBits = [];

        for (let i = 15; i >= 0; i--) messageBits.push((syncPattern >> i) & 1);
        for (let i = 31; i >= 0; i--) messageBits.push((timestamp >> i) & 1);
        for (let i = 63; i >= 0; i--) messageBits.push(Number((authCode >> BigInt(i)) & BigInt(1)));

        const crc = calculateCRC16(messageBits.slice(0, 112));
        for (let i = 15; i >= 0; i--) messageBits.push((crc >> i) & 1);

        return messageBits;
    };

    const resampleTo8kHz = (samples) => {
        const ratio = Math.floor(TARGET_SAMPLE_RATE / DOWNSAMPLE_RATE);
        const result = new Float32Array(Math.floor(samples.length / ratio));
        for (let i = 0; i < result.length; i++) {
            result[i] = samples[i * ratio];
        }
        return result;
    };

    const embedWatermarkSimple = (samples, messageBits) => {
        const output = new Float32Array(samples.length);
        const watermarkStrength = 0.02;
        const FRAME_SIZE = 320;

        for (let i = 0; i < samples.length; i++) {
            const frameIdx = Math.floor(i / FRAME_SIZE);
            const bitIdx = frameIdx % 128;
            const bit = messageBits[bitIdx];
            const watermark = bit ? watermarkStrength : -watermarkStrength;
            output[i] = samples[i] + watermark * Math.sin(2 * Math.PI * i / 10);
        }
        return output;
    };

    const createWavFile = async (samples) => {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = DOWNSAMPLE_RATE * numChannels * 2;
        const blockAlign = numChannels * 2;
        const dataSize = samples.length * 2;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, DOWNSAMPLE_RATE, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        const bytes = new Uint8Array(buffer);
        let binary = '';
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }

        const base64 = global.btoa ? global.btoa(binary) : Buffer.from(binary, 'binary').toString('base64');
        const fileUri = FileSystem.documentDirectory + `callcops_encoded_${Date.now()}.wav`;
        await FileSystem.writeAsStringAsync(fileUri, base64, { encoding: 'base64' });
        return fileUri;
    };

    const handleStartRecording = () => {
        setErrorMessage(null);
        setUsedOnnx(false);
        setStatusText('');
        pcmDataRef.current = [];

        if (!permissionGranted) {
            setErrorMessage('마이크 권한이 필요합니다');
            return;
        }

        try {
            LiveAudioStream.start();
            setState('recording');
            setRecordingTime(0);
            timerRef.current = setInterval(() => setRecordingTime(p => p + 1), 1000);
        } catch (err) {
            console.error('Recording error:', err);
            setErrorMessage(`녹음 오류: ${err.message}`);
        }
    };

    const handleStopRecording = async () => {
        if (timerRef.current) clearInterval(timerRef.current);

        try {
            LiveAudioStream.stop();
            setState('encoding');
            setStatusText('오디오 데이터 처리 중...');
            setProgress(10);

            // Wait a moment for last chunks
            await new Promise(r => setTimeout(r, 200));

            const rawSamples = new Float32Array(pcmDataRef.current);
            console.log(`Captured ${rawSamples.length} samples at ${TARGET_SAMPLE_RATE}Hz`);

            if (rawSamples.length === 0) throw new Error('녹음된 데이터가 없습니다');

            // Resample
            const audio8k = resampleTo8kHz(rawSamples);
            console.log(`Resampled to ${audio8k.length} samples at 8kHz`);
            setProgress(30);

            const messageBits = generateMessage();
            setBitProbs(messageBits.map(b => b ? 1.0 : 0.0));
            setProgress(40);

            let encoded;
            if (inference.onnxAvailable) {
                try {
                    setStatusText('ONNX 인코딩 중...');
                    await inference.loadEncoder();
                    setProgress(60);

                    const result = await inference.runEncoder(audio8k, messageBits);
                    encoded = result.encoded;
                    setUsedOnnx(true);
                    console.log('ONNX encoding successful');
                } catch (e) {
                    console.warn('ONNX failed:', e);
                    setStatusText('Fallback 인코딩...');
                    encoded = embedWatermarkSimple(audio8k, messageBits);
                }
            } else {
                setStatusText('Fallback 인코딩...');
                encoded = embedWatermarkSimple(audio8k, messageBits);
            }

            setProgress(80);
            setStatusText('저장 중...');
            const uri = await createWavFile(encoded);
            setEncodedUri(uri);
            setProgress(100);
            setState('complete');

        } catch (err) {
            console.error('Encoding error:', err);
            setErrorMessage(`오류: ${err.message}`);
            setState('idle');
        }
    };

    const handleDownload = async () => {
        if (encodedUri) {
            if (await Sharing.isAvailableAsync()) {
                await Sharing.shareAsync(encodedUri);
            } else {
                Alert.alert('저장됨', encodedUri);
            }
        }
    };

    const handleReset = () => {
        setState('idle');
        setRecordingTime(0);
        setProgress(0);
        setBitProbs(null);
        setEncodedUri(null);
        setErrorMessage(null);
        pcmDataRef.current = [];
    };

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            LiveAudioStream.stop();
        };
    }, []);

    // ... Render code remains similar, copying styles from previous version for consistency ...

    // Render bit matrix preview
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
                    <View style={styles.micCircle}>
                        <Text style={styles.micIcon}>🎤</Text>
                    </View>
                    <Text style={styles.title}>Ready to Call</Text>
                    <Text style={styles.subtitle}>실시간 오디오 스트림을 캡처합니다</Text>

                    {!permissionGranted && (
                        <View style={styles.errorBox}>
                            <Text style={styles.errorText}>마이크 권한이 필요합니다</Text>
                        </View>
                    )}

                    {!inference.onnxAvailable && (
                        <View style={styles.warningBox}>
                            <Text style={styles.warningText}>⚠️ ONNX Fallback 모드</Text>
                        </View>
                    )}

                    {inference.onnxAvailable && (
                        <View style={styles.successBox}>
                            <Text style={styles.successText}>✓ ONNX 모델 & Live Audio 준비됨</Text>
                        </View>
                    )}

                    {errorMessage && (
                        <View style={styles.errorBox}>
                            <Text style={styles.errorText}>{errorMessage}</Text>
                        </View>
                    )}
                </View>

                <TouchableOpacity
                    onPress={handleStartRecording}
                    style={[styles.startButton, !permissionGranted && styles.disabledButton]}
                    disabled={!permissionGranted}
                >
                    <Text style={styles.startButtonIcon}>📞</Text>
                </TouchableOpacity>
                <Text style={styles.startLabel}>Start Recording</Text>
            </View>
        );
    }

    // RECORDING STATE
    if (state === 'recording') {
        return (
            <View style={styles.container}>
                <TouchableOpacity onPress={onBack} style={styles.backButton}>
                    <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>

                <View style={styles.recordingBadge}>
                    <View style={styles.recordingDot} />
                    <Text style={styles.recordingText}>Recording (PCM)</Text>
                </View>

                <View style={styles.centerContent}>
                    <Text style={styles.timer}>{formatTime(recordingTime)}</Text>
                    <View style={styles.waveform}>
                        {Array(20).fill(0).map((_, i) => (
                            <View
                                key={i}
                                style={[styles.waveBar, { height: 20 + Math.random() * 30 }]}
                            />
                        ))}
                    </View>
                </View>

                <TouchableOpacity onPress={handleStopRecording} style={styles.stopButton}>
                    <View style={styles.stopIcon} />
                </TouchableOpacity>
                <Text style={styles.stopLabel}>Stop & Encode</Text>
            </View>
        );
    }

    // ENCODING & COMPLETE STATES (Same as before)
    if (state === 'encoding') {
        return (
            <View style={styles.container}>
                <TouchableOpacity onPress={onBack} style={styles.backButton}>
                    <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>

                <View style={styles.centerContent}>
                    <Text style={styles.title}>Encoding Watermark</Text>
                    <Text style={styles.subtitle}>{statusText || '처리 중...'}</Text>

                    <View style={styles.progressContainer}>
                        <View style={[styles.progressBar, { width: `${progress}%` }]} />
                    </View>
                    <Text style={styles.progressText}>{progress}%</Text>

                    {renderBitMatrix()}
                </View>
            </View>
        );
    }

    if (state === 'complete') {
        return (
            <View style={[styles.container, styles.completeContainer]}>
                <TouchableOpacity onPress={onBack} style={styles.backButton}>
                    <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>

                <View style={styles.centerContent}>
                    <View style={styles.successCircle}>
                        <Text style={styles.successIcon}>✓</Text>
                    </View>
                    <Text style={styles.successTitle}>Encoding Complete</Text>
                    <Text style={styles.subtitle}>워터마크가 삽입된 오디오 파일이 준비되었습니다</Text>

                    <View style={[styles.methodBadge, usedOnnx ? styles.onnxBadge : styles.fallbackBadge]}>
                        <Text style={styles.methodText}>
                            {usedOnnx ? '🧠 ONNX Model' : '📊 Fallback'}
                        </Text>
                    </View>

                    {renderBitMatrix()}

                    <TouchableOpacity onPress={handleDownload} style={styles.downloadButton}>
                        <Text style={styles.downloadIcon}>⬇️</Text>
                        <Text style={styles.downloadText}>Download WAV</Text>
                    </TouchableOpacity>
                </View>

                <TouchableOpacity onPress={handleReset} style={styles.resetButton}>
                    <Text style={styles.resetText}>🔄 Record Another</Text>
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
    completeContainer: {
        backgroundColor: '#0c1a0f',
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
    micCircle: {
        width: 128,
        height: 128,
        borderRadius: 64,
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 4,
        borderColor: 'rgba(34, 197, 94, 0.5)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    micIcon: {
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
        marginBottom: 16,
    },
    warningBox: {
        backgroundColor: 'rgba(234, 179, 8, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(234, 179, 8, 0.3)',
        borderRadius: 12,
        padding: 12,
        marginTop: 8,
    },
    warningText: {
        color: '#fbbf24',
        fontSize: 12,
        textAlign: 'center',
    },
    successBox: {
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderWidth: 1,
        borderColor: 'rgba(34, 197, 94, 0.3)',
        borderRadius: 12,
        padding: 12,
        marginTop: 8,
    },
    successText: {
        color: '#22c55e',
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
    },
    errorText: {
        color: '#f87171',
        fontSize: 14,
        textAlign: 'center',
    },
    startButton: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#22c55e',
        alignItems: 'center',
        justifyContent: 'center',
    },
    disabledButton: {
        backgroundColor: '#374151',
        opacity: 0.5,
    },
    startButtonIcon: {
        fontSize: 36,
    },
    startLabel: {
        color: '#22c55e',
        fontSize: 14,
        marginTop: 8,
    },
    recordingBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: 'rgba(239, 68, 68, 0.3)',
    },
    recordingDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#ef4444',
        marginRight: 8,
    },
    recordingText: {
        color: '#f87171',
        fontSize: 14,
    },
    timer: {
        fontSize: 64,
        fontWeight: 'bold',
        color: '#fff',
        fontFamily: 'monospace',
        marginBottom: 32,
    },
    waveform: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    waveBar: {
        width: 6,
        backgroundColor: '#22c55e',
        borderRadius: 3,
    },
    stopButton: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#ef4444',
        alignItems: 'center',
        justifyContent: 'center',
    },
    stopIcon: {
        width: 28,
        height: 28,
        backgroundColor: '#fff',
        borderRadius: 4,
    },
    stopLabel: {
        color: '#f87171',
        fontSize: 14,
        marginTop: 8,
    },
    progressContainer: {
        width: '80%',
        height: 8,
        backgroundColor: '#374151',
        borderRadius: 4,
        marginVertical: 16,
        overflow: 'hidden',
    },
    progressBar: {
        height: '100%',
        backgroundColor: '#22c55e',
    },
    progressText: {
        color: '#9ca3af',
        fontSize: 14,
        marginBottom: 24,
    },
    bitMatrix: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        width: 196,
        gap: 2,
        padding: 12,
        backgroundColor: 'rgba(55, 65, 81, 0.3)',
        borderRadius: 12,
        marginTop: 16,
    },
    bit: {
        width: 10,
        height: 10,
        borderRadius: 2,
    },
    successCircle: {
        width: 96,
        height: 96,
        borderRadius: 48,
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderWidth: 4,
        borderColor: '#22c55e',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    successIcon: {
        fontSize: 36,
        color: '#22c55e',
    },
    successTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#22c55e',
        marginBottom: 8,
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
    downloadButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#3b82f6',
        paddingHorizontal: 32,
        paddingVertical: 16,
        borderRadius: 12,
        marginTop: 24,
        gap: 12,
    },
    downloadIcon: {
        fontSize: 20,
    },
    downloadText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    resetButton: {
        marginTop: 24,
    },
    resetText: {
        color: '#9ca3af',
        fontSize: 16,
    },
});
