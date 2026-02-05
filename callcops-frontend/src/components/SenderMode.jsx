import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useInference } from '../hooks/useInference';
import { useAudioCapture } from '../hooks/useAudioCapture';
import { calculateCRC16 } from '../utils/crc';

/**
 * SenderMode - Record audio and encode with watermark
 * States: idle → recording → encoding → complete
 */
export default function SenderMode({ onBack }) {
    const [state, setState] = useState('idle');
    const [recordingTime, setRecordingTime] = useState(0);
    const [progress, setProgress] = useState(0);
    const [bitProbs, setBitProbs] = useState(null);
    const [encodedBlob, setEncodedBlob] = useState(null);
    const [message, setMessage] = useState(null);
    const [errorMessage, setErrorMessage] = useState(null);

    const inference = useInference();
    const audioCapture = useAudioCapture();
    const timerRef = useRef(null);
    const audioChunksRef = useRef([]);

    // Format time as MM:SS
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // Start recording
    const handleStartRecording = useCallback(async () => {
        setErrorMessage(null);

        try {
            // Check if browser supports getUserMedia
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setErrorMessage('이 브라우저는 마이크를 지원하지 않습니다');
                return;
            }

            // First try to get permission
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop());
            } catch (permErr) {
                setErrorMessage('마이크 권한이 필요합니다. 설정에서 허용해 주세요.');
                console.error('Microphone permission error:', permErr);
                return;
            }

            setState('recording');
            setRecordingTime(0);
            audioChunksRef.current = [];

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

            // Start microphone recording
            await audioCapture.startRecording((chunk) => {
                audioChunksRef.current.push(chunk);
            });
        } catch (err) {
            console.error('Recording error:', err);
            setErrorMessage(`녹음 오류: ${err.message || '알 수 없는 오류'}`);
            setState('idle');
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    }, [audioCapture]);

    // Stop recording and start encoding
    const handleStopRecording = useCallback(async () => {
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }

        audioCapture.stopRecording();
        setState('encoding');
        setProgress(0);

        try {
            if (!inference.isReady) {
                await inference.loadEncoder();
            }

            const audioData = audioCapture.audioData;
            if (!audioData || audioData.length === 0) {
                console.error('No audio data');
                setErrorMessage('녹음된 오디오가 없습니다');
                setState('idle');
                return;
            }

            setProgress(10);

            // Generate message (128 bits)
            const now = Math.floor(Date.now() / 1000);
            const syncPattern = 0xAAAA;
            const timestamp = now & 0xFFFFFFFF;
            const authCode = BigInt(Math.floor(Math.random() * 0xFFFFFFFFFFFFFFFF));

            const messageBits = [];

            for (let i = 15; i >= 0; i--) {
                messageBits.push((syncPattern >> i) & 1);
            }

            for (let i = 31; i >= 0; i--) {
                messageBits.push((timestamp >> i) & 1);
            }

            for (let i = 63; i >= 0; i--) {
                messageBits.push(Number((authCode >> BigInt(i)) & BigInt(1)));
            }

            const crc = calculateCRC16(messageBits.slice(0, 112));
            for (let i = 15; i >= 0; i--) {
                messageBits.push((crc >> i) & 1);
            }

            setMessage(messageBits);
            setProgress(20);
            setBitProbs(messageBits.map(b => b ? 1.0 : 0.0));
            setProgress(30);

            const result = await inference.runEncoder(audioData, messageBits);
            setProgress(80);

            if (result && result.encoded) {
                const wavBlob = createWavBlob(result.encoded, 8000);
                setEncodedBlob(wavBlob);
                setProgress(100);
                setState('complete');
            } else {
                setErrorMessage('인코딩 실패');
                setState('idle');
            }
        } catch (err) {
            console.error('Encoding error:', err);
            setErrorMessage(`인코딩 오류: ${err.message}`);
            setState('idle');
        }
    }, [audioCapture, inference]);

    const createWavBlob = (samples, sampleRate) => {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        const blockAlign = numChannels * (bitsPerSample / 8);
        const dataSize = samples.length * (bitsPerSample / 8);
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
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        const volume = 32767;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(44 + i * 2, sample * volume, true);
        }

        return new Blob([buffer], { type: 'audio/wav' });
    };

    const handleDownload = useCallback(() => {
        if (!encodedBlob) return;
        const url = URL.createObjectURL(encodedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `callcops_encoded_${Date.now()}.wav`;
        a.click();
        URL.revokeObjectURL(url);
    }, [encodedBlob]);

    const handleReset = useCallback(() => {
        if (timerRef.current) clearInterval(timerRef.current);
        audioCapture.stopRecording();
        audioCapture.clearAudio();
        setState('idle');
        setRecordingTime(0);
        setProgress(0);
        setBitProbs(null);
        setEncodedBlob(null);
        setMessage(null);
        setErrorMessage(null);
    }, [audioCapture]);

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    const renderBitMatrix = () => {
        if (!bitProbs) {
            return (
                <div className="grid grid-cols-16 gap-0.5">
                    {Array(128).fill(0).map((_, i) => (
                        <div key={i} className="w-3 h-3 bg-gray-700 rounded-sm" />
                    ))}
                </div>
            );
        }
        return (
            <div className="grid grid-cols-16 gap-0.5">
                {bitProbs.map((prob, i) => (
                    <div key={i} className={`w-3 h-3 rounded-sm ${prob > 0.5 ? 'bg-green-500' : 'bg-red-500'}`} />
                ))}
            </div>
        );
    };

    // IDLE STATE
    if (state === 'idle') {
        return (
            <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center justify-between py-12 px-6">
                <button onClick={onBack} className="self-start text-gray-400 hover:text-white flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Back
                </button>

                <div className="flex-1 flex flex-col items-center justify-center">
                    <div className="w-32 h-32 rounded-full bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-4 border-green-500/50 flex items-center justify-center mb-6">
                        <svg className="w-16 h-16 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">Ready to Call</h2>
                    <p className="text-gray-400 text-center mb-4">녹음 시작 후 음성에 워터마크를 삽입합니다</p>

                    {errorMessage && (
                        <div className="w-full max-w-xs mb-4 px-4 py-3 bg-red-500/20 border border-red-500/30 rounded-xl">
                            <p className="text-red-400 text-sm text-center">{errorMessage}</p>
                        </div>
                    )}
                </div>

                <button onClick={handleStartRecording} className="w-20 h-20 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center shadow-lg shadow-green-500/30 hover:scale-110 transition-transform active:scale-95">
                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                    </svg>
                </button>
                <span className="text-green-400 text-sm font-medium mt-4">Start Recording</span>
            </div>
        );
    }

    // RECORDING STATE
    if (state === 'recording') {
        return (
            <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center justify-between py-12 px-6">
                <div className="text-center">
                    <span className="inline-flex items-center gap-2 px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full text-red-400 text-sm">
                        <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                        Recording
                    </span>
                </div>

                <div className="flex-1 flex flex-col items-center justify-center">
                    <div className="text-6xl font-mono font-bold text-white mb-8">{formatTime(recordingTime)}</div>
                    <div className="flex items-center gap-1 h-16">
                        {Array(30).fill(0).map((_, i) => (
                            <div key={i} className="w-1.5 bg-green-500 rounded-full animate-pulse" style={{ height: `${20 + Math.random() * 20}px` }} />
                        ))}
                    </div>
                </div>

                <button onClick={handleStopRecording} className="w-20 h-20 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-500/30 hover:scale-110 transition-transform active:scale-95">
                    <div className="w-8 h-8 bg-white rounded-sm" />
                </button>
                <span className="text-red-400 text-sm font-medium mt-4">Stop & Encode</span>
            </div>
        );
    }

    // ENCODING STATE
    if (state === 'encoding') {
        return (
            <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center justify-center px-6">
                <h2 className="text-2xl font-bold text-white mb-2">Encoding Watermark</h2>
                <p className="text-gray-400 mb-8">128-bit 페이로드 삽입 중...</p>
                <div className="w-full max-w-xs mb-8">
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-300" style={{ width: `${progress}%` }} />
                    </div>
                    <p className="text-center text-gray-500 text-sm mt-2">{progress}%</p>
                </div>
                <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">{renderBitMatrix()}</div>
            </div>
        );
    }

    // COMPLETE STATE
    if (state === 'complete') {
        return (
            <div className="min-h-screen bg-gradient-to-b from-green-900/30 via-gray-900 to-gray-900 flex flex-col items-center justify-between py-12 px-6">
                <div />
                <div className="flex flex-col items-center">
                    <div className="w-24 h-24 rounded-full bg-green-500/20 border-4 border-green-500 flex items-center justify-center mb-6">
                        <svg className="w-12 h-12 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <h2 className="text-2xl font-bold text-green-400 mb-2">Encoding Complete</h2>
                    <p className="text-gray-400 mb-8 text-center">워터마크가 삽입된 오디오 파일이 준비되었습니다</p>
                    <div className="bg-gray-800/30 rounded-xl p-3 border border-gray-700/50 mb-8">{renderBitMatrix()}</div>
                    <button onClick={handleDownload} className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download WAV
                    </button>
                </div>
                <button onClick={handleReset} className="text-gray-400 hover:text-white flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Record Another
                </button>
            </div>
        );
    }

    return null;
}
