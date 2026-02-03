import React, { useState, useRef, useCallback } from 'react';
import ModeSelector from '../components/ModeSelector';
import SenderMode from '../components/SenderMode';
import IncomingCallScreen from '../components/IncomingCallScreen';
import CallResultScreen from '../components/CallResultScreen';
import LiveAnalysisScreen from '../components/LiveAnalysisScreen';
import { useInference } from '../hooks/useInference';
import { useAudioCapture } from '../hooks/useAudioCapture';
import { calculateCRC16 } from '../utils/crc';

/**
 * PhoneSimulator - Main page with mode selection
 * Modes: select → sender | receiver
 */
export default function PhoneSimulator() {
    const [mode, setMode] = useState('select'); // select, sender, receiver

    // Receiver mode states
    const [receiverState, setReceiverState] = useState('idle');
    const [audioFile, setAudioFile] = useState(null);
    const [isValid, setIsValid] = useState(null);
    const [crcValid, setCrcValid] = useState(null);
    const [payload, setPayload] = useState(null);
    const [progress, setProgress] = useState(0);
    const [bitProbs, setBitProbs] = useState(null);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [totalFrames, setTotalFrames] = useState(0);
    const [confidence, setConfidence] = useState(null);

    const fileInputRef = useRef(null);
    const inference = useInference();
    const audioCapture = useAudioCapture();
    const analysisCompleteRef = useRef(false);

    // Mode selection
    const handleSelectMode = useCallback((selectedMode) => {
        setMode(selectedMode);
        if (selectedMode === 'receiver') {
            setReceiverState('idle');
        }
    }, []);

    // Back to mode selection
    const handleBack = useCallback(() => {
        setMode('select');
        setReceiverState('idle');
        resetReceiverState();
    }, []);

    // Reset receiver state
    const resetReceiverState = () => {
        setAudioFile(null);
        setIsValid(null);
        setCrcValid(null);
        setPayload(null);
        setProgress(0);
        setBitProbs(null);
        setCurrentFrame(0);
        setTotalFrames(0);
        setConfidence(null);
        audioCapture.clearAudio();
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Receiver: Handle "Answer" button
    const handleAnswer = useCallback(() => {
        setReceiverState('uploading');
        fileInputRef.current?.click();
    }, []);

    // Receiver: Handle "Decline" button
    const handleDecline = useCallback(() => {
        handleBack();
    }, [handleBack]);

    // Receiver: Handle file selection
    const handleFileChange = useCallback(async (e) => {
        const file = e.target.files?.[0];
        if (!file) {
            setReceiverState('idle');
            return;
        }

        setAudioFile(file);
        setReceiverState('analyzing');
        setProgress(0);
        setBitProbs(null);
        setCurrentFrame(0);
        setConfidence(null);
        setCrcValid(null);
        setIsValid(null);
        analysisCompleteRef.current = false;

        try {
            if (!inference.isReady) {
                setProgress(5);
                await inference.loadDecoder();
            }

            if (!inference.isReady) {
                setReceiverState('result');
                return;
            }

            setProgress(10);

            const audioData = await audioCapture.loadFromFile(file);

            if (!audioData || audioData.length === 0) {
                setReceiverState('result');
                return;
            }

            setProgress(20);

            const FRAME_SAMPLES = inference.FRAME_SAMPLES || 320;
            const numFrames = Math.floor(audioData.length / FRAME_SAMPLES);
            setTotalFrames(numFrames);

            const CHUNK_SIZE = 8000;
            const numChunks = Math.ceil(audioData.length / CHUNK_SIZE);
            let accumulatedProbs = null;
            let chunksProcessed = 0;

            for (let i = 0; i < audioData.length; i += CHUNK_SIZE) {
                if (analysisCompleteRef.current) break;

                const audioSoFar = audioData.slice(0, Math.min(i + CHUNK_SIZE, audioData.length));
                const result = await inference.runDecoder(audioSoFar);

                chunksProcessed++;
                const progressPercent = 20 + Math.floor((chunksProcessed / numChunks) * 70);
                setProgress(progressPercent);
                setCurrentFrame(Math.floor(audioSoFar.length / FRAME_SAMPLES));

                if (result && result.bits128) {
                    const probs = Array.from(result.bits128);
                    setBitProbs(probs);
                    accumulatedProbs = probs;

                    const avgConf = probs.reduce((sum, p) => sum + Math.abs(p - 0.5), 0) / 128;
                    setConfidence(avgConf);

                    const bits = probs.map(p => p > 0.5 ? 1 : 0);
                    const dataBits = bits.slice(0, 112);
                    const crcBits = bits.slice(112, 128);

                    let actualCRC = 0;
                    for (let j = 0; j < 16; j++) {
                        if (crcBits[j]) actualCRC |= (1 << (15 - j));
                    }

                    const expectedCRC = calculateCRC16(dataBits);
                    const crcMatch = expectedCRC === actualCRC;
                    setCrcValid(crcMatch);

                    const hasWatermark = avgConf > 0.1;
                    setIsValid(hasWatermark && crcMatch);
                }

                await new Promise(resolve => setTimeout(resolve, 100));
            }

            setProgress(100);

            if (accumulatedProbs) {
                const bits = accumulatedProbs.map(p => p > 0.5 ? 1 : 0);
                setPayload(bits);
            }

            await new Promise(resolve => setTimeout(resolve, 800));
            setReceiverState('result');

        } catch (err) {
            console.error('Analysis error:', err);
            setIsValid(false);
            setCrcValid(false);
            setPayload(null);
            setReceiverState('result');
        }
    }, [inference, audioCapture]);

    // Receiver: End call
    const handleEndCall = useCallback(() => {
        analysisCompleteRef.current = true;
        resetReceiverState();
        setReceiverState('idle');
    }, []);

    // Hidden file input
    const fileInput = (
        <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm"
            onChange={handleFileChange}
            className="hidden"
        />
    );

    // MODE: SELECT
    if (mode === 'select') {
        return <ModeSelector onSelectMode={handleSelectMode} />;
    }

    // MODE: SENDER
    if (mode === 'sender') {
        return <SenderMode onBack={handleBack} />;
    }

    // MODE: RECEIVER
    if (mode === 'receiver') {
        if (receiverState === 'idle' || receiverState === 'uploading') {
            return (
                <>
                    {fileInput}
                    <IncomingCallScreen onAnswer={handleAnswer} onDecline={handleDecline} />
                </>
            );
        }

        if (receiverState === 'analyzing') {
            return (
                <>
                    {fileInput}
                    <LiveAnalysisScreen
                        progress={progress}
                        bitProbs={bitProbs}
                        currentFrame={currentFrame}
                        totalFrames={totalFrames}
                        crcValid={crcValid}
                        isValid={isValid}
                        confidence={confidence}
                        fileName={audioFile?.name}
                    />
                </>
            );
        }

        if (receiverState === 'result') {
            return (
                <>
                    {fileInput}
                    <CallResultScreen
                        isValid={isValid}
                        crcValid={crcValid}
                        payload={payload}
                        onEndCall={handleEndCall}
                    />
                </>
            );
        }
    }

    return null;
}
