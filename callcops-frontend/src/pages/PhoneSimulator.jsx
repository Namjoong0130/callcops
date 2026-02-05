import React, { useState, useRef, useCallback, useEffect } from 'react';
import IncomingCallScreen from '../components/IncomingCallScreen';
import CallResultScreen from '../components/CallResultScreen';
import LiveAnalysisScreen from '../components/LiveAnalysisScreen';
import { useInference } from '../hooks/useInference';
import { useAudioCapture } from '../hooks/useAudioCapture';
import { verifyRS } from '../utils/reedSolomon';

/**
 * PhoneSimulator - Main page simulating a phone call experience
 * States: idle → uploading → analyzing → result
 * Now with real-time detection visualization
 */
export default function PhoneSimulator() {
    const [state, setState] = useState('idle');
    const [audioFile, setAudioFile] = useState(null);
    const [isValid, setIsValid] = useState(null);
    const [crcValid, setCrcValid] = useState(null);
    const [payload, setPayload] = useState(null);
    const [progress, setProgress] = useState(0);

    // Real-time visualization state
    const [bitProbs, setBitProbs] = useState(null);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [totalFrames, setTotalFrames] = useState(0);
    const [confidence, setConfidence] = useState(null);

    const fileInputRef = useRef(null);
    const inference = useInference();
    const audioCapture = useAudioCapture();
    const analysisCompleteRef = useRef(false);

    // Handle "Answer" button
    const handleAnswer = useCallback(() => {
        setState('uploading');
        fileInputRef.current?.click();
    }, []);

    // Handle "Decline" button
    const handleDecline = useCallback(() => {
        setState('idle');
    }, []);

    // Process audio with progressive updates
    const handleFileChange = useCallback(async (e) => {
        const file = e.target.files?.[0];
        if (!file) {
            setState('idle');
            return;
        }

        setAudioFile(file);
        setState('analyzing');
        setProgress(0);
        setBitProbs(null);
        setCurrentFrame(0);
        setConfidence(null);
        setCrcValid(null);
        setIsValid(null);
        analysisCompleteRef.current = false;

        try {
            // Load decoder if not ready
            if (!inference.isReady) {
                setProgress(5);
                await inference.loadDecoder();
            }

            if (!inference.isReady) {
                console.error('Decoder failed to load');
                setState('result');
                return;
            }

            setProgress(10);

            // Load audio using lossless WAV parsing
            const audioData = await audioCapture.loadFromFile(file);

            if (!audioData || audioData.length === 0) {
                console.error('Failed to load audio data');
                setState('result');
                return;
            }

            setProgress(20);

            // Calculate frame info
            const FRAME_SAMPLES = inference.FRAME_SAMPLES || 320;
            const numFrames = Math.floor(audioData.length / FRAME_SAMPLES);
            setTotalFrames(numFrames);

            // Process in chunks for progressive visualization
            const CHUNK_SIZE = 8000; // 1 second chunks for updates
            const numChunks = Math.ceil(audioData.length / CHUNK_SIZE);
            let accumulatedProbs = null;
            let chunksProcessed = 0;

            for (let i = 0; i < audioData.length; i += CHUNK_SIZE) {
                if (analysisCompleteRef.current) break;

                const chunk = audioData.slice(i, Math.min(i + CHUNK_SIZE, audioData.length));
                const chunkFrames = Math.floor(chunk.length / FRAME_SAMPLES);

                // Run decoder on accumulated audio up to this point
                const audioSoFar = audioData.slice(0, i + chunk.length);
                const result = await inference.runDecoder(audioSoFar);

                chunksProcessed++;
                const progressPercent = 20 + Math.floor((chunksProcessed / numChunks) * 70);
                setProgress(progressPercent);
                setCurrentFrame(Math.floor((i + chunk.length) / FRAME_SAMPLES));

                if (result && result.bits128) {
                    const probs = Array.from(result.bits128);
                    setBitProbs(probs);
                    accumulatedProbs = probs;

                    // Calculate and update confidence
                    const avgConf = probs.reduce((sum, p) => sum + Math.abs(p - 0.5), 0) / 128;
                    setConfidence(avgConf);

                    // Verify with Reed-Solomon in real-time
                    const rsResult = verifyRS(probs);
                    setCrcValid(rsResult.isValid);  // Re-using crcValid state for RS validity

                    // Update validity status
                    const hasWatermark = avgConf > 0.1;
                    setIsValid(hasWatermark && rsResult.isValid);
                }

                // Small delay for visual effect
                await new Promise(resolve => setTimeout(resolve, 100));
            }

            // Final result
            setProgress(100);

            if (accumulatedProbs) {
                const bits = accumulatedProbs.map(p => p > 0.5 ? 1 : 0);
                setPayload(bits);
            }

            // Give user a moment to see the final analysis state
            await new Promise(resolve => setTimeout(resolve, 800));

            setState('result');

        } catch (err) {
            console.error('Analysis error:', err);
            setIsValid(false);
            setCrcValid(false);
            setPayload(null);
            setState('result');
        }
    }, [inference, audioCapture]);

    // Handle end call
    const handleEndCall = useCallback(() => {
        analysisCompleteRef.current = true;
        setState('idle');
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
    }, [audioCapture]);

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

    // Render based on state
    if (state === 'idle' || state === 'uploading') {
        return (
            <>
                {fileInput}
                <IncomingCallScreen onAnswer={handleAnswer} onDecline={handleDecline} />
            </>
        );
    }

    if (state === 'analyzing') {
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

    if (state === 'result') {
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

    return null;
}
