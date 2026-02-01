/**
 * useAudioCapture Hook
 * 
 * Handles microphone capture and file upload with 8kHz resampling.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { resampleTo8kHz, toMono, SAMPLE_RATE, WINDOW_SIZE } from '../utils/audioProcessor';

export function useAudioCapture() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, recording, processing
  
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const audioBufferRef = useRef([]);
  const onAudioChunkRef = useRef(null);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);
  
  /**
   * Start microphone recording
   */
  const startRecording = useCallback(async (onAudioChunk = null) => {
    try {
      setError(null);
      setStatus('recording');
      onAudioChunkRef.current = onAudioChunk;
      
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: 48000 },
          channelCount: { ideal: 1 },
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      
      mediaStreamRef.current = stream;
      
      // Create audio context
      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      // Use ScriptProcessor for real-time audio access
      // Note: ScriptProcessor is deprecated but still widely supported
      // AudioWorklet would be the modern alternative
      const bufferSize = 4096;
      scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(
        bufferSize,
        1,
        1
      );
      
      audioBufferRef.current = [];
      let accumulatedSamples = new Float32Array(0);
      const sourceSampleRate = audioContextRef.current.sampleRate;
      const targetWindowSize = WINDOW_SIZE; // 3200 samples @ 8kHz
      const sourceWindowSize = Math.ceil(targetWindowSize * (sourceSampleRate / SAMPLE_RATE));
      
      scriptProcessorRef.current.onaudioprocess = async (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        
        // Accumulate samples
        const newAccumulated = new Float32Array(accumulatedSamples.length + inputData.length);
        newAccumulated.set(accumulatedSamples);
        newAccumulated.set(inputData, accumulatedSamples.length);
        accumulatedSamples = newAccumulated;
        
        // Process windows
        while (accumulatedSamples.length >= sourceWindowSize) {
          const windowData = accumulatedSamples.slice(0, sourceWindowSize);
          accumulatedSamples = accumulatedSamples.slice(sourceWindowSize);
          
          // Resample to 8kHz
          try {
            const tempBuffer = audioContextRef.current.createBuffer(
              1,
              windowData.length,
              sourceSampleRate
            );
            tempBuffer.getChannelData(0).set(windowData);
            
            const resampled = await resampleTo8kHz(tempBuffer);
            
            // Store for full audio
            audioBufferRef.current.push(resampled);
            
            // Call callback with chunk
            if (onAudioChunkRef.current) {
              onAudioChunkRef.current(resampled);
            }
          } catch (err) {
            console.error('Resampling error:', err);
          }
        }
      };
      
      source.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(audioContextRef.current.destination);
      
      setIsRecording(true);
    } catch (err) {
      setError(err.message);
      setStatus('idle');
      throw err;
    }
  }, []);
  
  /**
   * Stop recording
   */
  const stopRecording = useCallback(() => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Combine all audio buffers
    if (audioBufferRef.current.length > 0) {
      const totalLength = audioBufferRef.current.reduce((sum, buf) => sum + buf.length, 0);
      const fullAudio = new Float32Array(totalLength);
      let offset = 0;
      for (const buf of audioBufferRef.current) {
        fullAudio.set(buf, offset);
        offset += buf.length;
      }
      setAudioData(fullAudio);
    }
    
    audioBufferRef.current = [];
    setIsRecording(false);
    setStatus('idle');
  }, []);
  
  /**
   * Load audio from file
   */
  const loadFromFile = useCallback(async (file, onAudioChunk = null) => {
    try {
      setError(null);
      setStatus('processing');
      
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to mono
      const monoData = toMono(audioBuffer);
      
      // Create mono buffer for resampling
      const monoBuffer = audioContext.createBuffer(
        1,
        monoData.length,
        audioBuffer.sampleRate
      );
      monoBuffer.getChannelData(0).set(monoData);
      
      // Resample to 8kHz
      const resampled = await resampleTo8kHz(monoBuffer);
      
      await audioContext.close();
      
      setAudioData(resampled);
      setStatus('idle');
      
      // Process in chunks if callback provided
      if (onAudioChunk) {
        const chunkSize = WINDOW_SIZE;
        for (let i = 0; i + chunkSize <= resampled.length; i += chunkSize) {
          const chunk = resampled.slice(i, i + chunkSize);
          onAudioChunk(chunk, i / chunkSize);
        }
      }
      
      return resampled;
    } catch (err) {
      setError(err.message);
      setStatus('idle');
      throw err;
    }
  }, []);
  
  /**
   * Clear audio data
   */
  const clearAudio = useCallback(() => {
    setAudioData(null);
    setError(null);
    audioBufferRef.current = [];
  }, []);
  
  return {
    isRecording,
    audioData,
    setAudioData,
    error,
    status,
    startRecording,
    stopRecording,
    loadFromFile,
    clearAudio,
  };
}

export default useAudioCapture;
