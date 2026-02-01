/**
 * RealtimeEmbedDemo Component
 * 
 * Demonstrates low-latency real-time watermark embedding
 * using live microphone input with chunk-based processing.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { RealtimeOscilloscope } from './RealtimeOscilloscope';

// Constants matching the encoder
const SAMPLE_RATE = 8000;
const FRAME_SAMPLES = 320;  // 40ms @ 8kHz
const CHUNK_SIZE = FRAME_SAMPLES * 4;  // 160ms chunks for low latency

export function RealtimeEmbedDemo({ onEmbed, isModelReady = false }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [message, setMessage] = useState(null);
  const [stats, setStats] = useState({
    latencyMs: 0,
    chunksProcessed: 0,
    totalSamples: 0,
    avgLatency: 0
  });
  const [audioLevels, setAudioLevels] = useState({ input: 0, output: 0 });
  
  // Buffers for oscilloscope visualization
  const [inputBuffer, setInputBuffer] = useState(new Float32Array(2048));
  const [outputBuffer, setOutputBuffer] = useState(new Float32Array(2048));
  
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const latenciesRef = useRef([]);
  const isStreamingRef = useRef(false);
  
  // Generate random 128-bit message
  const generateMessage = useCallback(() => {
    const msg = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      msg[i] = Math.random() > 0.5 ? 1 : 0;
    }
    setMessage(msg);
    return msg;
  }, []);
  
  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!onEmbed || !isModelReady) return;
    
    // Generate message if not set
    const currentMessage = message || generateMessage();
    
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      mediaStreamRef.current = stream;
      
      // Create audio context at 8kHz
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });
      audioContextRef.current = audioContext;
      
      // Create source from microphone
      const source = audioContext.createMediaStreamSource(stream);
      
      // Create script processor for chunk-based processing
      const bufferSize = 4096;  // Larger buffer for stability
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;
      
      let accumulatedSamples = new Float32Array(0);
      
      processor.onaudioprocess = async (e) => {
        if (!isStreamingRef.current) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        const outputData = e.outputBuffer.getChannelData(0);
        
        // Calculate input level
        const inputLevel = Math.sqrt(inputData.reduce((sum, s) => sum + s * s, 0) / inputData.length);
        
        // Accumulate samples
        const newAccumulated = new Float32Array(accumulatedSamples.length + inputData.length);
        newAccumulated.set(accumulatedSamples);
        newAccumulated.set(inputData, accumulatedSamples.length);
        accumulatedSamples = newAccumulated;
        
        // Process when we have enough samples
        if (accumulatedSamples.length >= CHUNK_SIZE) {
          const chunk = accumulatedSamples.slice(0, CHUNK_SIZE);
          accumulatedSamples = accumulatedSamples.slice(CHUNK_SIZE);
          
          // Measure latency
          const startTime = performance.now();
          
          try {
            // Embed watermark into chunk
            const watermarkedChunk = await onEmbed(chunk, currentMessage);
            
            const endTime = performance.now();
            const latency = endTime - startTime;
            
            // Track latencies
            latenciesRef.current.push(latency);
            if (latenciesRef.current.length > 100) {
              latenciesRef.current.shift();
            }
            
            const avgLatency = latenciesRef.current.reduce((a, b) => a + b, 0) / latenciesRef.current.length;
            
            // Calculate output level
            const outputLevel = Math.sqrt(watermarkedChunk.reduce((sum, s) => sum + s * s, 0) / watermarkedChunk.length);
            
            // Update stats
            setStats(prev => ({
              latencyMs: latency.toFixed(1),
              chunksProcessed: prev.chunksProcessed + 1,
              totalSamples: prev.totalSamples + chunk.length,
              avgLatency: avgLatency.toFixed(1)
            }));
            
            setAudioLevels({
              input: Math.min(inputLevel * 5, 1),
              output: Math.min(outputLevel * 5, 1)
            });
            
            // Update buffers for oscilloscope (keep last 2048 samples)
            setInputBuffer(prev => {
              const newBuf = new Float32Array(2048);
              newBuf.set(prev.slice(chunk.length));
              newBuf.set(chunk, 2048 - chunk.length);
              return newBuf;
            });
            
            setOutputBuffer(prev => {
              const newBuf = new Float32Array(2048);
              newBuf.set(prev.slice(watermarkedChunk.length));
              newBuf.set(watermarkedChunk, 2048 - watermarkedChunk.length);
              return newBuf;
            });
            
            // Output the watermarked audio (for playback if needed)
            // Note: In a real app, this would go to speakers
            for (let i = 0; i < outputData.length && i < watermarkedChunk.length; i++) {
              outputData[i] = watermarkedChunk[i];
            }
          } catch (err) {
            console.error('Chunk processing error:', err);
          }
        } else {
          // Pass through while accumulating
          outputData.set(inputData.slice(0, outputData.length));
        }
      };
      
      // Connect nodes
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      isStreamingRef.current = true;
      setIsStreaming(true);
      setStats({ latencyMs: 0, chunksProcessed: 0, totalSamples: 0, avgLatency: 0 });
      latenciesRef.current = [];
      
    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  }, [onEmbed, isModelReady, message, generateMessage]);
  
  // Stop streaming
  const stopStreaming = useCallback(() => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isStreamingRef.current) {
        stopStreaming();
      }
    };
  }, [stopStreaming]);
  
  const streamDuration = stats.totalSamples / SAMPLE_RATE;
  
  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Real-Time Streaming Demo
        </h3>
        <span className="text-xs text-gray-500">Low-Latency Embedding</span>
      </div>
      
      {/* Status */}
      {isStreaming ? (
        <div className="space-y-3">
          {/* Audio Levels */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-[10px] text-gray-500 mb-1">Input Level</p>
              <div className="h-2 bg-surface rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 transition-all duration-100"
                  style={{ width: `${audioLevels.input * 100}%` }}
                />
              </div>
            </div>
            <div>
              <p className="text-[10px] text-gray-500 mb-1">Output Level</p>
              <div className="h-2 bg-surface rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-100"
                  style={{ width: `${audioLevels.output * 100}%` }}
                />
              </div>
            </div>
          </div>
          
          {/* Stats Grid */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-surface/50 rounded-lg p-2 text-center">
              <p className="text-[10px] text-gray-500">Latency</p>
              <p className="text-lg font-mono font-bold text-green-400">{stats.latencyMs}</p>
              <p className="text-[10px] text-gray-500">ms</p>
            </div>
            <div className="bg-surface/50 rounded-lg p-2 text-center">
              <p className="text-[10px] text-gray-500">Avg Latency</p>
              <p className="text-lg font-mono font-bold text-blue-400">{stats.avgLatency}</p>
              <p className="text-[10px] text-gray-500">ms</p>
            </div>
            <div className="bg-surface/50 rounded-lg p-2 text-center">
              <p className="text-[10px] text-gray-500">Chunks</p>
              <p className="text-lg font-mono font-bold text-yellow-400">{stats.chunksProcessed}</p>
              <p className="text-[10px] text-gray-500">processed</p>
            </div>
            <div className="bg-surface/50 rounded-lg p-2 text-center">
              <p className="text-[10px] text-gray-500">Duration</p>
              <p className="text-lg font-mono font-bold text-purple-400">{streamDuration.toFixed(1)}</p>
              <p className="text-[10px] text-gray-500">sec</p>
            </div>
          </div>
          
          {/* Info */}
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
            <p className="text-xs text-green-400">
              🎤 실시간 워터마킹 진행 중...
            </p>
            <p className="text-[10px] text-gray-400 mt-1">
              마이크 입력 → 160ms 청크 → 워터마크 삽입 → 출력
            </p>
          </div>
          
          {/* Real-Time Oscilloscope */}
          <RealtimeOscilloscope
            inputBuffer={inputBuffer}
            outputBuffer={outputBuffer}
            isActive={isStreaming}
            width={400}
            height={120}
          />
          
          {/* Stop Button */}
          <button
            onClick={stopStreaming}
            className="w-full py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 
                     rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
            실시간 스트리밍 중지
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="bg-surface/50 rounded-lg p-4 text-center">
            <svg className="w-12 h-12 mx-auto text-gray-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            <p className="text-sm text-gray-400">
              마이크 입력을 실시간으로 워터마킹
            </p>
            <p className="text-xs text-gray-500 mt-1">
              160ms 청크 단위 저지연 처리
            </p>
          </div>
          
          {/* Start Button */}
          <button
            onClick={startStreaming}
            disabled={!isModelReady}
            className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all
              ${isModelReady
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:opacity-90 glow'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            실시간 스트리밍 시작
          </button>
          
          {/* Performance Note */}
          <div className="text-center text-[10px] text-gray-500">
            <p>🚀 목표 지연: &lt;100ms | 청크 크기: 160ms (1280 samples)</p>
            <p>WebAudio API + ONNX Runtime Web</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default RealtimeEmbedDemo;
