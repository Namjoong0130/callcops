/**
 * EmbedPanel Component
 * 
 * Workflow for embedding watermark into audio.
 */

import { useState, useCallback, useRef } from 'react';
import { MessageInput } from './MessageInput';
import { AudioUploader } from './AudioUploader';
import { RealtimeEmbedDemo } from './RealtimeEmbedDemo';
import { AudioComparisonPanel } from './AudioComparisonPanel';

export function EmbedPanel({ 
  onEmbed,
  onVerify, 
  isLoading = false,
  isModelReady = false,
  error = null 
}) {
  const [sourceAudio, setSourceAudio] = useState(null);
  const [sourceFileName, setSourceFileName] = useState(null);
  const [message, setMessage] = useState(null);
  const [watermarkedAudio, setWatermarkedAudio] = useState(null);
  const [embedStatus, setEmbedStatus] = useState('idle'); // idle, embedding, done
  const audioRef = useRef(null);
  const [originalDuration, setOriginalDuration] = useState(null);
  const [embedProgress, setEmbedProgress] = useState(0);  // 0-100 progress
  
  // Handle source file selection
  const handleFileSelect = useCallback(async (file) => {
    try {
      setSourceFileName(file.name);
      setWatermarkedAudio(null);
      setEmbedStatus('idle');
      
      // Create audio context and decode
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 8000
      });
      
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to mono Float32Array
      const channelData = audioBuffer.getChannelData(0);
      let audioData = new Float32Array(channelData);
      
      // Resample to 8kHz if needed
      if (audioBuffer.sampleRate !== 8000) {
        const offlineCtx = new OfflineAudioContext(1, 
          Math.ceil(audioData.length * 8000 / audioBuffer.sampleRate), 8000);
        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start();
        const resampled = await offlineCtx.startRendering();
        audioData = new Float32Array(resampled.getChannelData(0));
      }
      
      setOriginalDuration(audioData.length / 8000);
      setSourceAudio(audioData);
      audioContext.close();
    } catch (err) {
      console.error('Failed to load audio:', err);
    }
  }, []);
  
  // Embed watermark
  const handleEmbed = useCallback(async () => {
    if (!sourceAudio || !message || !onEmbed) return;
    
    try {
      setEmbedStatus('embedding');
      setEmbedProgress(0);
      
      // Pass progress callback to get real-time updates
      const result = await onEmbed(sourceAudio, message, (progress) => {
        setEmbedProgress(progress);
      });
      
      setEmbedProgress(100);
      setWatermarkedAudio(result);
      setEmbedStatus('done');
    } catch (err) {
      console.error('Embed failed:', err);
      setEmbedProgress(0);
      setEmbedStatus('idle');
    }
  }, [sourceAudio, message, onEmbed]);
  
  // Download watermarked audio as WAV
  const handleDownload = useCallback(() => {
    if (!watermarkedAudio) return;
    
    // Create WAV file
    const sampleRate = 8000;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = watermarkedAudio.length * 2;
    const bufferSize = 44 + dataSize;
    
    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
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
    
    // Audio data
    for (let i = 0; i < watermarkedAudio.length; i++) {
      const sample = Math.max(-1, Math.min(1, watermarkedAudio[i]));
      view.setInt16(44 + i * 2, sample * 32767, true);
    }
    
    // Create download link
    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `watermarked_${sourceFileName || 'audio'}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }, [watermarkedAudio, sourceFileName]);
  
  // Verify with decoder
  const handleVerify = useCallback(() => {
    if (watermarkedAudio && onVerify) {
      onVerify(watermarkedAudio);
    }
  }, [watermarkedAudio, onVerify]);
  
  // Create audio URL for playback
  const getAudioUrl = useCallback((audioData) => {
    if (!audioData) return null;
    
    const sampleRate = 8000;
    const dataSize = audioData.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, buffer.byteLength - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    for (let i = 0; i < audioData.length; i++) {
      const sample = Math.max(-1, Math.min(1, audioData[i]));
      view.setInt16(44 + i * 2, sample * 32767, true);
    }
    
    return URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
  }, []);
  
  const canEmbed = sourceAudio && message && message.length === 128 && isModelReady;
  
  return (
    <div className="space-y-6">
      {/* Real-Time Demo */}
      <RealtimeEmbedDemo onEmbed={onEmbed} isModelReady={isModelReady} />
      
      {/* Divider */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-gray-700/50" />
        <span className="text-xs text-gray-500">또는 파일 업로드</span>
        <div className="flex-1 h-px bg-gray-700/50" />
      </div>
      
      {/* Step 1: Upload Audio */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-bold 
                         flex items-center justify-center">1</span>
          <h3 className="text-sm font-semibold text-gray-300">Upload Source Audio</h3>
        </div>
        
        {!sourceAudio ? (
          <AudioUploader onFileSelect={handleFileSelect} disabled={isLoading} />
        ) : (
          <div className="flex items-center gap-3 p-3 bg-surface/50 rounded-lg">
            <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-200 truncate">{sourceFileName}</p>
              <p className="text-xs text-gray-500">
                {(sourceAudio.length / 8000).toFixed(2)}s @ 8kHz • {sourceAudio.length.toLocaleString()} samples
                <span className="ml-1 text-blue-400">
                  ({Math.ceil(sourceAudio.length / 8000)}개 청크로 처리)
                </span>
              </p>
            </div>
            <button
              onClick={() => { setSourceAudio(null); setSourceFileName(null); setWatermarkedAudio(null); }}
              className="p-2 text-gray-400 hover:text-red-400 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
      </div>
      
      {/* Step 2: Set Message */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-bold 
                         flex items-center justify-center">2</span>
          <h3 className="text-sm font-semibold text-gray-300">Set Watermark Message</h3>
        </div>
        <MessageInput 
          message={message} 
          onChange={setMessage} 
          disabled={isLoading || embedStatus === 'embedding'}
        />
      </div>
      
      {/* Step 3: Embed */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-bold 
                         flex items-center justify-center">3</span>
          <h3 className="text-sm font-semibold text-gray-300">Embed Watermark</h3>
        </div>
        
        <button
          onClick={handleEmbed}
          disabled={!canEmbed || embedStatus === 'embedding'}
          className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2
                     transition-all duration-200
                     ${canEmbed && embedStatus !== 'embedding'
                       ? 'bg-gradient-to-r from-primary to-purple-500 text-white glow hover:opacity-90' 
                       : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                     }`}
        >
          {embedStatus === 'embedding' ? (
            <>
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              Embedding...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Embed Watermark
            </>
          )}
        </button>
        
        {/* Progress Bar */}
        {embedStatus === 'embedding' && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Processing...</span>
              <span>{embedProgress}%</span>
            </div>
            <div className="h-2 bg-surface rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-primary to-purple-500 transition-all duration-300"
                style={{ width: `${embedProgress}%` }}
              />
            </div>
            <p className="text-[10px] text-gray-500 mt-1 text-center">
              {originalDuration ? `${originalDuration.toFixed(1)}초 오디오 처리 중...` : 'Processing...'}
            </p>
          </div>
        )}
        
        {error && (
          <p className="mt-2 text-xs text-red-400">{error}</p>
        )}
      </div>
      
      {/* Step 4: Result */}
      {watermarkedAudio && (
        <div className="glass rounded-xl p-4 border border-green-500/30">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-6 h-6 rounded-full bg-green-500/20 text-green-400 text-xs font-bold 
                           flex items-center justify-center">✓</span>
            <h3 className="text-sm font-semibold text-green-400">Watermark Embedded!</h3>
          </div>
          
          {/* Audio preview */}
          <div className="mb-4">
            <p className="text-[10px] text-gray-500 mb-2">워터마크 삽입된 오디오</p>
            <audio 
              ref={audioRef}
              src={getAudioUrl(watermarkedAudio)} 
              controls 
              className="w-full h-10"
            />
          </div>
          
          {/* Audio Comparison */}
          <AudioComparisonPanel 
            originalAudio={sourceAudio}
            watermarkedAudio={watermarkedAudio}
          />
          
          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleDownload}
              className="flex-1 py-2.5 bg-green-500/20 hover:bg-green-500/30 text-green-400 
                       rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download WAV
            </button>
            <button
              onClick={handleVerify}
              className="flex-1 py-2.5 bg-primary/20 hover:bg-primary/30 text-primary 
                       rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Verify with Decoder
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default EmbedPanel;
