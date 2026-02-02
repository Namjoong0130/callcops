/**
 * CallCops Preview - Real-Time Audio Watermarking Demo
 * 
 * A serverless on-device inference demo using ONNX Runtime Web.
 * Supports both Embed (encoder) and Detect (decoder) modes.
 */

import { useState, useCallback, useEffect } from 'react';
import { useAudioCapture } from './hooks/useAudioCapture';
import { useInference } from './hooks/useInference';
import { WaveformView } from './components/WaveformView';
import { BitMatrixView } from './components/BitMatrixView';
import { MetricsPanel } from './components/MetricsPanel';
import { AudioUploader } from './components/AudioUploader';
import { EmbedPanel } from './components/EmbedPanel';
import { MessageComparison } from './components/MessageComparison';
import { CRCVerificationPanel } from './components/CRCVerificationPanel';
import { ProgressiveDetection } from './components/ProgressiveDetection';
import PhoneSimulator from './pages/PhoneSimulator';

// Check URL for phone mode
const isPhoneMode = () => window.location.pathname === '/phone' || window.location.hash === '#phone';

function MainApp() {
  const [appMode, setAppMode] = useState('detect'); // 'embed' or 'detect'
  const [detectMode, setDetectMode] = useState('idle'); // idle, recording, processing
  const [currentBitProbs, setCurrentBitProbs] = useState(null);
  const [currentFrameProbs, setCurrentFrameProbs] = useState(null);  // NEW: Frame-wise probs
  const [frameInfo, setFrameInfo] = useState(null);  // NEW: { numFrames, cycleCoverage }
  const [encoderReady, setEncoderReady] = useState(false);
  const [currentBitIndex, setCurrentBitIndex] = useState(-1); // Current bit during playback
  const [currentFrameIndex, setCurrentFrameIndex] = useState(-1); // NEW: Current frame
  const [isPlaying, setIsPlaying] = useState(false);
  const [originalMessage, setOriginalMessage] = useState(null);  // Store embedded message for comparison

  const audioCapture = useAudioCapture();
  const inference = useInference();

  // Handle real-time audio chunks from microphone
  const handleAudioChunk = useCallback(async (chunk) => {
    if (!inference.isReady) {
      return;
    }

    try {
      const result = await inference.runDecoder(chunk);
      setCurrentBitProbs(result.bits128);
      setCurrentFrameProbs(result.frameProbs);
      setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });
    } catch (err) {
      console.error('Inference error:', err);
    }
  }, [inference.isReady, inference.runDecoder]);

  // Start microphone recording
  const handleStartRecording = async () => {
    try {
      if (!inference.isReady) {
        await inference.loadDecoder();
      }

      setDetectMode('recording');
      await audioCapture.startRecording(handleAudioChunk);
    } catch (err) {
      setDetectMode('idle');
      console.error('Failed to start recording:', err);
    }
  };

  // Stop recording
  const handleStopRecording = () => {
    audioCapture.stopRecording();
    setDetectMode('idle');
  };

  // Handle file upload for detect mode
  const handleFileSelect = async (file) => {
    try {
      setDetectMode('processing');

      if (!inference.isReady) {
        await inference.loadDecoder();
      }

      const audioData = await audioCapture.loadFromFile(file);

      if (audioData && inference.isReady) {
        const result = await inference.runDecoder(audioData);
        setCurrentBitProbs(result.bits128);
        setCurrentFrameProbs(result.frameProbs);
        setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });
      }

      setDetectMode('idle');
    } catch (err) {
      setDetectMode('idle');
      console.error('Failed to process file:', err);
    }
  };

  // Handle playback time updates for real-time detection sync
  const handleTimeUpdate = useCallback((time) => {
    if (!audioCapture.audioData) return;
    // Use frame-based bit index
    const bitIndex = inference.getPlaybackBitIndex(time);
    const frameIndex = inference.getPlaybackFrameIndex(time);
    setCurrentBitIndex(bitIndex);
    setCurrentFrameIndex(frameIndex);
  }, [audioCapture.audioData, inference]);

  // Handle play state changes
  const handlePlayStateChange = useCallback((playing) => {
    setIsPlaying(playing);
    if (!playing) {
      // When stopped, show all bits
      setCurrentBitIndex(-1);
      setCurrentFrameIndex(-1);
    }
  }, []);

  // Random position detection - proves watermark works from any position
  const [randomPosition, setRandomPosition] = useState(null);
  const [multiChunkResults, setMultiChunkResults] = useState(null);

  const handleRandomPositionDetect = useCallback(async () => {
    if (!audioCapture.audioData || audioCapture.audioData.length < 8000) return;
    if (!inference.isReady) return;

    try {
      setDetectMode('processing');

      const audioLength = audioCapture.audioData.length;
      const durationSec = audioLength / inference.SAMPLE_RATE;

      if (durationSec < 1) {
        // For short audio, just run normal detection
        const result = await inference.runDecoder(audioCapture.audioData);
        setCurrentBitProbs(result.bits128);
        setCurrentFrameProbs(result.frameProbs);
        setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });
        setRandomPosition({ startTime: '0.00', endTime: durationSec.toFixed(2) });
      } else {
        // Pick a random start position
        const maxStart = Math.max(0, durationSec - 1);
        const randomStart = Math.random() * maxStart;

        const result = await inference.runDecoderAtPosition(audioCapture.audioData, randomStart, 1.0);
        setCurrentBitProbs(result.bits128);
        setCurrentFrameProbs(result.frameProbs);
        setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });
        setRandomPosition({
          startTime: randomStart.toFixed(2),
          endTime: (randomStart + 1).toFixed(2),
        });
      }

      setDetectMode('idle');
    } catch (err) {
      setDetectMode('idle');
      console.error('Random position detection failed:', err);
    }
  }, [audioCapture.audioData, inference]);

  // Detect full audio using frame-wise approach (shows cyclic nature)
  const handleDetectAllChunks = useCallback(async () => {
    if (!audioCapture.audioData || audioCapture.audioData.length < inference.FRAME_SAMPLES) return;
    if (!inference.isReady) return;

    try {
      setDetectMode('processing');
      setMultiChunkResults(null);

      // Run full audio detection (frame-wise)
      const result = await inference.runDecoder(audioCapture.audioData);

      setCurrentBitProbs(result.bits128);
      setCurrentFrameProbs(result.frameProbs);
      setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });

      // Calculate sync pattern match
      const syncPattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
      let syncMatch = 0;
      for (let j = 0; j < 16; j++) {
        if ((result.bits128[j] > 0.5 ? 1 : 0) === syncPattern[j]) syncMatch++;
      }

      setMultiChunkResults([{
        numFrames: result.numFrames,
        cycleCoverage: result.cycleCoverage.toFixed(2),
        syncScore: (syncMatch / 16 * 100).toFixed(0),
        avgConfidence: inference.calculateConfidence(result.bits128).toFixed(0),
      }]);

      setRandomPosition(null);
      setDetectMode('idle');
    } catch (err) {
      setDetectMode('idle');
      console.error('Full analysis failed:', err);
    }
  }, [audioCapture.audioData, inference]);

  // Handle embed operation
  const handleEmbed = async (audioData, message, onProgress) => {
    try {
      if (!encoderReady) {
        await inference.loadEncoder();
        setEncoderReady(true);
      }

      // Save original message for comparison
      setOriginalMessage(message);

      const watermarked = await inference.runEncoder(audioData, message, onProgress);
      return watermarked;
    } catch (err) {
      console.error('Embed failed:', err);
      throw err;
    }
  };

  // Handle verify (switch to detect mode and run decoder)
  const handleVerify = async (audioData) => {
    setAppMode('detect');

    // Small delay to allow UI update
    setTimeout(async () => {
      try {
        setDetectMode('processing');

        if (!inference.isReady) {
          await inference.loadDecoder();
        }

        // Set audio data in capture hook for waveform display
        audioCapture.setAudioData(audioData);

        const result = await inference.runDecoder(audioData);
        setCurrentBitProbs(result.bits128);
        setCurrentFrameProbs(result.frameProbs);
        setFrameInfo({ numFrames: result.numFrames, cycleCoverage: result.cycleCoverage });

        setDetectMode('idle');
      } catch (err) {
        setDetectMode('idle');
        console.error('Verify failed:', err);
      }
    }, 100);
  };

  // Load decoder on initial render
  useEffect(() => {
    inference.loadDecoder().catch(console.error);
  }, []);

  return (
    <div className="min-h-screen bg-surface p-4 md:p-8">
      <div className="max-w-[95%] mx-auto space-y-6">
        {/* Header */}
        <header className="glass rounded-2xl p-12">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            {/* Logo & Title */}
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center glow">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
                  CallCops
                </h1>
                <p className="text-sm text-gray-400">Audio Watermarking Program</p>
              </div>
            </div>

            {/* Mode Toggle */}
            <div className="flex items-center gap-3 pr-16">
              <div className="flex bg-surface/50 rounded-xl p-1">
                <button
                  onClick={() => setAppMode('embed')}
                  className={`w-32 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${appMode === 'embed'
                      ? 'bg-gradient-to-r from-primary to-purple-500 text-white shadow-lg shadow-primary/20'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                    }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Embed
                  </span>
                </button>
                <button
                  onClick={() => setAppMode('detect')}
                  className={`w-32 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${appMode === 'detect'
                      ? 'bg-gradient-to-r from-primary to-purple-500 text-white shadow-lg shadow-primary/20'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                    }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    Detect
                  </span>
                </button>
              </div>


            </div>
          </div>
        </header>

        {/* Mode-specific content */}
        {appMode === 'embed' ? (
          /* Embed Mode */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <EmbedPanel
                onEmbed={handleEmbed}
                onVerify={handleVerify}
                isLoading={inference.isLoading}
                isModelReady={true}
                error={inference.error}
              />
            </div>
            <div className="space-y-6">
              <div className="glass rounded-xl p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">How Embed Mode Works</h3>
                <ol className="text-xs text-gray-400 space-y-2">
                  <li className="flex gap-2">
                    <span className="text-primary font-bold">1.</span>
                    Upload an audio file (WAV, MP3, etc.)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-bold">2.</span>
                    Enter a 128-bit message or generate random
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-bold">3.</span>
                    Click "Embed Watermark" to embed invisibly
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-bold">4.</span>
                    Download the watermarked audio file
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary font-bold">5.</span>
                    Verify with Decoder to confirm embedding
                  </li>
                </ol>
              </div>

              {/* Stats */}
              <div className="glass rounded-xl p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Encoder Stats</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Model</p>
                    <p className="text-sm font-medium text-gray-200">{inference.encoderModelName}</p>
                  </div>
                  <div className="bg-surface/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Sample Rate</p>
                    <p className="text-sm font-medium text-gray-200">8 kHz</p>
                  </div>
                  <div className="bg-surface/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Payload</p>
                    <p className="text-sm font-medium text-gray-200">128 bits</p>
                  </div>
                  <div className="bg-surface/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Last Inference</p>
                    <p className="text-sm font-medium text-gray-200">
                      {inference.lastInferenceTime > 0
                        ? `${inference.lastInferenceTime.toFixed(0)} ms`
                        : '—'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Detect Mode */
          <>
            {/* Controls */}
            <div className="flex justify-center">
              <button
                onClick={detectMode === 'recording' ? handleStopRecording : handleStartRecording}
                disabled={detectMode === 'processing'}
                className={`
                  w-64 py-3 rounded-xl font-medium flex items-center justify-center gap-2
                  transition-all duration-200 shadow-xl
                  ${detectMode === 'recording'
                    ? 'bg-red-500 hover:bg-red-600 text-white glow'
                    : 'bg-primary hover:bg-primary/80 text-white shadow-primary/25'
                  }
                  ${detectMode === 'processing' ? 'opacity-50 cursor-not-allowed' : ''}
                `}
              >
                {detectMode === 'recording' ? (
                  <>
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-white"></span>
                    </span>
                    Stop Recording
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    Record
                  </>
                )}
              </button>
            </div>

            {/* Progressive Detection - Animated Real-time Effect */}
            {audioCapture.audioData && (
              <ProgressiveDetection
                audioData={audioCapture.audioData}
                onRunDecoder={inference.runDecoder}
                onProgressUpdate={(bits, frame) => {
                  setCurrentBitProbs(bits);
                  setCurrentFrameIndex(frame);
                }}
                onComplete={(finalBits) => {
                  setCurrentBitProbs(finalBits);
                  setDetectMode('idle');
                }}
              />
            )}

            {/* Waveform Section */}
            <section className="glass rounded-2xl p-12">
              <h2 className="text-lg font-semibold text-gray-200 mb-4">Waveform Analysis</h2>
              <WaveformView
                audioData={audioCapture.audioData}
                bitProbs={currentBitProbs}
                frameProbs={currentFrameProbs}
                onTimeUpdate={handleTimeUpdate}
                onPlayStateChange={handlePlayStateChange}
              />
            </section>

            {/* Spacer to force separation */}
            <div className="h-6"></div>

            {/* Bottom Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Bit Matrix */}
              <div className="lg:col-span-1">
                <BitMatrixView
                  bitProbs={currentBitProbs}
                  currentBitIndex={isPlaying ? currentBitIndex : -1}
                  isProgressive={true}
                  isPlaying={isPlaying}
                />
              </div>

              {/* Metrics */}
              <div className="lg:col-span-1">
                <MetricsPanel
                  bitProbs={currentBitProbs}
                  status={detectMode === 'recording' ? 'recording' : detectMode === 'processing' ? 'processing' : 'idle'}
                  inferenceTime={inference.lastInferenceTime}
                  error={inference.error || audioCapture.error}
                />
              </div>

              {/* Upload Section */}
              <div className="lg:col-span-1">
                <div className="glass rounded-xl p-10 h-full flex flex-col">
                  <h3 className="text-sm font-semibold text-gray-300 mb-5 flex-none">Upload Audio</h3>
                  <div className="flex-1 min-h-[150px]">
                    <AudioUploader
                      onFileSelect={handleFileSelect}
                      disabled={detectMode === 'recording' || detectMode === 'processing'}
                    />
                  </div>

                  {/* Random Position Detection - Proves cyclic watermark works */}
                  {audioCapture.audioData && audioCapture.audioData.length > 8000 && (
                    <div className="mt-4 pt-4 border-t border-gray-700/50">
                      <h4 className="text-xs font-semibold text-gray-400 mb-2">
                        🔀 임의 위치 탐지
                      </h4>
                      <p className="text-xs text-gray-500 mb-3">
                        워터마크는 반복 삽입되어 어느 위치에서든 탐지 가능합니다.
                      </p>

                      <div className="flex gap-2 mb-3">
                        <button
                          onClick={handleRandomPositionDetect}
                          disabled={detectMode === 'processing'}
                          className="flex-1 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 
                                   rounded-lg text-xs font-medium transition-colors
                                   disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          🎲 랜덤 위치
                        </button>
                        <button
                          onClick={handleDetectAllChunks}
                          disabled={detectMode === 'processing'}
                          className="flex-1 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 
                                   rounded-lg text-xs font-medium transition-colors
                                   disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          📊 전체 분석
                        </button>
                      </div>

                      {/* Current Position Info */}
                      {randomPosition && (
                        <div className="bg-surface/50 rounded-lg p-2 text-xs">
                          <div className="flex items-center justify-between text-gray-400">
                            <span>분석 구간:</span>
                            <span className="text-gray-200">
                              {randomPosition.startTime}s ~ {randomPosition.endTime}s
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-gray-400 mt-1">
                            <span>청크:</span>
                            <span className="text-gray-200">
                              {randomPosition.index + 1} / {randomPosition.total}
                              {randomPosition.isBest && <span className="ml-1 text-green-400">(최고점)</span>}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Multi-chunk Results */}
                      {multiChunkResults && (
                        <div className="mt-3">
                          <div className="text-xs text-gray-400 mb-2">
                            🔍 프레임 단위 분석 결과:
                          </div>
                          {multiChunkResults.map((r, i) => (
                            <div key={i} className="bg-surface/50 rounded-lg p-3 space-y-2">
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="flex justify-between text-gray-400">
                                  <span>프레임 수:</span>
                                  <span className="text-cyan-400 font-mono">{r.numFrames}</span>
                                </div>
                                <div className="flex justify-between text-gray-400">
                                  <span>사이클:</span>
                                  <span className="text-purple-400 font-mono">{r.cycleCoverage}x</span>
                                </div>
                                <div className="flex justify-between text-gray-400">
                                  <span>Sync 일치:</span>
                                  <span className={`font-mono ${parseFloat(r.syncScore) >= 80 ? 'text-green-400' :
                                    parseFloat(r.syncScore) >= 50 ? 'text-yellow-400' : 'text-red-400'
                                    }`}>{r.syncScore}%</span>
                                </div>
                                <div className="flex justify-between text-gray-400">
                                  <span>신뢰도:</span>
                                  <span className="text-blue-400 font-mono">{r.avgConfidence}%</span>
                                </div>
                              </div>
                              <div className="text-[10px] text-gray-500 text-center mt-2">
                                40ms × {r.numFrames} = {(r.numFrames * 40 / 1000).toFixed(2)}초
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Message Comparison Section - Only shows when verifying embedded audio */}
            {originalMessage && currentBitProbs && (
              <section className="glass rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Verification: Original vs Decoded
                </h2>
                <MessageComparison
                  originalMessage={originalMessage}
                  decodedProbs={currentBitProbs}
                />
                <CRCVerificationPanel
                  decodedMessage={currentBitProbs}
                  originalMessage={originalMessage}
                />
              </section>
            )}

            {/* CRC Verification - Shows even without original message */}
            {!originalMessage && currentBitProbs && (
              <section className="glass rounded-2xl p-6">
                <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
                  <span className="text-lg">🔒</span>
                  Payload Verification
                </h2>
                <CRCVerificationPanel
                  decodedMessage={currentBitProbs}
                />
              </section>
            )}
          </>
        )}

        {/* Footer */}
        <footer className="text-center text-xs text-gray-500 py-4">
          <p>CallCops Preview • On-device ONNX Inference • 8kHz Audio Watermarking</p>
          <p className="mt-1">
            Model: {appMode === 'embed' ? inference.encoderModelName : inference.decoderModelName} •
            Sample Rate: 8kHz • Payload: 128 bits
          </p>
        </footer>
      </div>
    </div>
  );
}

// Router wrapper that chooses between MainApp and PhoneSimulator
function App() {
  if (isPhoneMode()) {
    return <PhoneSimulator />;
  }
  return <MainApp />;
}

export default App;
