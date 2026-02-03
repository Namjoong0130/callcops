/**
 * SenderMode - Real-time audio watermark encoding
 *
 * Uses react-native-live-audio-stream to capture PCM data at 8kHz
 * and StreamingEncoderWrapper for frame-by-frame ONNX encoding.
 *
 * Timing: 40ms chunks (320 samples @ 8kHz) - matches frontend exactly.
 */
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, Platform, PermissionsAndroid } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import LiveAudioStream from 'react-native-live-audio-stream';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';
import { useInference } from '../hooks/useInference';
import { calculateCRC16 } from '../utils/crc';
import { Buffer } from 'buffer';
import {
  StreamingEncoderWrapper,
  FRAME_SAMPLES,
  PAYLOAD_LENGTH
} from '../utils/StreamingEncoderWrapper';

// Audio settings - Native 8kHz capture, NO resampling
const SAMPLE_RATE = 8000;
const WATERMARK_STRENGTH = 0.02;

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

  // Real-time visualization state
  const [inputWaveform, setInputWaveform] = useState(new Array(30).fill(0));
  const [outputWaveform, setOutputWaveform] = useState(new Array(30).fill(0));
  const [currentBitIndex, setCurrentBitIndex] = useState(0);

  const inference = useInference();
  const timerRef = useRef(null);

  // Streaming encoder wrapper (holds ONNX session + state)
  const streamingEncoderRef = useRef(null);

  // Audio buffers - Use regular Arrays for push/splice, convert to Float32Array when needed
  const inputBufferRef = useRef([]);          // Accumulate incoming PCM (regular Array)
  const encodedBufferRef = useRef([]);        // All encoded samples (regular Array)
  const messageBitsRef = useRef(null);        // 128-bit message as Float32Array
  const frameCountRef = useRef(0);            // For visualization throttling
  const useOnnxRef = useRef(false);           // Track ONNX availability

  // Configure audio session on mount
  // CRITICAL: Configure for RAW audio capture without voice processing
  useEffect(() => {
    (async () => {
      try {
        // iOS: Use 'measurement' mode to disable voice processing (AGC, NS, EC)
        // This gives us the cleanest, most unprocessed audio signal
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
          staysActiveInBackground: true,
          // Android settings
          shouldDuckAndroid: false,        // Don't reduce volume of other apps
          playThroughEarpieceAndroid: false,
          // iOS: Use AVAudioSessionModeDefault or avoid voice modes
          // The key is NOT using interruptionModeIOS that triggers voice processing
        });
        console.log('Audio mode configured for RAW capture');
      } catch (e) {
        console.warn('Failed to set audio mode:', e);
      }

      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO
        );
        setPermissionGranted(granted === PermissionsAndroid.RESULTS.GRANTED);
      } else {
        const response = await Audio.requestPermissionsAsync();
        if (response.status !== 'granted') {
          Alert.alert('Permission Error', `Microphone permission: ${response.status}`);
        }
        setPermissionGranted(response.status === 'granted');
      }
    })();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      LiveAudioStream.stop();
    };
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  /**
   * Generate 128-bit watermark message
   * Format: [16 sync | 32 timestamp | 64 auth | 16 CRC]
   * Returns Float32Array for direct use with ONNX
   */
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

    // Return as Float32Array for ONNX compatibility
    return new Float32Array(messageBits);
  };

  /**
   * Fallback watermark embedding (when ONNX unavailable)
   */
  const embedWatermarkFallback = (frame, frameIndex, messageBits) => {
    const bitIdx = frameIndex % PAYLOAD_LENGTH;
    const bit = messageBits[bitIdx];
    const output = new Float32Array(frame.length);

    for (let i = 0; i < frame.length; i++) {
      const watermark = bit ? WATERMARK_STRENGTH : -WATERMARK_STRENGTH;
      output[i] = frame[i] + watermark * Math.sin(2 * Math.PI * (frameIndex * FRAME_SAMPLES + i) / 10);
    }
    return output;
  };

  /**
   * Create WAV file from encoded samples
   */
  const createWavFile = async (samples) => {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = SAMPLE_RATE * numChannels * 2;
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
    view.setUint32(24, SAMPLE_RATE, true);
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
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }

    const base64 = global.btoa ? global.btoa(binary) : Buffer.from(binary, 'binary').toString('base64');
    const fileUri = FileSystem.documentDirectory + `callcops_encoded_${Date.now()}.wav`;
    await FileSystem.writeAsStringAsync(fileUri, base64, { encoding: 'base64' });
    return fileUri;
  };

  const handleStartRecording = async () => {
    setErrorMessage(null);
    setUsedOnnx(false);
    setStatusText('');

    // Reset all buffers - USE REGULAR ARRAYS (not Float32Array) for push/splice
    inputBufferRef.current = [];
    encodedBufferRef.current = [];
    frameCountRef.current = 0;
    streamingEncoderRef.current = null;
    useOnnxRef.current = false;
    setInputWaveform(new Array(30).fill(0));
    setOutputWaveform(new Array(30).fill(0));
    setCurrentBitIndex(0);

    if (!permissionGranted) {
      setErrorMessage('Microphone permission required');
      return;
    }

    try {
      // Generate 128-bit message as Float32Array
      const bits = generateMessage();
      messageBitsRef.current = bits;
      setBitProbs(Array.from(bits));

      // Configure Audio session for RAW capture (no voice processing)
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: true,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      // Initialize ONNX encoder if available
      if (inference.onnxAvailable) {
        try {
          const session = await inference.loadEncoder();
          if (session && inference.Tensor) {
            streamingEncoderRef.current = new StreamingEncoderWrapper(session, inference.Tensor);
            useOnnxRef.current = true;
            setUsedOnnx(true);
            console.log('ONNX StreamingEncoderWrapper initialized');
          } else {
            console.warn('Missing session or Tensor class');
          }
        } catch (e) {
          console.warn('ONNX encoder failed to load, using fallback:', e);
        }
      }

      console.log('Starting real-time recording at 8kHz with 320-sample frames...');

      // Initialize LiveAudioStream for RAW audio capture
      // CRITICAL: audioSource selection determines whether hardware DSP is applied
      //
      // Android AudioSource values:
      //   0 = DEFAULT (system default, may have processing)
      //   1 = MIC (raw microphone, minimal processing) ← BEST FOR WATERMARKING
      //   6 = VOICE_COMMUNICATION (AGC + NS + EC enabled) ← DESTROYS WATERMARK
      //   7 = VOICE_RECOGNITION (optimized for speech)
      //   9 = UNPROCESSED (raw PCM, Android 7.0+, device-dependent)
      //
      // iOS: react-native-live-audio-stream uses AVAudioSession internally.
      //      The audioSource value is ignored on iOS; processing depends on
      //      the AVAudioSession category/mode set via expo-av.
      //
      LiveAudioStream.init({
        sampleRate: SAMPLE_RATE,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 1, // MIC - Raw microphone without voice processing
        bufferSize: FRAME_SAMPLES, // 320 samples = 40ms @ 8kHz
      });

      // Frame-by-frame processing callback
      LiveAudioStream.on('data', async (base64Data) => {
        try {
          if (!base64Data) return;

          // Decode base64 to Buffer
          const pcmBuffer = Buffer.from(base64Data, 'base64');
          if (!pcmBuffer || pcmBuffer.length === 0) return;

          // Convert Int16 PCM to Float32 [-1, 1]
          const numSamples = Math.floor(pcmBuffer.length / 2);
          for (let i = 0; i < numSamples; i++) {
            const sample = pcmBuffer.readInt16LE(i * 2) / 32768.0;
            inputBufferRef.current.push(sample);  // Regular Array supports push()
          }

          // Process complete frames (320 samples each)
          while (inputBufferRef.current.length >= FRAME_SAMPLES) {
            // Extract frame using splice() - works on regular Array
            const frameArray = inputBufferRef.current.splice(0, FRAME_SAMPLES);
            const frame = new Float32Array(frameArray);  // Convert to Float32Array for processing

            let encodedFrame;
            const bitIdx = frameCountRef.current % PAYLOAD_LENGTH;

            // Try ONNX encoding
            if (useOnnxRef.current && streamingEncoderRef.current) {
              try {
                encodedFrame = await streamingEncoderRef.current.processFrame(
                  frame,
                  messageBitsRef.current
                );
              } catch (e) {
                console.warn('ONNX encoding failed:', e.message);
                encodedFrame = null;
              }
            }

            // Fallback encoding if ONNX failed or unavailable
            if (!encodedFrame) {
              encodedFrame = embedWatermarkFallback(frame, frameCountRef.current, messageBitsRef.current);
            }

            // Accumulate encoded samples
            for (let i = 0; i < encodedFrame.length; i++) {
              encodedBufferRef.current.push(encodedFrame[i]);
            }

            frameCountRef.current++;

            // Update visualization (every 4 frames)
            if (frameCountRef.current % 4 === 0) {
              setCurrentBitIndex(bitIdx);

              // RMS amplitude for visualization
              let inputSum = 0, outputSum = 0;
              for (let i = 0; i < frame.length; i++) {
                inputSum += frame[i] * frame[i];
                outputSum += encodedFrame[i] * encodedFrame[i];
              }
              const inputAmp = Math.sqrt(inputSum / frame.length);
              const outputAmp = Math.sqrt(outputSum / encodedFrame.length);

              setInputWaveform(prev => [...prev.slice(1), inputAmp * 5]);
              setOutputWaveform(prev => [...prev.slice(1), outputAmp * 5]);
            }
          }
        } catch (e) {
          console.warn('Audio chunk error:', e.message || e);
        }
      });

      LiveAudioStream.start();
      setState('recording');
      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime(p => p + 1), 1000);
      console.log(`Real-time encoding started (${useOnnxRef.current ? 'ONNX' : 'fallback'})`);

    } catch (err) {
      console.error('Failed to start recording', err);
      setErrorMessage('Recording start failed: ' + err.message);
    }
  };

  const handleStopRecording = async () => {
    if (timerRef.current) clearInterval(timerRef.current);

    try {
      console.log('Stopping real-time recording...');
      LiveAudioStream.stop();

      setState('encoding');
      setStatusText('Saving encoded audio...');
      setProgress(50);

      const encoded = encodedBufferRef.current;
      console.log(`Total encoded samples: ${encoded.length} (${(encoded.length / SAMPLE_RATE).toFixed(2)}s)`);
      console.log(`Total frames processed: ${frameCountRef.current}`);

      if (encoded.length === 0) {
        Alert.alert('Error', 'No audio captured. Please check microphone.');
        setState('idle');
        return;
      }

      setProgress(80);
      setStatusText('Creating WAV file...');
      const uri = await createWavFile(new Float32Array(encoded));
      setEncodedUri(uri);
      setProgress(100);
      setState('complete');
      console.log('Encoded file saved:', uri);

    } catch (err) {
      console.error('Stop recording error:', err);
      setErrorMessage(`Error: ${err.message}`);
      setState('idle');
    }
  };

  const handleDownload = async () => {
    if (encodedUri) {
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(encodedUri);
      } else {
        Alert.alert('Saved', encodedUri);
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
    streamingEncoderRef.current = null;
  };

  // Render bit matrix preview
  const renderBitMatrix = () => (
    <View style={styles.bitMatrix}>
      {(bitProbs || Array(128).fill(0.5)).map((prob, i) => (
        <View
          key={i}
          style={[
            styles.bit,
            { backgroundColor: prob > 0.5 ? '#22c55e' : '#ef4444' },
            i === currentBitIndex && state === 'recording' && styles.currentBit
          ]}
        />
      ))}
    </View>
  );

  // IDLE STATE
  if (state === 'idle') {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#7c2d12', '#831843', '#2e1065']}
          style={StyleSheet.absoluteFill}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        />
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.centerContent}>
          <View style={styles.micCircle}>
            <Ionicons name="mic" size={48} color="#fff" />
          </View>
          <Text style={styles.title}>Ready to Call</Text>
          <Text style={styles.subtitle}>8kHz real-time audio (40ms frames)</Text>

          {!permissionGranted && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>Microphone permission required</Text>
            </View>
          )}

          {!inference.onnxAvailable && (
            <View style={styles.warningBox}>
              <Text style={styles.warningText}>ONNX Fallback mode</Text>
            </View>
          )}

          {inference.onnxAvailable && (
            <View style={styles.successBox}>
              <Text style={styles.successText}>ONNX Model Ready</Text>
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
          <Ionicons name="call" size={36} color="#22c55e" />
        </TouchableOpacity>
      </View>
    );
  }

  // RECORDING STATE
  if (state === 'recording') {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#7c2d12', '#831843', '#2e1065']}
          style={StyleSheet.absoluteFill}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        />
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.recordingBadge}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>
            Real-time Encoding {usedOnnx ? '(ONNX)' : '(Fallback)'}
          </Text>
        </View>

        <View style={styles.centerContent}>
          <Text style={styles.timer}>{formatTime(recordingTime)}</Text>

          <Text style={styles.waveformLabel}>Input Audio</Text>
          <View style={styles.waveform}>
            {inputWaveform.map((amp, i) => (
              <View
                key={`in-${i}`}
                style={[styles.waveBar, styles.inputWaveBar, { height: Math.max(4, amp * 150) }]}
              />
            ))}
          </View>

          <Text style={styles.waveformLabel}>Bit Matrix ({currentBitIndex}/128)</Text>
          <View style={styles.bitMatrix}>
            {(bitProbs || Array(128).fill(0.5)).map((prob, i) => (
              <View
                key={i}
                style={[
                  styles.bit,
                  { backgroundColor: prob > 0.5 ? '#22c55e' : '#ef4444' },
                  i === currentBitIndex && styles.currentBit
                ]}
              />
            ))}
          </View>

          <Text style={styles.waveformLabel}>Encoded Audio</Text>
          <View style={styles.waveform}>
            {outputWaveform.map((amp, i) => (
              <View
                key={`out-${i}`}
                style={[styles.waveBar, styles.outputWaveBar, { height: Math.max(4, amp * 150) }]}
              />
            ))}
          </View>
        </View>

        <TouchableOpacity onPress={handleStopRecording} style={styles.stopButton}>
          <View style={styles.stopIcon} />
        </TouchableOpacity>
        <Text style={styles.stopLabel}>Stop Recording</Text>
      </View>
    );
  }

  // ENCODING STATE
  if (state === 'encoding') {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#7c2d12', '#831843', '#2e1065']}
          style={StyleSheet.absoluteFill}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        />
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.centerContent}>
          <Text style={styles.title}>Encoding Watermark</Text>
          <Text style={styles.subtitle}>{statusText || 'Processing...'}</Text>

          <View style={styles.progressContainer}>
            <View style={[styles.progressBar, { width: `${progress}%` }]} />
          </View>
          <Text style={styles.progressText}>{progress}%</Text>

          {renderBitMatrix()}
        </View>
      </View>
    );
  }

  // COMPLETE STATE
  if (state === 'complete') {
    return (
      <View style={[styles.container, styles.completeContainer]}>
        <LinearGradient
          colors={['#7c2d12', '#831843', '#2e1065']}
          style={StyleSheet.absoluteFill}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        />
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.centerContent}>
          <View style={styles.successCircle}>
            <Text style={styles.successIcon}>✓</Text>
          </View>
          <Text style={styles.successTitle}>Encoding Complete</Text>
          <Text style={styles.subtitle}>Watermarked audio file is ready</Text>

          <View style={[styles.methodBadge, usedOnnx ? styles.onnxBadge : styles.fallbackBadge]}>
            <Text style={styles.methodText}>
              {usedOnnx ? 'ONNX Model' : 'Fallback'}
            </Text>
          </View>

          {renderBitMatrix()}

          <TouchableOpacity onPress={handleDownload} style={styles.downloadButton}>
            <Text style={styles.downloadText}>Download WAV</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity onPress={handleReset} style={styles.resetButton}>
          <Text style={styles.resetText}>Record Another</Text>
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
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabledButton: {
    backgroundColor: '#374151',
    opacity: 0.5,
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
  inputWaveBar: {
    backgroundColor: '#60a5fa',
  },
  outputWaveBar: {
    backgroundColor: '#a78bfa',
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
  currentBit: {
    borderWidth: 2,
    borderColor: '#fff',
    transform: [{ scale: 1.3 }],
  },
  waveformLabel: {
    color: '#d1d5db',
    fontSize: 12,
    marginTop: 12,
    marginBottom: 4,
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
