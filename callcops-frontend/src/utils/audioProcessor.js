/**
 * Audio Processor Utility
 * 
 * Critical functions for 8kHz resampling required by CallCops model.
 */

const TARGET_SAMPLE_RATE = 8000;
const WINDOW_MS = 400;
const WINDOW_SAMPLES = (TARGET_SAMPLE_RATE * WINDOW_MS) / 1000; // 3200 samples

/**
 * Resample audio buffer to 8kHz
 * @param {AudioBuffer} audioBuffer - Original audio buffer
 * @returns {Promise<Float32Array>} - Resampled 8kHz audio data
 */
export async function resampleTo8kHz(audioBuffer) {
  const originalSampleRate = audioBuffer.sampleRate;

  // If already 8kHz, just return the data
  if (originalSampleRate === TARGET_SAMPLE_RATE) {
    return audioBuffer.getChannelData(0);
  }

  // Create offline context for resampling
  const offlineCtx = new OfflineAudioContext(
    1, // mono
    Math.ceil(audioBuffer.duration * TARGET_SAMPLE_RATE),
    TARGET_SAMPLE_RATE
  );

  // Create buffer source
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);

  // Render resampled audio
  const resampledBuffer = await offlineCtx.startRendering();
  return resampledBuffer.getChannelData(0);
}

/**
 * Resample Float32Array audio data from source rate to 8kHz
 * @param {Float32Array} audioData - Original audio data
 * @param {number} sourceSampleRate - Source sample rate
 * @returns {Promise<Float32Array>} - Resampled 8kHz audio data
 */
export async function resampleArrayTo8kHz(audioData, sourceSampleRate) {
  if (sourceSampleRate === TARGET_SAMPLE_RATE) {
    return audioData;
  }

  // Create a temporary AudioContext for resampling
  const audioContext = new AudioContext({ sampleRate: sourceSampleRate });
  const audioBuffer = audioContext.createBuffer(1, audioData.length, sourceSampleRate);
  audioBuffer.getChannelData(0).set(audioData);

  const result = await resampleTo8kHz(audioBuffer);
  await audioContext.close();

  return result;
}

/**
 * Create sliding windows from audio data
 * @param {Float32Array} audioData - 8kHz audio data
 * @param {number} windowSize - Window size in samples (default: 3200 for 400ms)
 * @param {number} hopSize - Hop size in samples (default: half window)
 * @returns {Float32Array[]} - Array of audio windows
 */
export function createSlidingWindows(audioData, windowSize = WINDOW_SAMPLES, hopSize = null) {
  if (hopSize === null) {
    hopSize = Math.floor(windowSize / 2);
  }

  const windows = [];

  for (let i = 0; i + windowSize <= audioData.length; i += hopSize) {
    windows.push(audioData.slice(i, i + windowSize));
  }

  // Handle last window (pad with zeros if needed)
  const remaining = audioData.length % hopSize;
  if (remaining > 0 && windows.length > 0) {
    const lastStart = windows.length * hopSize;
    if (lastStart < audioData.length) {
      const lastWindow = new Float32Array(windowSize);
      const lastData = audioData.slice(lastStart);
      lastWindow.set(lastData);
      windows.push(lastWindow);
    }
  }

  return windows;
}

/**
 * Normalize audio to [-1, 1] range
 * @param {Float32Array} audioData - Audio data
 * @returns {Float32Array} - Normalized audio data
 */
export function normalizeAudio(audioData) {
  const maxAbs = Math.max(...audioData.map(Math.abs));
  if (maxAbs === 0) return audioData;

  const normalized = new Float32Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    normalized[i] = audioData[i] / maxAbs;
  }
  return normalized;
}

/**
 * Convert AudioBuffer to mono Float32Array
 * @param {AudioBuffer} buffer - Audio buffer
 * @returns {Float32Array} - Mono audio data
 */
export function toMono(buffer) {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0);
  }

  const mono = new Float32Array(buffer.length);
  const channels = [];

  for (let i = 0; i < buffer.numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  for (let i = 0; i < buffer.length; i++) {
    let sum = 0;
    for (const channel of channels) {
      sum += channel[i];
    }
    mono[i] = sum / buffer.numberOfChannels;
  }

  return mono;
}

/**
 * Parse WAV file directly without browser resampling
 * Handles 16-bit PCM mono files specifically for 8kHz watermarking
 * @param {ArrayBuffer} arrayBuffer - Raw file data
 * @returns {Float32Array|null} - Audio data if compatible, null otherwise
 */
export function parseWav(arrayBuffer) {
  try {
    const view = new DataView(arrayBuffer);

    // Check RIFF header
    if (view.getUint32(0, false) !== 0x52494646) return null; // "RIFF"
    if (view.getUint32(8, false) !== 0x57415645) return null; // "WAVE"

    // Search for "fmt " chunk
    let offset = 12;
    let fmtOffset = -1;
    let dataOffset = -1;
    let dataSize = 0;

    while (offset < view.byteLength) {
      const chunkId = view.getUint32(offset, false);
      const chunkSize = view.getUint32(offset + 4, true);

      if (chunkId === 0x666d7420) { // "fmt "
        fmtOffset = offset + 8;
      } else if (chunkId === 0x64617461) { // "data"
        dataOffset = offset + 8;
        dataSize = chunkSize;
      }

      offset += 8 + chunkSize;
    }

    if (fmtOffset === -1 || dataOffset === -1) return null;

    // Check format
    const audioFormat = view.getUint16(fmtOffset, true);
    const channels = view.getUint16(fmtOffset + 2, true);
    const sampleRate = view.getUint32(fmtOffset + 4, true);
    const bitsPerSample = view.getUint16(fmtOffset + 14, true);

    // Only accept 8kHz mono 16-bit PCM for direct parsing
    // Others should fall back to standard decoding
    if (audioFormat !== 1 || channels !== 1 || sampleRate !== 8000 || bitsPerSample !== 16) {
      console.warn('WAV format mismatch for direct parsing:', {
        audioFormat, channels, sampleRate, bitsPerSample
      });
      return null;
    }

    // Convert 16-bit PCM to Float32
    const samples = dataSize / 2;
    const floatData = new Float32Array(samples);

    for (let i = 0; i < samples; i++) {
      const int16 = view.getInt16(dataOffset + i * 2, true);
      // Use symmetric scaling to match export logic (divide by 32767)
      // This is crucial for bit-exact roundtrip of watermarked signals
      floatData[i] = Math.max(-1, Math.min(1, int16 / 32767.0));
    }

    return floatData;
  } catch (err) {
    console.error('WAV parsing failed:', err);
    return null;
  }
}

export const SAMPLE_RATE = TARGET_SAMPLE_RATE;
export const WINDOW_SIZE = WINDOW_SAMPLES;
