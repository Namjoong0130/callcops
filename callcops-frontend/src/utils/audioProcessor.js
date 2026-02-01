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

export const SAMPLE_RATE = TARGET_SAMPLE_RATE;
export const WINDOW_SIZE = WINDOW_SAMPLES;
