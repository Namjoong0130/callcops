/**
 * AudioResampler.js
 * 
 * Provides proper anti-aliasing downsampling and resampling for React Native.
 * Replaces naive nearest-neighbor / sample skipping that causes accuracy issues.
 */

/**
 * Resamples audio data using Linear Interpolation.
 * Good balance between performance and quality for file processing.
 * 
 * @param {Float32Array} audioData - Source audio samples
 * @param {number} originalSampleRate - Source rate (e.g. 44100)
 * @param {number} targetSampleRate - Target rate (e.g. 8000)
 * @returns {Float32Array} - Resampled audio
 */
export function resampleLinear(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) return audioData;

    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.floor(audioData.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const pos = i * ratio;
        const idx = Math.floor(pos);
        const frac = pos - idx;

        // Linear interpolation: (1-t)*y0 + t*y1
        const y0 = audioData[idx] || 0;
        const y1 = audioData[idx + 1] || y0; // Clamp edge

        resampled[i] = y0 * (1 - frac) + y1 * frac;
    }

    return resampled;
}

/**
 * Stateful Block Averaging Resampler for Real-time Streams
 * 
 * Downsamples by averaging N samples (approx 5.5 for 44.1->8k).
 * This acts as a rudimentary Low-Pass Filter, preventing severe aliasing 
 * that happens with simple sample skipping.
 */
export class StreamingResampler {
    constructor(sourceRate, targetRate) {
        this.ratio = sourceRate / targetRate;
        this.accum = 0;
        this.count = 0;
        this.position = 0; // Fractional position in source measure
    }

    /**
     * Process a chunk of Int16 samples (from LiveAudioStream)
     * @param {Buffer} buffer - Raw PCM buffer (Int16LE)
     * @returns {number[]} - Array of Float32 samples at target rate
     */
    processInt16Buffer(buffer) {
        const output = [];

        // Convert buffer to samples first for simplicity
        const inputSamples = [];
        for (let i = 0; i < buffer.length; i += 2) {
            if (i + 1 < buffer.length) {
                // Read Int16 and convert to float [-1, 1]
                const val = buffer.readInt16LE(i) / 32768.0;
                inputSamples.push(val);
            }
        }

        // Resample with averaging (Boxcar filter)
        // We accumulate samples until we cross the ratio threshold
        for (const sample of inputSamples) {
            this.accum += sample;
            this.count++;
            this.position += 1; // Advanced by 1 source sample

            // Whenever we cross the ratio boundary, output a sample
            // Note: This is an approximation. Ideally we'd weight the fractional parts.
            // But for 44.1->8 (ratio ~5.5), averaging ~5 samples is better than skipping 4.
            if (this.position >= this.ratio) {
                const avg = this.accum / this.count;
                output.push(avg);

                // Reset for next bin
                this.accum = 0;
                this.count = 0;
                this.position -= this.ratio;
            }
        }

        return output;
    }

    /**
     * Reset internal state
     */
    reset() {
        this.accum = 0;
        this.count = 0;
        this.position = 0;
    }
}
