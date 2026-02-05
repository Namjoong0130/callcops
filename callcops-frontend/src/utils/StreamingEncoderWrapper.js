/**
 * StreamingEncoderWrapper (Causal Model) - Mini-Batch Version
 *
 * Stateful wrapper around the ONNX causal encoder session for streaming
 * watermark embedding with mini-batch processing for improved quality.
 *
 * KEY IMPROVEMENTS over single-frame processing:
 * 1. Mini-batch: Processes 8 frames (320ms) at once → 8x fewer ONNX calls
 * 2. Extended history: 4 frames (160ms) of past context → better quality
 * 3. Reduced latency overhead: ~31 ONNX calls/10s vs ~250 calls/10s
 *
 * The causal encoder model uses left-only padding (causal Conv1d), so each
 * output sample depends only on past context. We maintain HISTORY_FRAMES of
 * past raw audio as context. On each batch, we feed [history + batch_frames]
 * to the encoder, then extract only the last BATCH_SAMPLES from the output.
 *
 * This guarantees:
 * 1. Extended context is always available (4 history frames = 160ms,
 *    well beyond the ~78-sample causal receptive field).
 * 2. The global frame index persists across calls, so the cyclic bit
 *    assignment (frame_index % 128) is correct across the full stream.
 * 3. Message rotation compensates for the model's internal 0-based
 *    frame_indices = torch.arange(num_frames).
 *
 * BUFFERING BEHAVIOR:
 * - Frames are accumulated in an internal buffer
 * - When buffer reaches BATCH_FRAMES, a batch inference runs
 * - flush() forces processing of any remaining buffered frames
 *
 * Performance: All intermediate buffers are pre-allocated in reset()
 * to avoid GC pressure during real-time streaming.
 */

import * as ort from 'onnxruntime-web';

const FRAME_SAMPLES = 320;       // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload

// Mini-batch configuration for improved quality
// Larger batches = better quality but more latency
const BATCH_FRAMES = 32;         // 32 frames = 1.28s per batch (~8 ONNX calls/10s)
const HISTORY_FRAMES = 8;        // 8 frames of history = 320ms context

const BATCH_SAMPLES = BATCH_FRAMES * FRAME_SAMPLES;      // 10,240 samples
const HISTORY_SAMPLES = HISTORY_FRAMES * FRAME_SAMPLES;  // 2,560 samples
const TOTAL_SAMPLES = HISTORY_SAMPLES + BATCH_SAMPLES;   // 12,800 samples

export class StreamingEncoderWrapper {
  /**
   * @param {ort.InferenceSession} session - Pre-loaded ONNX causal encoder session
   */
  constructor(session) {
    this._session = session;
    this.reset();
  }

  /**
   * Reset all internal state. Call before starting a new stream.
   * Pre-allocates all buffers to avoid GC during streaming.
   */
  reset() {
    this._historyBuffer = new Float32Array(HISTORY_SAMPLES);  // Zeros = silence
    this._globalFrameIndex = 0;

    // Frame accumulation buffer
    this._pendingFrames = new Float32Array(BATCH_SAMPLES);
    this._pendingCount = 0;  // Number of frames currently buffered

    // Pre-allocated buffers for batch inference (reused — no GC pressure)
    this._inputAudio = new Float32Array(TOTAL_SAMPLES);
    this._rotatedMessage = new Float32Array(PAYLOAD_LENGTH);

    // Output queue for processed frames
    this._outputQueue = [];
  }

  /**
   * Process exactly one frame (320 samples) of raw audio.
   * Frames are buffered internally; actual ONNX inference runs when
   * BATCH_FRAMES accumulate. Returns watermarked audio if available.
   *
   * @param {Float32Array} chunk - Exactly FRAME_SAMPLES (320) raw audio samples
   * @param {Float32Array} message - 128-element message array (0/1 floats)
   * @returns {Promise<Float32Array|null>} - Watermarked 320 samples, or null if buffering
   */
  async processFrame(chunk, message) {
    if (chunk.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${chunk.length}`);
    }

    // Add frame to pending buffer
    const offset = this._pendingCount * FRAME_SAMPLES;
    this._pendingFrames.set(chunk, offset);
    this._pendingCount++;

    // If we have a full batch, run inference
    if (this._pendingCount >= BATCH_FRAMES) {
      await this._runBatchInference(message);
    }

    // Return next frame from output queue if available
    if (this._outputQueue.length > 0) {
      return this._outputQueue.shift();
    }
    return null;
  }

  /**
   * Process exactly one frame and always return watermarked output immediately.
   * This method runs inference immediately if the output queue is empty,
   * providing lower latency but more ONNX calls.
   *
   * @param {Float32Array} chunk - Exactly FRAME_SAMPLES (320) raw audio samples
   * @param {Float32Array} message - 128-element message array (0/1 floats)
   * @returns {Promise<Float32Array>} - Watermarked 320 samples (always returns)
   */
  async processFrameImmediate(chunk, message) {
    if (chunk.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${chunk.length}`);
    }

    // Add frame to pending buffer
    const offset = this._pendingCount * FRAME_SAMPLES;
    this._pendingFrames.set(chunk, offset);
    this._pendingCount++;

    // Run inference if batch is full OR if output queue is empty
    if (this._pendingCount >= BATCH_FRAMES || this._outputQueue.length === 0) {
      await this._runBatchInference(message);
    }

    // Return next frame from output queue
    return this._outputQueue.shift();
  }

  /**
   * Flush any remaining buffered frames. Call at the end of a stream
   * to process partial batches.
   *
   * @param {Float32Array} message - 128-element message array
   * @returns {Promise<Float32Array[]>} - Array of remaining watermarked frames
   */
  async flush(message) {
    if (this._pendingCount > 0) {
      await this._runBatchInference(message);
    }

    const remaining = [...this._outputQueue];
    this._outputQueue = [];
    return remaining;
  }

  /**
   * Internal: Run batch inference on accumulated frames.
   * Processes all pending frames (1 to BATCH_FRAMES).
   */
  async _runBatchInference(message) {
    if (this._pendingCount === 0) return;

    const framesToProcess = this._pendingCount;
    const samplesToProcess = framesToProcess * FRAME_SAMPLES;
    const totalInputSamples = HISTORY_SAMPLES + samplesToProcess;

    // 1. Build input buffer: [history | pending_frames]
    const inputAudio = new Float32Array(totalInputSamples);
    inputAudio.set(this._historyBuffer, 0);
    inputAudio.set(this._pendingFrames.subarray(0, samplesToProcess), HISTORY_SAMPLES);

    // 2. Compute message rotation
    //    Model internally uses frame_indices = [0, 1, 2, ..., numFrames-1]
    //    We want the first NEW frame (after history) to correspond to globalFrameIndex
    //    Internal frame index for first new frame = HISTORY_FRAMES
    //    We need: rotated[HISTORY_FRAMES] = original[globalFrameIndex % 128]
    //    So: offset = (globalFrameIndex - HISTORY_FRAMES) % 128
    const firstNewFrameGlobalIndex = this._globalFrameIndex;
    const rotationOffset = ((firstNewFrameGlobalIndex - HISTORY_FRAMES) % PAYLOAD_LENGTH
                            + PAYLOAD_LENGTH) % PAYLOAD_LENGTH;
    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
      this._rotatedMessage[i] = message[(i + rotationOffset) % PAYLOAD_LENGTH];
    }

    // 3. Run ONNX inference
    const audioTensor = new ort.Tensor('float32', inputAudio, [1, 1, totalInputSamples]);
    const messageTensor = new ort.Tensor('float32', this._rotatedMessage, [1, PAYLOAD_LENGTH]);

    const results = await this._session.run({
      audio: audioTensor,
      message: messageTensor,
    });

    // 4. Extract watermarked frames from output (skip history portion)
    const resultData = results.watermarked.data;
    for (let i = 0; i < framesToProcess; i++) {
      const srcStart = HISTORY_SAMPLES + (i * FRAME_SAMPLES);
      const watermarkedFrame = new Float32Array(FRAME_SAMPLES);
      for (let j = 0; j < FRAME_SAMPLES; j++) {
        watermarkedFrame[j] = resultData[srcStart + j];
      }
      this._outputQueue.push(watermarkedFrame);
    }

    // 5. Update history with last HISTORY_FRAMES of RAW audio
    //    Shift history and add newly processed raw frames
    if (framesToProcess >= HISTORY_FRAMES) {
      // Take last HISTORY_FRAMES from pending buffer
      const startFrame = framesToProcess - HISTORY_FRAMES;
      const startSample = startFrame * FRAME_SAMPLES;
      this._historyBuffer.set(this._pendingFrames.subarray(startSample, startSample + HISTORY_SAMPLES));
    } else {
      // Shift existing history and append new frames
      const samplesToKeep = (HISTORY_FRAMES - framesToProcess) * FRAME_SAMPLES;
      const shiftedHistory = this._historyBuffer.subarray(HISTORY_SAMPLES - samplesToKeep);
      this._historyBuffer.set(shiftedHistory, 0);
      this._historyBuffer.set(this._pendingFrames.subarray(0, samplesToProcess), samplesToKeep);
    }

    // 6. Advance global state
    this._globalFrameIndex += framesToProcess;

    // 7. Clear pending buffer
    this._pendingCount = 0;
  }

  /**
   * Process multiple frames sequentially. Convenience method for
   * chunks that are multiples of FRAME_SAMPLES.
   *
   * @param {Float32Array} audio - Raw audio, length must be multiple of FRAME_SAMPLES
   * @param {Float32Array} message - 128-element message array
   * @returns {Promise<Float32Array>} - Watermarked audio, same length as input
   */
  async processChunk(audio, message) {
    if (audio.length % FRAME_SAMPLES !== 0) {
      throw new Error(`Audio length ${audio.length} not a multiple of ${FRAME_SAMPLES}`);
    }

    const numFrames = audio.length / FRAME_SAMPLES;
    const outputFrames = [];

    // Process all frames
    for (let i = 0; i < numFrames; i++) {
      const frameStart = i * FRAME_SAMPLES;
      const frame = audio.subarray(frameStart, frameStart + FRAME_SAMPLES);
      const result = await this.processFrame(frame, message);
      if (result) {
        outputFrames.push(result);
      }
    }

    // Flush remaining
    const flushed = await this.flush(message);
    outputFrames.push(...flushed);

    // Concatenate all output frames
    const output = new Float32Array(audio.length);
    for (let i = 0; i < outputFrames.length; i++) {
      output.set(outputFrames[i], i * FRAME_SAMPLES);
    }

    return output;
  }

  /** Current global frame index (read-only for diagnostics) */
  get frameIndex() {
    return this._globalFrameIndex;
  }

  /** Number of frames currently buffered, waiting for batch */
  get pendingFrames() {
    return this._pendingCount;
  }

  /** Number of processed frames ready to be returned */
  get outputQueueSize() {
    return this._outputQueue.length;
  }

  /** Whether the history buffer is fully warmed up */
  get isWarmedUp() {
    return this._globalFrameIndex >= HISTORY_FRAMES;
  }
}

export { FRAME_SAMPLES, PAYLOAD_LENGTH, HISTORY_FRAMES, BATCH_FRAMES, TOTAL_SAMPLES };
