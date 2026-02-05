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
 * Ported 1:1 from frontend/utils/StreamingEncoderWrapper.js
 */

const FRAME_SAMPLES = 320;       // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload

// Mini-batch configuration for improved quality
const BATCH_FRAMES = 8;          // 8 frames = 320ms per batch
const HISTORY_FRAMES = 4;        // 4 frames of history = 160ms context

const BATCH_SAMPLES = BATCH_FRAMES * FRAME_SAMPLES;      // 2,560 samples
const HISTORY_SAMPLES = HISTORY_FRAMES * FRAME_SAMPLES;  // 1,280 samples
const TOTAL_SAMPLES = HISTORY_SAMPLES + BATCH_SAMPLES;   // 3,840 samples

export class StreamingEncoderWrapper {
  /**
   * @param {Object} session - Pre-loaded ONNX encoder session
   * @param {Function} TensorConstructor - The Tensor class from onnxruntime-react-native
   */
  constructor(session, TensorConstructor) {
    if (!session) {
      throw new Error('StreamingEncoderWrapper: session is required');
    }
    if (!TensorConstructor) {
      throw new Error('StreamingEncoderWrapper: TensorConstructor is required');
    }
    this._session = session;
    this._Tensor = TensorConstructor;
    this.reset();
  }

  /**
   * Reset all internal state. Call before starting a new stream.
   */
  reset() {
    this._historyBuffer = new Float32Array(HISTORY_SAMPLES);  // Zeros = silence
    this._globalFrameIndex = 0;

    // Frame accumulation buffer
    this._pendingFrames = new Float32Array(BATCH_SAMPLES);
    this._pendingCount = 0;  // Number of frames currently buffered

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
    if (!chunk || chunk.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${chunk ? chunk.length : 'null'}`);
    }
    if (!message || message.length !== PAYLOAD_LENGTH) {
      throw new Error(`Expected ${PAYLOAD_LENGTH} message bits, got ${message ? message.length : 'null'}`);
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
    const firstNewFrameGlobalIndex = this._globalFrameIndex;
    const rotationOffset = ((firstNewFrameGlobalIndex - HISTORY_FRAMES) % PAYLOAD_LENGTH
                            + PAYLOAD_LENGTH) % PAYLOAD_LENGTH;
    const rotatedMessage = new Float32Array(PAYLOAD_LENGTH);
    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
      rotatedMessage[i] = message[(i + rotationOffset) % PAYLOAD_LENGTH];
    }

    // 3. Create tensors using injected Tensor class
    const audioTensor = new this._Tensor('float32', inputAudio, [1, 1, totalInputSamples]);
    const messageTensor = new this._Tensor('float32', rotatedMessage, [1, PAYLOAD_LENGTH]);

    // 4. Run ONNX inference
    const inputNames = this._session.inputNames;
    const feeds = {};
    feeds[inputNames[0]] = audioTensor;
    feeds[inputNames[1]] = messageTensor;

    const results = await this._session.run(feeds);

    // 5. Extract watermarked frames from output (skip history portion)
    const outputName = this._session.outputNames[0];
    const resultData = new Float32Array(results[outputName].data);

    for (let i = 0; i < framesToProcess; i++) {
      const srcStart = HISTORY_SAMPLES + (i * FRAME_SAMPLES);
      const watermarkedFrame = resultData.slice(srcStart, srcStart + FRAME_SAMPLES);
      this._outputQueue.push(watermarkedFrame);
    }

    // 6. Update history with last HISTORY_FRAMES of RAW audio
    if (framesToProcess >= HISTORY_FRAMES) {
      const startFrame = framesToProcess - HISTORY_FRAMES;
      const startSample = startFrame * FRAME_SAMPLES;
      this._historyBuffer.set(this._pendingFrames.subarray(startSample, startSample + HISTORY_SAMPLES));
    } else {
      const samplesToKeep = (HISTORY_FRAMES - framesToProcess) * FRAME_SAMPLES;
      const shiftedHistory = this._historyBuffer.subarray(HISTORY_SAMPLES - samplesToKeep);
      this._historyBuffer.set(shiftedHistory, 0);
      this._historyBuffer.set(this._pendingFrames.subarray(0, samplesToProcess), samplesToKeep);
    }

    // 7. Advance global state
    this._globalFrameIndex += framesToProcess;

    // 8. Clear pending buffer
    this._pendingCount = 0;
  }

  /**
   * Process multiple frames sequentially.
   */
  async processChunk(audio, message) {
    if (audio.length % FRAME_SAMPLES !== 0) {
      throw new Error(`Audio length ${audio.length} not a multiple of ${FRAME_SAMPLES}`);
    }

    const numFrames = audio.length / FRAME_SAMPLES;
    const outputFrames = [];

    for (let i = 0; i < numFrames; i++) {
      const frameStart = i * FRAME_SAMPLES;
      const frame = audio.slice(frameStart, frameStart + FRAME_SAMPLES);
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

  get frameIndex() {
    return this._globalFrameIndex;
  }

  get pendingFrames() {
    return this._pendingCount;
  }

  get outputQueueSize() {
    return this._outputQueue.length;
  }

  get isWarmedUp() {
    return this._globalFrameIndex >= HISTORY_FRAMES;
  }
}

export { FRAME_SAMPLES, PAYLOAD_LENGTH, HISTORY_FRAMES, BATCH_FRAMES, TOTAL_SAMPLES };
