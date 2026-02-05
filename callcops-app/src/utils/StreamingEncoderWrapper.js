/**
 * StreamingEncoderWrapper
 *
 * Stateful wrapper around the ONNX encoder session for frame-by-frame
 * streaming watermark embedding with correct cyclic alignment.
 *
 * The encoder model uses symmetric padding (non-causal Conv1d), so each
 * output sample depends on both past and future context. We maintain a
 * rolling history of HISTORY_FRAMES past raw audio frames. On each call,
 * we feed [history + new_frame] (1920 samples) to the encoder, then
 * extract only the last FRAME_SAMPLES (320) from the output.
 *
 * Ported 1:1 from frontend/utils/StreamingEncoderWrapper.js
 */

const FRAME_SAMPLES = 320;       // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload
const HISTORY_FRAMES = 5;        // 5 frames of past context = 1600 samples
const HISTORY_SAMPLES = HISTORY_FRAMES * FRAME_SAMPLES;  // 1600
const TOTAL_SAMPLES = HISTORY_SAMPLES + FRAME_SAMPLES;   // 1920

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
    this._historyFilled = 0;
  }

  /**
   * Process exactly one frame (320 samples) of raw audio.
   *
   * @param {Float32Array} chunk - Exactly FRAME_SAMPLES (320) raw audio samples
   * @param {Float32Array} message - 128-element message array (0/1 floats)
   * @returns {Promise<Float32Array>} - Watermarked 320 samples
   */
  async processFrame(chunk, message) {
    if (!chunk || chunk.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${chunk ? chunk.length : 'null'}`);
    }
    if (!message || message.length !== PAYLOAD_LENGTH) {
      throw new Error(`Expected ${PAYLOAD_LENGTH} message bits, got ${message ? message.length : 'null'}`);
    }

    // 1. Build input buffer: [history (1600) | new_frame (320)] = 1920 samples
    const inputAudio = new Float32Array(TOTAL_SAMPLES);
    inputAudio.set(this._historyBuffer, 0);
    inputAudio.set(chunk, HISTORY_SAMPLES);

    // 2. Compute message rotation for cyclic alignment
    const rotationOffset = ((this._globalFrameIndex - this._historyFilled) % PAYLOAD_LENGTH
      + PAYLOAD_LENGTH) % PAYLOAD_LENGTH;
    const rotatedMessage = new Float32Array(PAYLOAD_LENGTH);
    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
      rotatedMessage[i] = message[(i + rotationOffset) % PAYLOAD_LENGTH];
    }

    // 3. Create tensors using injected Tensor class
    const audioTensor = new this._Tensor('float32', inputAudio, [1, 1, TOTAL_SAMPLES]);
    const messageTensor = new this._Tensor('float32', rotatedMessage, [1, PAYLOAD_LENGTH]);

    // 4. Run ONNX inference using session's input names
    const inputNames = this._session.inputNames;
    const feeds = {};
    feeds[inputNames[0]] = audioTensor;
    feeds[inputNames[1]] = messageTensor;

    const results = await this._session.run(feeds);

    // 5. Extract output
    const outputName = this._session.outputNames[0];
    const watermarkedFull = new Float32Array(results[outputName].data);

    // 6. Extract only the last FRAME_SAMPLES from output (the new frame)
    const watermarkedFrame = watermarkedFull.slice(
      TOTAL_SAMPLES - FRAME_SAMPLES,
      TOTAL_SAMPLES
    );

    // 7. Update history: shift left by one frame, append new RAW audio
    this._historyBuffer.copyWithin(0, FRAME_SAMPLES);
    this._historyBuffer.set(chunk, HISTORY_SAMPLES - FRAME_SAMPLES);

    // 8. Advance global state
    this._globalFrameIndex++;
    if (this._historyFilled < HISTORY_FRAMES) {
      this._historyFilled++;
    }

    return watermarkedFrame;
  }

  /**
   * Process multiple frames sequentially.
   */
  async processChunk(audio, message) {
    if (audio.length % FRAME_SAMPLES !== 0) {
      throw new Error(`Audio length ${audio.length} not a multiple of ${FRAME_SAMPLES}`);
    }

    const numFrames = audio.length / FRAME_SAMPLES;
    const output = new Float32Array(audio.length);

    for (let i = 0; i < numFrames; i++) {
      const frameStart = i * FRAME_SAMPLES;
      const frame = audio.slice(frameStart, frameStart + FRAME_SAMPLES);
      const watermarkedFrame = await this.processFrame(frame, message);
      output.set(watermarkedFrame, frameStart);
    }

    return output;
  }

  get frameIndex() {
    return this._globalFrameIndex;
  }

  get isWarmedUp() {
    return this._historyFilled >= HISTORY_FRAMES;
  }
}

export { FRAME_SAMPLES, PAYLOAD_LENGTH, HISTORY_FRAMES, TOTAL_SAMPLES };
