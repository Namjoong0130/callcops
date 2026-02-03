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
 * This guarantees:
 * 1. Every output sample has full receptive-field context (~57 samples
 *    one-sided, well within 5 history frames = 1600 samples).
 * 2. The global frame index persists across calls, so the cyclic bit
 *    assignment (frame_index % 128) is correct across the full stream.
 * 3. Message rotation compensates for the model's internal 0-based
 *    frame_indices = torch.arange(num_frames).
 */

import * as ort from 'onnxruntime-web';

const FRAME_SAMPLES = 320;       // 40ms @ 8kHz
const PAYLOAD_LENGTH = 128;      // 128-bit cyclic payload
const HISTORY_FRAMES = 5;        // 5 frames of past context = 1600 samples
const HISTORY_SAMPLES = HISTORY_FRAMES * FRAME_SAMPLES;  // 1600
const TOTAL_SAMPLES = HISTORY_SAMPLES + FRAME_SAMPLES;   // 1920

export class StreamingEncoderWrapper {
  /**
   * @param {ort.InferenceSession} session - Pre-loaded ONNX encoder session
   */
  constructor(session) {
    this._session = session;
    this.reset();
  }

  /**
   * Reset all internal state. Call before starting a new stream.
   */
  reset() {
    this._historyBuffer = new Float32Array(HISTORY_SAMPLES);  // Zeros = silence
    this._globalFrameIndex = 0;
    this._historyFilled = 0;  // How many history frames have been filled so far
  }

  /**
   * Process exactly one frame (320 samples) of raw audio.
   *
   * @param {Float32Array} chunk - Exactly FRAME_SAMPLES (320) raw audio samples
   * @param {Float32Array} message - 128-element message array (0/1 floats)
   * @returns {Promise<Float32Array>} - Watermarked 320 samples
   */
  async processFrame(chunk, message) {
    if (chunk.length !== FRAME_SAMPLES) {
      throw new Error(`Expected ${FRAME_SAMPLES} samples, got ${chunk.length}`);
    }

    // 1. Build input buffer: [history (1600) | new_frame (320)] = 1920 samples
    const inputAudio = new Float32Array(TOTAL_SAMPLES);
    inputAudio.set(this._historyBuffer, 0);
    inputAudio.set(chunk, HISTORY_SAMPLES);

    // 2. Compute message rotation.
    //
    //    The model internally assigns:
    //      frame_indices = torch.arange(num_frames)  → [0, 1, 2, 3, 4, 5]
    //      bit_indices = frame_indices % 128
    //      frame_bits = message[:, bit_indices]
    //
    //    For our 1920-sample input (6 frames), we want:
    //      - internal frame 0 → global bit (globalFrameIndex - historyFilled) % 128
    //      - internal frame 5 → global bit (globalFrameIndex) % 128
    //
    //    So we rotate the message by offset = (globalFrameIndex - historyFilled) % 128.
    //    The model reads rotated[i] = original[(i + offset) % 128], which maps to
    //    the correct global bit for each frame.
    const rotationOffset = ((this._globalFrameIndex - this._historyFilled) % PAYLOAD_LENGTH
                            + PAYLOAD_LENGTH) % PAYLOAD_LENGTH;
    const rotatedMessage = new Float32Array(PAYLOAD_LENGTH);
    for (let i = 0; i < PAYLOAD_LENGTH; i++) {
      rotatedMessage[i] = message[(i + rotationOffset) % PAYLOAD_LENGTH];
    }

    // 3. Run ONNX inference
    const audioTensor = new ort.Tensor('float32', inputAudio, [1, 1, TOTAL_SAMPLES]);
    const messageTensor = new ort.Tensor('float32', rotatedMessage, [1, PAYLOAD_LENGTH]);

    const results = await this._session.run({
      audio: audioTensor,
      message: messageTensor,
    });

    const watermarkedFull = new Float32Array(results.watermarked.data);

    // 4. Extract only the last FRAME_SAMPLES from output (the new frame)
    const watermarkedFrame = watermarkedFull.slice(
      TOTAL_SAMPLES - FRAME_SAMPLES,
      TOTAL_SAMPLES
    );

    // 5. Update history: shift left by one frame, append new RAW audio.
    //    We store RAW audio in history (not watermarked) because the encoder
    //    expects original audio as input. Feeding watermarked audio would cause
    //    double-embedding artifacts.
    this._historyBuffer.copyWithin(0, FRAME_SAMPLES);
    this._historyBuffer.set(chunk, HISTORY_SAMPLES - FRAME_SAMPLES);

    // 6. Advance global state
    this._globalFrameIndex++;
    if (this._historyFilled < HISTORY_FRAMES) {
      this._historyFilled++;
    }

    return watermarkedFrame;
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
    const output = new Float32Array(audio.length);

    for (let i = 0; i < numFrames; i++) {
      const frameStart = i * FRAME_SAMPLES;
      const frame = audio.slice(frameStart, frameStart + FRAME_SAMPLES);
      const watermarkedFrame = await this.processFrame(frame, message);
      output.set(watermarkedFrame, frameStart);
    }

    return output;
  }

  /** Current global frame index (read-only for diagnostics) */
  get frameIndex() {
    return this._globalFrameIndex;
  }

  /** Whether the history buffer is fully warmed up */
  get isWarmedUp() {
    return this._historyFilled >= HISTORY_FRAMES;
  }
}

export { FRAME_SAMPLES, PAYLOAD_LENGTH, HISTORY_FRAMES, TOTAL_SAMPLES };
