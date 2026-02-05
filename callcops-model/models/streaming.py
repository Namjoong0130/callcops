"""
StreamingEncoderWrapper - Python Reference Implementation (Causal Model)
========================================================================

Stateful wrapper for frame-by-frame streaming watermark embedding.
Maintains a minimal rolling history buffer and manages cyclic message rotation.

The causal encoder model uses left-only padding (causal Conv1d), so each
output sample depends only on past context. We maintain 1 frame (320 samples)
of past raw audio as context, which exceeds the ~78-sample causal receptive
field. On each call, we feed [history + new_frame] (640 samples) to the
encoder and extract only the last 320 samples (the new frame's output).

Since the causal model has zero look-ahead, there is no future context
degradation — every output sample is fully determined by past input.

Usage:
    encoder = CausalEncoder(...)
    encoder.load_state_dict(...)
    encoder.eval()

    wrapper = StreamingEncoderWrapper(encoder, history_frames=1)

    for frame in audio_frames:  # Each frame is 320 samples
        watermarked_frame = wrapper.process_frame(frame, message_128)
        output.append(watermarked_frame)
"""

import torch
import numpy as np
from typing import Optional

from .rtaw_net import FRAME_SAMPLES, PAYLOAD_LENGTH


class StreamingEncoderWrapper:
    """
    Stateful wrapper for real-time frame-by-frame encoding (causal model).

    Maintains a rolling history buffer of RAW audio (not watermarked)
    to provide temporal context for the encoder's causal convolutional layers.
    Manages global frame indexing and message rotation for correct
    cyclic bit assignment across session.run() calls.
    """

    def __init__(self, encoder, history_frames: int = 1):
        """
        Args:
            encoder: CallCops CausalEncoder model (must be in eval mode)
            history_frames: Number of past frames to keep as context.
                           1 frame = 320 samples, which exceeds the
                           causal receptive field (~78 samples).
        """
        self.encoder = encoder
        self.encoder.eval()
        self.history_frames = history_frames
        self.history_samples = history_frames * FRAME_SAMPLES
        self.total_samples = self.history_samples + FRAME_SAMPLES
        self.reset()

    def reset(self):
        """Reset all internal state. Call before starting a new stream."""
        device = next(self.encoder.parameters()).device
        self._history = torch.zeros(1, 1, self.history_samples, device=device)
        self._global_frame_index = 0

    @torch.no_grad()
    def process_frame(
        self,
        frame: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Process exactly one frame (320 samples) of raw audio.

        Args:
            frame: [320] or [1, 1, 320] raw audio samples
            message: [128] or [1, 128] message bits (0/1 floats)

        Returns:
            watermarked_frame: [320] float tensor
        """
        device = next(self.encoder.parameters()).device

        # Normalize shapes
        if frame.dim() == 1:
            frame = frame.unsqueeze(0).unsqueeze(0).to(device)
        elif frame.dim() == 2:
            frame = frame.unsqueeze(0).to(device)
        else:
            frame = frame.to(device)

        if message.dim() == 1:
            message = message.unsqueeze(0).to(device)
        else:
            message = message.to(device)

        assert frame.shape[-1] == FRAME_SAMPLES, \
            f"Expected {FRAME_SAMPLES} samples, got {frame.shape[-1]}"

        # 1. Build input: [history | new_frame]
        input_audio = torch.cat([self._history, frame], dim=-1)  # [1, 1, total_samples]

        # 2. Compute rotation offset
        # The ONNX model maps internal frame i → message[i % 128] (frame_offset=0).
        # We extract the LAST frame (internal index = history_frames).
        # We want: rotated[history_frames] = original[global_frame_index % 128]
        # Since rotated[i] = original[(i + offset) % 128]:
        #   (history_frames + offset) % 128 = global_frame_index % 128
        #   offset = (global_frame_index - history_frames) % 128
        offset = ((self._global_frame_index - self.history_frames)
                  % PAYLOAD_LENGTH + PAYLOAD_LENGTH) % PAYLOAD_LENGTH

        # 3. Rotate message
        indices = (torch.arange(PAYLOAD_LENGTH, device=device) + offset) % PAYLOAD_LENGTH
        rotated_message = message[:, indices]

        # 4. Run encoder
        watermarked_full = self.encoder(input_audio, rotated_message)  # [1, 1, total_samples]

        # 5. Extract last frame only
        watermarked_frame = watermarked_full[:, :, -FRAME_SAMPLES:]  # [1, 1, 320]

        # 6. Update history with RAW audio (not watermarked!)
        #    With history_frames=1, the entire buffer is replaced.
        self._history = frame.clone()

        # 7. Advance state
        self._global_frame_index += 1

        return watermarked_frame.squeeze()  # [320]

    @torch.no_grad()
    def process_chunk(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Process multiple frames sequentially. Convenience method.

        Args:
            audio: Raw audio, length must be multiple of FRAME_SAMPLES
            message: [128] or [1, 128] message bits

        Returns:
            watermarked: Same length as input audio
        """
        if audio.dim() == 1:
            T = audio.shape[0]
        else:
            T = audio.shape[-1]

        assert T % FRAME_SAMPLES == 0, \
            f"Length {T} not a multiple of {FRAME_SAMPLES}"

        num_frames = T // FRAME_SAMPLES
        output_frames = []

        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            if audio.dim() == 1:
                frame = audio[start:start + FRAME_SAMPLES]
            else:
                frame = audio[..., start:start + FRAME_SAMPLES]
            wm_frame = self.process_frame(frame, message)
            output_frames.append(wm_frame)

        return torch.cat(output_frames, dim=-1)

    @property
    def frame_index(self) -> int:
        """Current global frame index."""
        return self._global_frame_index

    @property
    def is_warmed_up(self) -> bool:
        """Whether the history buffer is fully populated."""
        return self._global_frame_index >= self.history_frames


def validate_streaming_vs_batch(encoder, num_frames: int = 20, history_frames: int = 1):
    """
    Validate that streaming produces similar output to batch processing.

    For the causal model, streaming and batch outputs should be very close
    after the history buffer warms up (just 1 frame), since the causal model
    uses InstanceNorm (no batch statistics dependency) and left-only padding.

    Args:
        encoder: CallCops CausalEncoder model
        num_frames: Number of frames to test
        history_frames: History buffer size for wrapper (default 1 for causal)

    Returns:
        dict with comparison statistics
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    T = num_frames * FRAME_SAMPLES
    audio = torch.randn(1, 1, T, device=device)
    message = torch.randint(0, 2, (1, PAYLOAD_LENGTH), device=device).float()

    # Batch processing
    with torch.no_grad():
        batch_output = encoder(audio, message)  # [1, 1, T]

    # Streaming processing
    wrapper = StreamingEncoderWrapper(encoder, history_frames=history_frames)
    streaming_frames = []

    for i in range(num_frames):
        start = i * FRAME_SAMPLES
        frame = audio[:, :, start:start + FRAME_SAMPLES]
        wm_frame = wrapper.process_frame(frame, message)
        streaming_frames.append(wm_frame)

    # Compare (skip warmup frames)
    skip = history_frames
    results = []

    print(f"\nStreaming vs Batch Validation ({num_frames} frames, {history_frames} history)")
    print(f"{'Frame':>6} {'MaxDiff':>10} {'MeanDiff':>10} {'Status':>8}")
    print("-" * 40)

    for i in range(skip, num_frames):
        start = i * FRAME_SAMPLES
        batch_frame = batch_output[0, 0, start:start + FRAME_SAMPLES]
        stream_frame = streaming_frames[i]

        diff = (batch_frame - stream_frame).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "PASS" if max_diff < 0.05 else "WARN" if max_diff < 0.1 else "FAIL"
        print(f"{i:6d} {max_diff:10.6f} {mean_diff:10.6f} {status:>8}")

        results.append({
            'frame': i,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'status': status
        })

    num_pass = sum(1 for r in results if r['status'] == 'PASS')
    num_total = len(results)
    overall_max = max(r['max_diff'] for r in results) if results else 0

    print(f"\nOverall: {num_pass}/{num_total} PASS, max_diff={overall_max:.6f}")

    return {
        'results': results,
        'num_pass': num_pass,
        'num_total': num_total,
        'overall_max_diff': overall_max,
        'passed': num_pass == num_total
    }
