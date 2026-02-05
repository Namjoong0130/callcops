# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CallCops

CallCops is a real-time audio watermarking system for Korean telephony authentication and voice phishing prevention. It embeds imperceptible 128-bit watermarks into 8kHz call audio, enabling verification of call authenticity at any point during a conversation.

## Project Structure

```
callcops/
├── callcops-model/          # PyTorch model training & inference
│   ├── models/              # Neural network architectures
│   │   ├── rtaw_net.py          # Non-causal encoder/decoder (symmetric padding)
│   │   ├── rtaw_net_causal.py   # Causal encoder/decoder (zero look-ahead)
│   │   ├── codec_simulator.py   # Differentiable G.711/G.729/AMR-NB
│   │   ├── losses.py            # Combined loss functions
│   │   ├── streaming.py         # Python StreamingEncoderWrapper
│   │   └── attention.py         # Psychoacoustic masking attention
│   ├── configs/             # Training configurations
│   │   ├── default.yaml         # Standard training config
│   │   ├── finetune.yaml        # Fine-tuning config
│   │   ├── causal.yaml          # Causal model config
│   │   └── rescue_snr.yaml      # SNR rescue config (audio-priority)
│   ├── scripts/             # Training & export scripts
│   │   ├── train.py             # Main training loop
│   │   ├── train_causal.py      # Causal model training
│   │   ├── evaluate.py          # Evaluation metrics
│   │   ├── export_onnx.py       # ONNX export + INT8 quantization
│   │   ├── export_mobile.py     # Mobile optimization
│   │   ├── merge_onnx.py        # ONNX model merging
│   │   └── dataset.py           # Data loading + augmentation
│   ├── utils/               # Utilities
│   │   ├── audio_utils.py       # Audio processing
│   │   ├── metrics.py           # SNR, BER computation
│   │   └── messenger.py         # Telegram notifications
│   ├── checkpoints/         # Saved model checkpoints
│   └── exported/            # ONNX model exports
│
├── callcops-frontend/       # Web frontend (React + Vite)
│   ├── src/
│   │   ├── App.jsx              # Main app (embed/detect modes)
│   │   ├── components/
│   │   │   ├── EmbedPanel.jsx           # Watermark embedding UI
│   │   │   ├── RealtimeEmbedDemo.jsx    # Live mic streaming (40ms frames)
│   │   │   ├── RealtimeOscilloscope.jsx # Waveform + FFT spectrogram
│   │   │   ├── AudioUploader.jsx        # File upload
│   │   │   ├── WaveformView.jsx         # WaveSurfer.js integration
│   │   │   ├── BitMatrixView.jsx        # 128-bit visualization
│   │   │   ├── MetricsPanel.jsx         # SNR/BER/confidence display
│   │   │   ├── ProgressiveDetection.jsx # Animated bit reveal
│   │   │   ├── CRCVerificationPanel.jsx # CRC-8 verification
│   │   │   ├── MessageComparison.jsx    # Original vs decoded bits
│   │   │   ├── AudioComparisonPanel.jsx # SNR/PSNR metrics
│   │   │   ├── CallResultScreen.jsx     # Call result display
│   │   │   ├── IncomingCallScreen.jsx   # Incoming call UI
│   │   │   └── LiveAnalysisScreen.jsx   # Real-time analysis
│   │   ├── hooks/
│   │   │   ├── useInference.js          # ONNX model loading & inference
│   │   │   └── useAudioCapture.js       # Mic recording & file loading
│   │   ├── utils/
│   │   │   ├── StreamingEncoderWrapper.js  # Stateful 40ms streaming
│   │   │   ├── audioProcessor.js        # Resampling, normalization
│   │   │   ├── audioComparison.js       # Quality metrics
│   │   │   └── crc.js                   # CRC-8 implementation
│   │   └── pages/
│   │       └── PhoneSimulator.jsx       # Phone call simulation
│   └── public/models/       # ONNX models for browser
│       ├── encoder_int8.onnx
│       └── decoder_int8.onnx
│
└── callcops-app/            # React Native mobile app
```

## Common Commands

```bash
# ============ Model Training ============

# Install dependencies
pip install -r requirements.txt

# Standard training
python scripts/train.py --config configs/default.yaml --epochs 100

# Causal model training
python scripts/train_causal.py --config configs/causal.yaml --epochs 100

# SNR rescue training (audio-priority weights)
python scripts/train.py --config configs/rescue_snr.yaml --epochs 50

# Resume from checkpoint
python scripts/train.py --resume checkpoints/latest.pth

# Background training
nohup stdbuf -oL python3 scripts/train.py \
    --config configs/default.yaml --epochs 100 \
    > training.log 2>&1 &

# Evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/validation

# ONNX Export (auto-detects causal/non-causal from checkpoint)
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported/onnx \
    --quantize --validate

# Monitor training
tail -f training.log

# TensorBoard
tensorboard --logdir logs/

# ============ Frontend ============

cd callcops-frontend

# Development server
npm run dev

# Production build
npx vite build

# Preview build
npm run preview

# Lint
npm run lint
```

## Architecture

### Core Design: Frame-Wise Cyclic Watermarking

- **Sample Rate**: 8kHz (telephony standard)
- **Frame Size**: 40ms = 320 samples = 1 watermark bit
- **Payload**: 128-bit cyclic message
- **Cycle Length**: 128 frames = 5.12 seconds
- **Bandwidth**: 300-3400Hz (telephony band)

Each 40ms audio frame embeds exactly 1 bit. Frame `i` maps to `message[i % 128]`. After 128 frames (5.12s), the full message repeats. The decoder extracts per-frame bits and aggregates across cycle boundaries.

### Model Architecture — Causal (`rtaw_net_causal.py`)

**Production model** — Zero look-ahead, designed for real-time streaming.

- Left-only padding: `pad = (kernel_size - 1) * dilation`
- InstanceNorm1d for streaming stability (no batch statistics)
- Receptive field: ~78 samples (~9.75ms), all past context
- `frame_offset` parameter for correct cyclic alignment in streaming
- Alpha range: [0.35, 1.0]

Note: `rtaw_net.py` (non-causal, symmetric padding) is kept for reference but is NOT used in production.

### Encoder Pipeline

```
Audio [B, 1, T] + Message [B, 128]
        │
        ▼
  Audio Feature Extraction (4 ConvBlock layers)
  channels: [32, 64, 128, 256]
        │
        ▼
  FrameWiseFusionBlock
  - Reshape to T/320 frames
  - Frame i → inject bit message[i % 128]
  - Gated modulation per frame
        │
        ▼
  Residual Refinement (4 ResidualBlocks, dilated)
        │
        ▼
  Audio Decoder (3 ConvBlocks, upsample)
        │
        ▼
  Perturbation = α × tanh(output)
  α ∈ [0.25, 1.0] (learnable, clamped)
        │
        ▼
  Watermarked = Original + Perturbation
```

### Decoder Pipeline

```
Watermarked Audio [B, 1, T]
        │
        ▼
  Feature Extraction + Downsampling
        │
        ▼
  SE Block (channel attention)
        │
        ▼
  FrameWiseBitExtractor
  - 1 bit per 40ms frame → [B, num_frames] logits
        │
        ▼
  128-bit Aggregation
  - Average bits where (frame_idx % 128) matches
        │
        ▼
  Bits [B, 128] (probabilities 0-1)
```

## Streaming Architecture

### Causal Streaming (`StreamingEncoderWrapper`)

The causal model uses left-only padding, so each output sample depends only on past context. The `StreamingEncoderWrapper` maintains 1 frame of history (320 samples, well beyond the ~78-sample causal RF):

```
40ms frame (320 samples) arrives
         │
         ▼
┌─────────────────────────────────────┐
│ StreamingEncoderWrapper             │
│                                     │
│ historyBuffer [1×320 = 320]         │
│ + new frame [320]                   │
│ = 640 samples → ONNX inference     │
│                                     │
│ Message rotation:                   │
│   offset = (globalFrameIndex        │
│             - HISTORY_FRAMES) % 128 │
│                                     │
│ Output: watermarked.slice(-320)     │
│ History: replace with RAW frame     │
│ globalFrameIndex++                  │
└─────────────────────────────────────┘
```

**Why 1 history frame**: 320 samples > 78-sample causal RF. Since the causal model has zero look-ahead, there is no future context issue.

**Why RAW history**: Feeding watermarked history back would cause double-embedding.

**Message rotation**: The ONNX model has `frame_offset=0` baked in (not a dynamic input), so we pre-rotate the 128-bit message to compensate: `offset = (globalFrameIndex - HISTORY_FRAMES) % 128`.

**Implementations**:
- JS: `callcops-frontend/src/utils/StreamingEncoderWrapper.js`
- Python: `callcops-model/models/streaming.py`

Note: `CausalStreamingEncoder` in `rtaw_net_causal.py` is a native PyTorch implementation that uses `frame_offset` directly, without needing message rotation.

## Loss Configuration

### Loss Components

| Loss | Weight Key | Purpose |
|------|-----------|---------|
| L_BCE | `lambda_bit` | Bit accuracy (binary cross-entropy) |
| L_Mel | `lambda_audio` | Perceptual quality (multi-res mel spectrogram) |
| L_STFT | `lambda_stft` | Spectral fidelity |
| L_L1 | `lambda_l1` | Direct waveform SNR |
| L_Adv | `lambda_adv` | GAN adversarial (natural audio) |
| L_Det | `lambda_det` | Watermark presence detection |

### Config Variants

| Config | lambda_bit | lambda_audio | lambda_l1 | lr | Purpose |
|--------|-----------|-------------|----------|-----|---------|
| `default.yaml` | 2.0 | 50.0 | 1.0 | 1e-5 | Balanced training |
| `finetune.yaml` | 2.0 | 50.0 | 30.0 | 5e-5 | Fine-tuning |
| `rescue_snr.yaml` | 0.1 | 350.0 | 500.0 | 1e-6 | SNR emergency rescue |
| `causal.yaml` | 2.0 | 50.0 | 10.0 | 1e-4 | Causal model training |

## Training Features

- **GAN Training**: Alternating encoder/decoder + discriminator updates
- **Mixed Precision**: BF16 AMP for faster training
- **Gradient Clipping**: Default 1.0, rescue mode 0.3
- **Codec Augmentation**: G.711, G.729, AMR-NB simulation during training
- **Data Augmentation**: Noise injection (15-40dB), bandpass filtering
- **Early Stopping**: SNR-based monitoring (configurable threshold)
- **Alpha Override**: Config-driven encoder alpha enforcement surviving checkpoint load
- **Dynamic Weight Controller**: Auto-adjusts lambda_bit/lambda_audio based on BER/SNR
- **Telegram Notifications**: Training progress alerts via `utils/messenger.py`

## Quality Targets

| Metric | Target | Notes |
|--------|--------|-------|
| PESQ | >= 4.0 | Perceptual quality |
| SNR | > 30dB | Signal-to-noise ratio |
| BER | < 5% | Bit error rate |
| BER (G.729) | < 5% | After codec compression |
| Latency | < 200ms | End-to-end |

BCH(511, 128) error correction tolerates up to 10% raw BER.

## Frontend Tech Stack

- **React 19** + **Vite 7** — UI framework and build tool
- **TailwindCSS 4** — Styling
- **ONNX Runtime Web** — Browser-based model inference
- **WaveSurfer.js 7** — Audio waveform visualization
- **Web Audio API** — Microphone capture, resampling, playback

### Real-Time Streaming Flow (RealtimeEmbedDemo)

```
Microphone (48kHz) → ScriptProcessor → inputQueue
        │
        ▼
  Downsample to 8kHz → pendingBuffer
        │
        ▼
  processLoop (sequential):
    While pendingBuffer >= 320 samples:
      wrapper.processFrame(frame320, message128)
      → push to watermarkedChunks
        │
        ▼
  Stop → drain remaining → finalize WAV
```

Queue-based architecture ensures sequential ONNX inference calls (no overlapping).

## Known Issues & Solutions

### 1. Mode Collapse (High SNR, No Watermark)
Model learns zero perturbation. Solution: Clamped alpha [0.25, 1.0] forces minimum perturbation.

### 2. Destructive Convergence (Low SNR, Good BER)
Model over-optimizes BER by increasing perturbation energy, destroying audio quality. Solution: `rescue_snr.yaml` with aggressive audio-priority weights (lambda_audio=350, lambda_l1=500) and SNR-based early stopping.

### 3. Loss Divergence After Few Epochs
Learning rate too high or loss weight imbalance. Solution: LR scheduler, reduce lambda_l1, gradient clipping.

### 4. WebAudio 8kHz Resampling
Browser operates at 44.1/48kHz. Use OfflineAudioContext for polyphase interpolation down to 8kHz.

## Training History

| Attempt | Config | Result | Notes |
|---------|--------|--------|-------|
| v1 | lambda_bit=50, audio=0.01 | SNR 100dB | Mode collapse |
| v2 | Clamped alpha [0.6, 1.0] | Still high SNR | Not enough perturbation |
| v3 | Curriculum 3-phase | Unstable | Loss oscillation |
| v4 | Frame-wise 40ms/bit | Better BER | Architecture change |
| v5 | lambda_l1=50, audio=10 | Loss diverged | L1 too strong |
| v6 | lambda_l1=30, lr=5e-5 | Diverges epoch 4 | SNR 17dB, BER 5.5% |
| v7 | rescue_snr.yaml | Pending | SNR rescue mode |

## Development Notes

### 2026-01-31 Session

**Web Demo Implementation:**
- RealtimeEmbedDemo — live mic streaming with chunked ONNX inference
- RealtimeOscilloscope — Canvas waveform + FFT spectrogram
- BitMatrixView progressive reveal — syncs with audio playback
- ProgressiveDetection — animated bit scan effect
- CRC-8 verification panel — error detection
- AudioComparisonPanel — SNR/PSNR/correlation metrics

**Model Changes:**
- Added L1 waveform loss for direct SNR optimization
- Enabled codec simulation (G.711, G.729, AMR-NB)
- Adjusted alpha range to [0.01, 0.3]
- Frame-wise architecture (40ms = 1 bit)

### 2026-02-02 Session

**SNR Rescue Mode:**
- Created `configs/rescue_snr.yaml` with aggressive audio-priority weights
- Changed encoder alpha from 0.3 to 0.25 in `rtaw_net.py`
- Added config-driven alpha override in `train.py` (survives checkpoint load)
- Added SNR-based early stopping in `train.py` (threshold: 15dB)
- Added best-SNR model checkpoint saving
- Fixed double-update bug where `best_val_snr` was updated in two places

**Stateful Streaming Encoder (40ms / 25 FPS):**
- Created `StreamingEncoderWrapper.js` — JS streaming wrapper with rolling history buffer
- Created `streaming.py` — Python reference implementation + batch validation
- Modified `useInference.js` — added `createStreamingEncoder` factory
- Refactored `RealtimeEmbedDemo.jsx` — 40ms frame processing (was 160ms chunks)
- Modified `EmbedPanel.jsx` — passes `createStreamingEncoder` prop
- Modified `App.jsx` — passes `inference.createStreamingEncoder` to EmbedPanel
- Modified `export_onnx.py` — added `validate_streaming_shape()` for dynamic input sizes

**Key Findings:**
- ONNX encoder already supports dynamic `audio_length` — no model re-export needed
- Message rotation formula: `offset = (globalFrameIndex - HISTORY_FRAMES) % 128`

### 2026-02-03 Session (User Changes)

**Causal Architecture:**
- Created `rtaw_net_causal.py` — zero look-ahead model with left-only padding
- Created `configs/causal.yaml` — causal training config with InstanceNorm
- Created `train_causal.py` — specialized causal training script
- Updated `__init__.py` — exports CausalCallCopsNet and all causal components
- Updated `export_onnx.py` — auto-detects causal/non-causal from checkpoint

**Streaming Refinements:**
- RealtimeEmbedDemo refactored to queue-based architecture (ScriptProcessor → inputQueue → processLoop)
- Fixed unnecessary message rotation in streaming mode

### 2026-02-05 Session

**Causal-Only Migration:**
- Rewrote `StreamingEncoderWrapper.js` for causal model: HISTORY_FRAMES=1 (was 5), TOTAL_SAMPLES=640 (was 1920)
- Rewrote `streaming.py` for causal model: default history_frames=1, fixed `is_warmed_up` bug
- Updated `RealtimeEmbedDemo.jsx` pipeline label: Encoder (640→320) (was 1920→320)
- Updated CLAUDE.md streaming architecture docs

**Streaming Quality Fixes (earlier in session):**
- Fixed message rotation warmup bug: used constant `HISTORY_FRAMES` instead of variable `_historyFilled`
- Added 4th-order Butterworth anti-aliasing filter (cascaded BiquadFilter @ 4kHz) for downsampling
- Decoupled oscilloscope visualization from ONNX processing (independent rAF loop)

**Note:** ONNX models in `public/models/` need to be re-exported from the causal checkpoint. The ONNX export (`export_onnx.py`) does NOT expose `frame_offset` as a dynamic input — it defaults to 0. The `StreamingEncoderWrapper` compensates via message rotation.

## Team

- **안준영** — Full-stack (Model + Frontend)
- **임남중** — Frontend
