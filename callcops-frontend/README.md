# CallCops Preview

Real-time audio watermarking demo using ONNX Runtime Web. This is a serverless, on-device inference demo that runs entirely in the browser.

## Features

- 🎤 **Microphone Capture**: Real-time audio recording with 8kHz resampling
- 📁 **File Upload**: Support for WAV, MP3, OGG, M4A, FLAC formats
- 🔊 **Waveform Visualization**: Live audio waveform with bit zone overlay
- 🔢 **Bit Matrix**: 8×16 grid showing 128-bit watermark detection
- 📊 **Metrics Panel**: Confidence scores and detection statistics
- ⚡ **On-Device Inference**: ONNX Runtime Web (WASM backend)

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Copy ONNX Models

Copy the exported ONNX models from the CallCops training project:

```bash
# From the parent directory
cp ../call/exported/onnx/decoder_int8.onnx public/models/
cp ../call/exported/onnx/encoder_int8.onnx public/models/
```

If models haven't been exported yet, run:

```bash
cd ../call
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported/onnx \
    --quantize --validate
```

### 3. Run Development Server

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

## Production Build

```bash
npm run build
```

The production files will be in the `dist/` folder.

## GitHub Pages Deployment

1. Add the following to `vite.config.js`:

```javascript
export default defineConfig({
  base: '/callcops-preview/',
  // ... other config
})
```

2. Build the project:

```bash
npm run build
```

3. Deploy to GitHub Pages:

```bash
# Using gh-pages package
npm install -D gh-pages
npx gh-pages -d dist
```

Or manually push the `dist` folder to a `gh-pages` branch.

## Technical Details

| Property | Value |
|----------|-------|
| Sample Rate | 8kHz (telephony) |
| Window Size | 400ms (3200 samples) |
| Payload | 128 bits |
| Model | decoder_int8.onnx |
| Backend | ONNX Runtime Web (WASM) |

## Model Input/Output

**Decoder:**
- Input: `audio [1, 1, T]` - 8kHz Float32 audio
- Output: `bit_probs [1, 128]` - Bit probabilities (0~1)

**Encoder (optional):**
- Input: `audio [1, 1, T]` + `message [1, 128]`
- Output: `watermarked [1, 1, T]`

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 78+
- ✅ Safari 14.1+
- ✅ Edge 80+

## License

MIT
