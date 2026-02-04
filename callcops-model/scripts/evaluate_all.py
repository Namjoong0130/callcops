#!/usr/bin/env python3
"""
CallCops: Batch Model Evaluation
=================================
checkpoints/ 하위의 모든 *.pth 파일을 ONNX INT8로 변환 후 평가합니다.

평가 과정 (모델마다):
  1. PyTorch 체크포인트 로드 (causal / non-causal 자동 감지)
  2. Encoder & Decoder → ONNX export → INT8 양자화
  3. Encoder-INT8로 워터마크 삽입
  4. 오디오 품질 지표: SNR, PSNR, Correlation, RMSE
  5. Decoder-INT8로 워터마크 검출
  6. 프레임 확률 → 128비트 집계 → Bit Accuracy 계산

워터마크 비트:
  Sync Pattern  AAAA          (16 bits)
  Timestamp     0F3AD454      (32 bits)
  Auth Data     3E2D73BB8F3DD74F (64 bits)
  CRC           18CF          (16 bits)
  Total:                      128 bits

사용법:
    cd callcops-model
    python scripts/evaluate_all.py

    # PyTorch 직접 추론 (ONNX 변환 생략, 빠름)
    python scripts/evaluate_all.py --pytorch

    # 결과를 CSV로 저장
    python scripts/evaluate_all.py --csv results.csv
"""

import os
import sys
import glob
import argparse
import tempfile
import traceback
import time
import csv
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
import logging

import numpy as np
import torch
from tqdm import tqdm

# ONNX TracerWarning + quantization WARNING 억제
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message=".*TracerWarning.*")
warnings.filterwarnings("ignore", message=".*pre-processing before quantization.*")
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# ============================================================
# Path Setup
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from models import CallCopsNet
from models.rtaw_net_causal import CausalCallCopsNet
from export_onnx import (
    EncoderONNXWrapper,
    DecoderONNXWrapper,
    export_encoder_onnx,
    export_decoder_onnx,
    quantize_onnx_dynamic,
)

# ============================================================
# Constants
# ============================================================

FRAME_SAMPLES = 320       # 40ms @ 8kHz
PAYLOAD_LENGTH = 128      # 128-bit cyclic payload
SAMPLE_RATE = 8000

# Watermark: Sync(AAAA) + Timestamp(0F3AD454) + Auth(3E2D73BB8F3DD74F) + CRC(18CF)
WATERMARK_HEX = "AAAA" + "0F3AD454" + "3E2D73BB8F3DD74F" + "18CF"


# ============================================================
# 1. Watermark Message
# ============================================================

def hex_to_bits(hex_str: str) -> np.ndarray:
    """16진수 문자열 → 128-bit float32 배열"""
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return np.array(bits, dtype=np.float32)


def bits_to_hex(bits: np.ndarray) -> str:
    """비트 배열 → 16진수 문자열"""
    hex_str = ""
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        val = int(nibble[0]) * 8 + int(nibble[1]) * 4 + int(nibble[2]) * 2 + int(nibble[3])
        hex_str += format(val, 'X')
    return hex_str


def format_watermark_display(bits: np.ndarray) -> str:
    """워터마크 비트를 구조적으로 표시"""
    h = bits_to_hex(bits)
    return f"{h[:4]}-{h[4:12]}-{h[12:28]}-{h[28:32]}"


# ============================================================
# 2. Audio Loading
# ============================================================

def load_audio_8khz(path: str) -> np.ndarray:
    """
    오디오 파일을 8kHz 모노 float32 [-1, 1]로 로드.
    torchaudio → librosa → pydub 순서로 시도.
    """
    path = str(path)

    # torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return waveform.squeeze(0).numpy().astype(np.float32)
    except Exception:
        pass

    # librosa
    try:
        import librosa
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak
        return y.astype(np.float32)
    except Exception:
        pass

    raise ImportError(
        "오디오 로드 실패. torchaudio 또는 librosa를 설치하세요:\n"
        "  pip install torchaudio\n"
        "  pip install librosa"
    )


# ============================================================
# 3. Audio Quality Metrics
# ============================================================

def compute_snr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB)"""
    noise = original - watermarked
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float('inf')
    if signal_power < 1e-20:
        return 0.0
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB), MAX=1.0"""
    mse = np.mean((original - watermarked) ** 2)
    if mse < 1e-20:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def compute_rmse(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Root Mean Square Error"""
    return float(np.sqrt(np.mean((original - watermarked) ** 2)))


def compute_correlation(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Pearson Correlation Coefficient"""
    n = min(len(original), len(watermarked))
    o, w = original[:n], watermarked[:n]
    mean_o, mean_w = np.mean(o), np.mean(w)
    diff_o, diff_w = o - mean_o, w - mean_w
    num = np.sum(diff_o * diff_w)
    denom = np.sqrt(np.sum(diff_o ** 2) * np.sum(diff_w ** 2))
    if denom < 1e-20:
        return 1.0
    return float(num / denom)


def quality_rating(snr: float) -> str:
    """SNR 기반 품질 등급"""
    if snr >= 40:
        return "Excellent"
    elif snr >= 30:
        return "Good"
    elif snr >= 20:
        return "Fair"
    elif snr >= 10:
        return "Poor"
    else:
        return "Bad"


# ============================================================
# 4. Bit Detection & Aggregation
# ============================================================

def aggregate_to_128bits(frame_probs: np.ndarray) -> np.ndarray:
    """
    프레임별 비트 확률을 128비트로 집계 (Cyclic Averaging).
    Frontend의 useInference.js aggregateTo128Bits 동일 로직.
    """
    bits128 = np.zeros(128, dtype=np.float64)
    counts = np.zeros(128, dtype=np.float64)

    for i in range(len(frame_probs)):
        bit_idx = i % 128
        bits128[bit_idx] += float(frame_probs[i])
        counts[bit_idx] += 1

    mask = counts > 0
    bits128[mask] /= counts[mask]
    bits128[~mask] = 0.5  # 데이터 없으면 불확실

    return bits128.astype(np.float64)


def compute_bit_accuracy(aggregated_probs: np.ndarray, original_bits: np.ndarray) -> float:
    """128비트 일치율 계산"""
    pred_bits = (aggregated_probs > 0.5).astype(np.float32)
    matches = np.sum(pred_bits == original_bits)
    return float(matches / len(original_bits))


def compute_confidence(aggregated_probs: np.ndarray) -> float:
    """
    Confidence Score (MetricsPanel.jsx 동일 로직)
    avg(max(prob, 1-prob)) * 100
    """
    decisiveness = np.maximum(aggregated_probs, 1.0 - aggregated_probs)
    return float(np.mean(decisiveness) * 100.0)


def compute_detection_score(aggregated_probs: np.ndarray) -> float:
    """
    Detection Score (MetricsPanel.jsx 동일 로직)
    avg(|prob - 0.5| * 2) * 100
    """
    signal_strength = np.abs(aggregated_probs - 0.5) * 2.0
    return float(np.mean(signal_strength) * 100.0)


# ============================================================
# 5. Model Loading
# ============================================================

def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[torch.nn.Module, Dict]:
    """
    체크포인트에서 모델 로드 (causal / non-causal 자동 감지).

    Returns:
        model: 평가 모드의 CallCopsNet 또는 CausalCallCopsNet
        info: 체크포인트 메타데이터
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})

    is_causal = checkpoint.get('architecture', '') == 'causal'
    message_dim = config.get('watermark', {}).get('payload_length', 128)
    hidden_channels = config.get('model', {}).get('hidden_channels', [32, 64, 128, 256])
    num_residual_blocks = config.get('model', {}).get('num_residual_blocks', 4)

    if is_causal:
        model = CausalCallCopsNet(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            use_discriminator=False,
        )
    else:
        model = CallCopsNet(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            use_discriminator=False,
        )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    info = {
        'architecture': 'causal' if is_causal else 'non-causal',
        'epoch': checkpoint.get('epoch', '?'),
        'best_val_ber': checkpoint.get('best_val_ber', None),
        'best_val_snr': checkpoint.get('best_val_snr', None),
    }

    return model, info


# ============================================================
# 6. Evaluation (ONNX FP32 / INT8)
# ============================================================

def evaluate_onnx(
    model: torch.nn.Module,
    audio: np.ndarray,
    message_bits: np.ndarray,
    tmp_dir: str,
    quantize: bool = False,
) -> Dict:
    """
    ONNX로 변환 후 워터마크 삽입 → 품질 측정 → 워터마크 검출.

    Args:
        quantize: True이면 INT8 양자화 적용, False이면 FP32 그대로 사용.

    Steps:
        1. Export Encoder/Decoder → ONNX (FP32)
        2. (옵션) Dynamic INT8 Quantize
        3. ONNX Runtime으로 Encode (워터마크 삽입)
        4. SNR / PSNR / Correlation / RMSE 계산
        5. ONNX Runtime으로 Decode (워터마크 검출)
        6. 128비트 집계 → Bit Accuracy
    """
    import onnxruntime as ort

    tmp_path = Path(tmp_dir)

    # --- Export (stdout 억제) ---
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # Encoder
        enc_fp32_path = tmp_path / "encoder.onnx"
        export_encoder_onnx(model.encoder, enc_fp32_path, opset_version=16)

        # Decoder
        dec_fp32_path = tmp_path / "decoder.onnx"
        export_decoder_onnx(model.decoder, dec_fp32_path, opset_version=16)

        if quantize:
            enc_final_path = tmp_path / "encoder_int8.onnx"
            quantize_onnx_dynamic(enc_fp32_path, enc_final_path)
            dec_final_path = tmp_path / "decoder_int8.onnx"
            quantize_onnx_dynamic(dec_fp32_path, dec_final_path)
        else:
            enc_final_path = enc_fp32_path
            dec_final_path = dec_fp32_path

    # --- ONNX Runtime Session ---
    so = ort.SessionOptions()
    so.log_severity_level = 3  # 경고 억제
    enc_session = ort.InferenceSession(str(enc_final_path), so, providers=['CPUExecutionProvider'])
    dec_session = ort.InferenceSession(str(dec_final_path), so, providers=['CPUExecutionProvider'])

    # --- 입력 준비 ---
    # 프레임 경계 정렬 (320 배수)
    T = len(audio)
    remainder = T % FRAME_SAMPLES
    if remainder != 0:
        pad_len = FRAME_SAMPLES - remainder
        audio_padded = np.pad(audio, (0, pad_len), mode='constant')
    else:
        audio_padded = audio.copy()
        pad_len = 0

    audio_input = audio_padded.reshape(1, 1, -1).astype(np.float32)
    message_input = message_bits.reshape(1, 128).astype(np.float32)

    # --- Encode (워터마크 삽입) ---
    t0 = time.perf_counter()
    watermarked = enc_session.run(None, {'audio': audio_input, 'message': message_input})[0]
    encode_time = (time.perf_counter() - t0) * 1000  # ms

    # --- 품질 지표 ---
    orig_flat = audio_input.flatten()
    wm_flat = watermarked.flatten()

    snr = compute_snr(orig_flat, wm_flat)
    psnr = compute_psnr(orig_flat, wm_flat)
    rmse = compute_rmse(orig_flat, wm_flat)
    corr = compute_correlation(orig_flat, wm_flat)

    # --- Decode (워터마크 검출) ---
    t0 = time.perf_counter()
    frame_probs = dec_session.run(None, {'audio': watermarked})[0]
    decode_time = (time.perf_counter() - t0) * 1000  # ms

    frame_probs = frame_probs.flatten()
    num_frames = len(frame_probs)

    # --- 128비트 집계 ---
    agg_probs = aggregate_to_128bits(frame_probs)
    bit_accuracy = compute_bit_accuracy(agg_probs, message_bits)
    confidence = compute_confidence(agg_probs)
    detection_score = compute_detection_score(agg_probs)

    # 세부 비트 비교
    pred_bits = (agg_probs > 0.5).astype(np.float32)
    bit_errors = int(np.sum(pred_bits != message_bits))

    # ONNX 파일 크기
    enc_size_mb = enc_final_path.stat().st_size / (1024 * 1024)
    dec_size_mb = dec_final_path.stat().st_size / (1024 * 1024)

    return {
        'snr': snr,
        'psnr': psnr,
        'rmse': rmse,
        'correlation': corr,
        'bit_accuracy': bit_accuracy,
        'bit_errors': bit_errors,
        'confidence': confidence,
        'detection_score': detection_score,
        'num_frames': num_frames,
        'encode_time_ms': encode_time,
        'decode_time_ms': decode_time,
        'enc_size_mb': enc_size_mb,
        'dec_size_mb': dec_size_mb,
    }


# ============================================================
# 7. Evaluation (PyTorch Direct)
# ============================================================

def evaluate_pytorch(
    model: torch.nn.Module,
    audio: np.ndarray,
    message_bits: np.ndarray,
) -> Dict:
    """
    PyTorch 직접 추론으로 평가 (ONNX 변환 생략, 빠름).
    EncoderONNXWrapper / DecoderONNXWrapper 동일 로직.
    """
    # 프레임 경계 정렬
    T = len(audio)
    remainder = T % FRAME_SAMPLES
    if remainder != 0:
        pad_len = FRAME_SAMPLES - remainder
        audio_padded = np.pad(audio, (0, pad_len), mode='constant')
    else:
        audio_padded = audio.copy()

    audio_tensor = torch.from_numpy(audio_padded).float().reshape(1, 1, -1)
    message_tensor = torch.from_numpy(message_bits).float().reshape(1, 128)

    with torch.no_grad():
        # Encode
        t0 = time.perf_counter()
        watermarked = model.encoder(audio_tensor, message_tensor)
        encode_time = (time.perf_counter() - t0) * 1000

        # Decode
        t0 = time.perf_counter()
        logits = model.decoder(watermarked)
        frame_probs = torch.sigmoid(logits)
        decode_time = (time.perf_counter() - t0) * 1000

    # numpy 변환
    orig_np = audio_tensor.numpy().flatten()
    wm_np = watermarked.numpy().flatten()
    fp_np = frame_probs.numpy().flatten()

    snr = compute_snr(orig_np, wm_np)
    psnr = compute_psnr(orig_np, wm_np)
    rmse = compute_rmse(orig_np, wm_np)
    corr = compute_correlation(orig_np, wm_np)

    agg_probs = aggregate_to_128bits(fp_np)
    bit_accuracy = compute_bit_accuracy(agg_probs, message_bits)
    confidence = compute_confidence(agg_probs)
    detection_score = compute_detection_score(agg_probs)

    pred_bits = (agg_probs > 0.5).astype(np.float32)
    bit_errors = int(np.sum(pred_bits != message_bits))

    return {
        'snr': snr,
        'psnr': psnr,
        'rmse': rmse,
        'correlation': corr,
        'bit_accuracy': bit_accuracy,
        'bit_errors': bit_errors,
        'confidence': confidence,
        'detection_score': detection_score,
        'num_frames': len(fp_np),
        'encode_time_ms': encode_time,
        'decode_time_ms': decode_time,
        'enc_size_mb': 0.0,
        'dec_size_mb': 0.0,
    }


# ============================================================
# 8. Result Formatting
# ============================================================

def print_results_table(results: List[Dict], mode: str):
    """결과 테이블 출력 (Bit Accuracy 내림차순 → SNR 내림차순 정렬)"""

    # 정렬: Bit Accuracy (primary, desc) → SNR (secondary, desc)
    results.sort(key=lambda r: (r['bit_accuracy'], r['snr']), reverse=True)

    # 헤더
    header = (
        f"{'Rank':>4} | {'Model':<50} | {'Arch':<10} | {'Epoch':>5} | "
        f"{'SNR(dB)':>8} | {'PSNR(dB)':>9} | {'Corr':>10} | {'RMSE':>10} | "
        f"{'BitAcc':>8} | {'Errors':>6} | {'Conf%':>6} | {'Det%':>5} | {'Quality':<10}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS  (Mode: {mode.upper()})")
    print(f"  Sorted by: Bit Accuracy (desc) → SNR (desc)")
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(results):
        rating = quality_rating(r['snr']) if r['snr'] != float('inf') else "Perfect"
        snr_str = f"{r['snr']:.2f}" if r['snr'] != float('inf') else "inf"
        psnr_str = f"{r['psnr']:.2f}" if r['psnr'] != float('inf') else "inf"

        row = (
            f"{i+1:>4} | {r['path']:<50} | {r['architecture']:<10} | {str(r['epoch']):>5} | "
            f"{snr_str:>8} | {psnr_str:>9} | {r['correlation']:>10.6f} | {r['rmse']:>10.6f} | "
            f"{r['bit_accuracy']:>7.2%} | {r['bit_errors']:>6} | "
            f"{r['confidence']:>5.1f}% | {r['detection_score']:>4.1f}% | {rating:<10}"
        )
        print(row)

    print(sep)

    # Best model 요약
    if results:
        best = results[0]
        print(f"\n{'=' * 70}")
        print(f"  BEST MODEL: {best['path']}")
        print(f"{'=' * 70}")
        print(f"  Architecture  : {best['architecture']}")
        print(f"  Epoch         : {best['epoch']}")
        print(f"  SNR           : {best['snr']:.2f} dB ({quality_rating(best['snr'])})")
        psnr_display = f"{best['psnr']:.2f}" if best['psnr'] != float('inf') else "inf"
        print(f"  PSNR          : {psnr_display} dB")
        print(f"  Correlation   : {best['correlation']:.6f}")
        print(f"  RMSE          : {best['rmse']:.6f}")
        print(f"  Bit Accuracy  : {best['bit_accuracy']:.2%} ({128 - best['bit_errors']}/128 correct)")
        print(f"  Bit Errors    : {best['bit_errors']}/128")
        print(f"  Confidence    : {best['confidence']:.1f}%")
        print(f"  Detection     : {best['detection_score']:.1f}%")
        print(f"  Encode Time   : {best['encode_time_ms']:.1f} ms")
        print(f"  Decode Time   : {best['decode_time_ms']:.1f} ms")
        if best['enc_size_mb'] > 0:
            print(f"  Encoder INT8  : {best['enc_size_mb']:.2f} MB")
            print(f"  Decoder INT8  : {best['dec_size_mb']:.2f} MB")
        print(f"{'=' * 70}")


def save_results_csv(results: List[Dict], csv_path: str):
    """결과를 CSV로 저장"""
    if not results:
        return

    fieldnames = [
        'rank', 'path', 'architecture', 'epoch',
        'snr', 'psnr', 'correlation', 'rmse',
        'bit_accuracy', 'bit_errors', 'confidence', 'detection_score',
        'num_frames', 'encode_time_ms', 'decode_time_ms',
        'enc_size_mb', 'dec_size_mb', 'quality_rating',
    ]

    results.sort(key=lambda r: (r['bit_accuracy'], r['snr']), reverse=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(results):
            row = {
                'rank': i + 1,
                'quality_rating': quality_rating(r['snr']),
                **{k: r[k] for k in r if k in fieldnames},
            }
            writer.writerow(row)

    print(f"\nCSV saved: {csv_path}")


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CallCops: 전체 체크포인트 일괄 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--audio', type=str,
        default=str(ROOT_DIR / '전화연결음 여성 1.mp3'),
        help='평가용 오디오 파일 경로 (기본: 전화연결음 여성 1.mp3)',
    )
    parser.add_argument(
        '--checkpoints', type=str,
        default=str(PROJECT_DIR / 'checkpoints'),
        help='체크포인트 디렉토리 (기본: callcops-model/checkpoints)',
    )
    parser.add_argument(
        '--mode', type=str, default='onnx_fp32',
        choices=['onnx_fp32', 'onnx_int8', 'pytorch'],
        help='추론 모드 (기본: onnx_fp32)',
    )
    parser.add_argument(
        '--pytorch', action='store_true',
        help='--mode pytorch 단축',
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='결과를 CSV 파일로 저장 (예: results.csv)',
    )
    parser.add_argument(
        '--filter', type=str, default=None,
        help='파일명 필터 (예: best_ber 또는 causal)',
    )

    args = parser.parse_args()
    if args.pytorch:
        mode = 'pytorch'
    else:
        mode = args.mode

    # ---- 체크포인트 탐색 ----
    checkpoint_dir = Path(args.checkpoints)
    pth_files = sorted(glob.glob(str(checkpoint_dir / '**' / '*.pth'), recursive=True))

    # 루트 checkpoints/ 직속 파일도 포함
    root_pths = sorted(glob.glob(str(checkpoint_dir / '*.pth')))
    all_paths = list(dict.fromkeys(pth_files + root_pths))  # 중복 제거, 순서 유지

    if args.filter:
        all_paths = [p for p in all_paths if args.filter in os.path.basename(p)]

    if not all_paths:
        print(f"ERROR: {checkpoint_dir}에서 .pth 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"  CallCops Batch Model Evaluation")
    print(f"{'=' * 70}")
    print(f"  Mode            : {mode.upper()}")
    print(f"  Checkpoints     : {len(all_paths)} files")
    print(f"  Checkpoint Dir  : {checkpoint_dir}")
    print(f"  Audio           : {args.audio}")

    # ---- 오디오 로드 ----
    print(f"\nLoading audio...")
    audio = load_audio_8khz(args.audio)
    duration = len(audio) / SAMPLE_RATE
    num_cycles = duration / (PAYLOAD_LENGTH * FRAME_SAMPLES / SAMPLE_RATE)
    print(f"  Samples   : {len(audio):,}")
    print(f"  Duration  : {duration:.2f}s")
    print(f"  Cycles    : {num_cycles:.1f} (128-bit repeats)")

    # ---- 워터마크 메시지 ----
    message_bits = hex_to_bits(WATERMARK_HEX)
    assert len(message_bits) == 128, f"Expected 128 bits, got {len(message_bits)}"
    print(f"\nWatermark   : {format_watermark_display(message_bits)}")
    print(f"  Sync      : AAAA")
    print(f"  Timestamp : 0F3AD454")
    print(f"  Auth Data : 3E2D73BB8F3DD74F")
    print(f"  CRC       : 18CF")

    # ---- 평가 루프 ----
    results = []
    errors = []
    start_time = time.time()

    # 현재까지의 최고 Bit Accuracy 추적
    best_acc_so_far = 0.0

    pbar = tqdm(all_paths, desc="Evaluating", unit="model", dynamic_ncols=True)
    for pth_path in pbar:
        rel_path = os.path.relpath(pth_path, checkpoint_dir)
        # 파일명만 간략히 표시
        short_name = os.path.basename(os.path.dirname(rel_path)) + "/" + os.path.basename(rel_path)
        if short_name.startswith("/"):
            short_name = os.path.basename(rel_path)
        pbar.set_description(f"Eval: {short_name[:40]:<40}")

        try:
            # 모델 로드
            model, info = load_model_from_checkpoint(pth_path)

            # 평가
            if mode == 'pytorch':
                metrics = evaluate_pytorch(model, audio, message_bits)
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    metrics = evaluate_onnx(
                        model, audio, message_bits, tmp_dir,
                        quantize=(mode == 'onnx_int8'),
                    )

            # 결과 저장
            result = {
                'path': rel_path,
                'architecture': info['architecture'],
                'epoch': info['epoch'],
                **metrics,
            }
            results.append(result)

            # 최고 기록 갱신 추적
            if metrics['bit_accuracy'] > best_acc_so_far:
                best_acc_so_far = metrics['bit_accuracy']

            # postfix에 현재 모델 결과 + 최고 기록 표시
            snr_s = f"{metrics['snr']:.1f}" if metrics['snr'] != float('inf') else "inf"
            pbar.set_postfix_str(
                f"BitAcc={metrics['bit_accuracy']:.1%} SNR={snr_s}dB | "
                f"Best={best_acc_so_far:.1%}",
                refresh=True,
            )

            # 메모리 해제
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            err_msg = str(e).split('\n')[0][:60]
            pbar.set_postfix_str(f"ERROR: {err_msg}", refresh=True)
            errors.append({'path': rel_path, 'error': str(e)})
            continue

    pbar.close()
    elapsed = time.time() - start_time

    # ---- 결과 출력 ----
    print(f"\nCompleted: {len(results)} success, {len(errors)} errors")
    print(f"Total time: {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s per model)")

    if errors:
        print(f"\nFailed models:")
        for e in errors:
            print(f"  - {e['path']}: {e['error'][:80]}")

    if results:
        print_results_table(results, mode)

        if args.csv:
            save_results_csv(results, args.csv)

        # 검출된 비트 상세 (Best Model)
        results.sort(key=lambda r: (r['bit_accuracy'], r['snr']), reverse=True)
        best = results[0]
        print(f"\nBest model detected watermark:")
        print(f"  Expected : {format_watermark_display(message_bits)}")

        # 베스트 모델 재실행해서 검출 비트 표시
        try:
            model, _ = load_model_from_checkpoint(
                str(checkpoint_dir / best['path'])
            )

            # 오디오 패딩
            T = len(audio)
            rem = T % FRAME_SAMPLES
            audio_aligned = np.pad(audio, (0, FRAME_SAMPLES - rem)) if rem else audio.copy()

            if mode == 'pytorch':
                at = torch.from_numpy(audio_aligned).float().reshape(1, 1, -1)
                mt = torch.from_numpy(message_bits).float().reshape(1, 128)
                with torch.no_grad():
                    wm = model.encoder(at, mt)
                    lo = model.decoder(wm)
                    fp = torch.sigmoid(lo).numpy().flatten()
                agg = aggregate_to_128bits(fp)
            else:
                import onnxruntime as ort
                with tempfile.TemporaryDirectory() as tmp_dir:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        enc_fp = Path(tmp_dir) / "encoder.onnx"
                        export_encoder_onnx(model.encoder, enc_fp)
                        dec_fp = Path(tmp_dir) / "decoder.onnx"
                        export_decoder_onnx(model.decoder, dec_fp)
                        if mode == 'onnx_int8':
                            enc_final = Path(tmp_dir) / "encoder_int8.onnx"
                            quantize_onnx_dynamic(enc_fp, enc_final)
                            dec_final = Path(tmp_dir) / "decoder_int8.onnx"
                            quantize_onnx_dynamic(dec_fp, dec_final)
                        else:
                            enc_final, dec_final = enc_fp, dec_fp

                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    enc_s = ort.InferenceSession(str(enc_final), so, providers=['CPUExecutionProvider'])
                    dec_s = ort.InferenceSession(str(dec_final), so, providers=['CPUExecutionProvider'])

                    ai = audio_aligned.reshape(1, 1, -1).astype(np.float32)
                    mi = message_bits.reshape(1, 128).astype(np.float32)
                    wm = enc_s.run(None, {'audio': ai, 'message': mi})[0]
                    fp = dec_s.run(None, {'audio': wm})[0].flatten()
                    agg = aggregate_to_128bits(fp)

            pred = (agg > 0.5).astype(np.float32)
            print(f"  Detected : {format_watermark_display(pred)}")

            mismatches = [bi for bi in range(128) if pred[bi] != message_bits[bi]]
            if mismatches:
                print(f"  Mismatched bits ({len(mismatches)}): {mismatches}")
            else:
                print(f"  All 128 bits matched perfectly!")

            del model
        except Exception:
            pass


if __name__ == "__main__":
    main()
