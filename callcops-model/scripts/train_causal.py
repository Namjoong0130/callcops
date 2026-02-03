"""
CallCops: Causal Model Training Script
======================================

True Causal Architecture 모델 학습 스크립트.

변경사항 (vs train.py):
1. CausalCallCopsNet 사용 (rtaw_net_causal.py)
2. 기존 체크포인트와 호환 불가 (새로운 가중치 구조)
3. Streaming validation 추가

Usage:
    python scripts/train_causal.py --epochs 100 --batch_size 64
    python scripts/train_causal.py --config configs/causal.yaml
"""

import os
import sys
import argparse
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# Causal model import
from models.rtaw_net_causal import (
    CausalCallCopsNet,
    CausalStreamingEncoder,
    calculate_receptive_field,
    FRAME_SAMPLES,
    PAYLOAD_LENGTH
)
from models import CallCopsLoss, DifferentiableCodecSimulator
from scripts.dataset import create_train_val_loaders
from utils.messenger import CallCopsMessenger


# =============================================================================
# Utility Functions
# =============================================================================

def compute_snr(original: torch.Tensor, watermarked: torch.Tensor) -> float:
    """Signal-to-Noise Ratio 계산"""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - watermarked) ** 2)

    if noise_power < 1e-10:
        return 100.0

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def compute_ber(pred_logits: torch.Tensor, target_bits: torch.Tensor) -> float:
    """Bit Error Rate 계산"""
    B, num_frames = pred_logits.shape

    if target_bits.shape[1] != num_frames:
        frame_indices = torch.arange(num_frames, device=target_bits.device) % target_bits.shape[1]
        target_bits_expanded = target_bits[:, frame_indices]
    else:
        target_bits_expanded = target_bits

    pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
    errors = (pred_bits != target_bits_expanded).float()
    return errors.mean().item()


def get_frame_target_bits(bits: torch.Tensor, num_frames: int) -> torch.Tensor:
    """128비트를 프레임 수에 맞게 Cyclic 확장"""
    frame_indices = torch.arange(num_frames, device=bits.device) % bits.shape[1]
    return bits[:, frame_indices]


# =============================================================================
# Causal Trainer
# =============================================================================

class CausalTrainer:
    """
    Causal CallCops 모델 트레이너
    ============================

    Non-Causal Trainer와 동일한 구조이지만,
    CausalCallCopsNet을 사용합니다.
    """

    def __init__(
        self,
        model: CausalCallCopsNet,
        loss_fn: CallCopsLoss,
        opt_g: optim.Optimizer,
        opt_d: optim.Optimizer,
        device: torch.device,
        codec_sim: Optional[DifferentiableCodecSimulator] = None,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        messenger: Optional[CallCopsMessenger] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.device = device
        self.codec_sim = codec_sim
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.messenger = messenger

        self.scaler = GradScaler() if use_amp else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_ber = 1.0
        self.best_loss = float('inf')

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ber': [], 'val_ber': []
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """단일 학습 스텝"""
        audio = batch['audio'].to(self.device)
        bits = batch['bits'].to(self.device)

        # ========================================
        # 1. Discriminator Update
        # ========================================
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            with torch.no_grad():
                watermarked, _ = self.model.embed(audio, bits)

            disc_real = self.model.discriminator(audio)
            disc_fake = self.model.discriminator(watermarked.detach())
            d_loss = self.loss_fn.adv_loss.discriminator_loss(disc_real, disc_fake)

        if self.use_amp:
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.grad_clip)
            self.scaler.step(self.opt_d)
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.grad_clip)
            self.opt_d.step()

        # ========================================
        # 2. Generator Update
        # ========================================
        self.opt_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            watermarked, _ = self.model.embed(audio, bits)

            if self.codec_sim is not None:
                watermarked_degraded, codec_used = self.codec_sim(watermarked)
            else:
                watermarked_degraded = watermarked
                codec_used = 'none'

            pred_logits = self.model.decoder(watermarked_degraded)
            num_frames = pred_logits.shape[1]
            target_bits_expanded = get_frame_target_bits(bits, num_frames)

            detection_logits = torch.abs(pred_logits).mean(dim=1, keepdim=True)
            disc_fake = self.model.discriminator(watermarked)

            losses = self.loss_fn(
                pred_audio=watermarked,
                target_audio=audio,
                pred_bits=pred_logits,
                target_bits=target_bits_expanded,
                detection_pred=detection_logits,
                disc_fake=disc_fake
            )

            g_loss = losses['total']

        if self.use_amp:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                self.grad_clip
            )
            self.scaler.step(self.opt_g)
            self.scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                self.grad_clip
            )
            self.opt_g.step()

        # ========================================
        # 3. Metrics
        # ========================================
        with torch.no_grad():
            ber = compute_ber(pred_logits.detach().float(), bits.detach().float())
            snr = compute_snr(audio.detach().float(), watermarked.detach().float())

        self.global_step += 1

        return {
            'loss_total': losses['total'].item(),
            'loss_bit': losses['bit'].item(),
            'loss_mel': losses['mel'].item(),
            'loss_stft': losses['stft'].item(),
            'loss_adv_g': losses['adv_g'].item(),
            'loss_adv_d': d_loss.item(),
            'ber': ber,
            'snr': snr,
            'codec': codec_used
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 루프"""
        self.model.eval()

        total_metrics = {'loss': 0.0, 'ber': 0.0, 'snr': 0.0, 'accuracy': 0.0}
        num_batches = 0

        for batch in val_loader:
            audio = batch['audio'].to(self.device)
            bits = batch['bits'].to(self.device)

            watermarked, _ = self.model.embed(audio, bits)
            pred_logits = self.model.decoder(watermarked)

            num_frames = pred_logits.shape[1]
            target_bits_expanded = get_frame_target_bits(bits, num_frames)
            detection_logits = torch.abs(pred_logits).mean(dim=1, keepdim=True)

            losses = self.loss_fn(
                pred_audio=watermarked,
                target_audio=audio,
                pred_bits=pred_logits,
                target_bits=target_bits_expanded,
                detection_pred=detection_logits
            )

            ber = compute_ber(pred_logits, bits)
            snr = compute_snr(audio, watermarked)

            total_metrics['loss'] += losses['total'].item()
            total_metrics['ber'] += ber
            total_metrics['snr'] += snr
            total_metrics['accuracy'] += (1.0 - ber)
            num_batches += 1

        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)

        self.model.train()
        return total_metrics

    @torch.no_grad()
    def validate_streaming(self) -> Dict[str, float]:
        """
        Causal 스트리밍 검증
        
        히스토리 버퍼 없이 단일 프레임 처리가 가능한지 확인.
        """
        self.model.eval()

        # Create streaming wrapper
        wrapper = CausalStreamingEncoder(self.model.encoder)

        # Generate test data
        num_frames = 20
        T = num_frames * FRAME_SAMPLES
        audio = torch.randn(1, 1, T, device=self.device)
        message = torch.randint(0, 2, (1, PAYLOAD_LENGTH), device=self.device).float()

        # Batch processing
        batch_output = self.model.encoder(audio, message)

        # Streaming processing (frame-by-frame)
        streaming_frames = []
        wrapper.reset()

        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            frame = audio[0, 0, start:start + FRAME_SAMPLES]
            wm_frame = wrapper.process_frame(frame, message[0])
            streaming_frames.append(wm_frame)

        streaming_output = torch.stack(streaming_frames, dim=-1).view(1, 1, -1)

        # Compare
        diff = (batch_output - streaming_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        self.model.train()

        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'passed': max_diff < 1e-5  # Should be exactly equal for causal!
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """에포크 학습"""
        self.model.train()
        self.current_epoch = epoch

        if self.codec_sim is not None:
            self.codec_sim.set_epoch(epoch)

        epoch_metrics = {'loss_total': 0.0, 'loss_bit': 0.0, 'ber': 0.0, 'snr': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]

            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'ber': f"{metrics['ber']:.4f}",
                'snr': f"{metrics['snr']:.1f}dB"
            })

        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    def save_checkpoint(self, path: Path, metrics: Dict[str, float], config: Optional[Dict] = None, is_latest: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'best_ber': self.best_ber,
            'best_loss': self.best_loss,
            'metrics': metrics,
            'history': self.history,
            'architecture': 'causal'  # Mark as causal model
        }

        if config is not None:
            checkpoint['config'] = config

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved causal checkpoint to {path}")

        if is_latest:
            latest_path = path.parent / "latest.pth"
            try:
                if latest_path.exists():
                    latest_path.unlink()
                shutil.copy(path, latest_path)
            except Exception as e:
                print(f"Warning: Failed to create latest.pth: {e}")

    def load_checkpoint(self, path: Path, new_lr: float = None):
        """체크포인트 로드"""
        print(f"Loading causal checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        # Verify architecture
        if checkpoint.get('architecture') != 'causal':
            raise ValueError(
                "ERROR: This checkpoint is from a NON-CAUSAL model!\n"
                "Causal models require training from scratch with train_causal.py"
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])

        if new_lr is not None:
            for param_group in self.opt_g.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.opt_d.param_groups:
                param_group['lr'] = new_lr

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_ber = checkpoint.get('best_ber', 1.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        print(f"Loaded causal checkpoint from epoch {checkpoint['epoch']+1}")


# =============================================================================
# Main Training Function
# =============================================================================

def train_causal(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    save_dir: Path,
    device: torch.device,
    resume_path: Optional[Path] = None
):
    """Causal 모델 학습 함수"""

    messenger = CallCopsMessenger()
    messenger.send_message(
        f"🚀 **Causal CallCops Training Started**\n"
        f"Architecture: TRUE CAUSAL (Zero Look-ahead)\n"
        f"Device: {device}\n"
        f"Save Dir: `{save_dir}`"
    )

    try:
        training_config = config.get('training', {})
        model_config = config.get('model', {})

        # ========================================
        # 1. Causal Model 초기화
        # ========================================
        model = CausalCallCopsNet(
            message_dim=config.get('watermark', {}).get('payload_length', 128),
            hidden_channels=model_config.get('hidden_channels', [32, 64, 128, 256]),
            num_residual_blocks=model_config.get('num_residual_blocks', 4),
            use_discriminator=True
        ).to(device)

        # Print receptive field
        rf = calculate_receptive_field()
        print(f"\n{'='*60}")
        print("CAUSAL Architecture Initialized")
        print(f"{'='*60}")
        print(f"  Receptive Field: {rf['total_samples']} samples = {rf['total_ms']:.2f}ms")
        print(f"  Look-ahead: {rf['look_ahead']} samples (ZERO!)")
        print(f"{'='*60}\n")

        params = model.count_parameters()
        print(f"Model Parameters:")
        print(f"  Encoder: {params['encoder']:,}")
        print(f"  Decoder: {params['decoder']:,}")
        print(f"  Discriminator: {params['discriminator']:,}")
        print(f"  Total: {params['total']:,}")

        # ========================================
        # 2. Loss Function
        # ========================================
        loss_fn = CallCopsLoss(
            lambda_bit=training_config.get('lambda_bit', 10.0),
            lambda_audio=training_config.get('lambda_audio', 10.0),
            lambda_adv=training_config.get('lambda_adv', 0.1),
            lambda_det=training_config.get('lambda_det', 0.5),
            lambda_stft=training_config.get('lambda_stft', 2.0),
            lambda_l1=training_config.get('lambda_l1', 10.0),
            sample_rate=config.get('audio', {}).get('sample_rate', 8000)
        ).to(device)

        # ========================================
        # 3. Optimizers
        # ========================================
        lr = training_config.get('learning_rate', 2e-4)
        betas = tuple(training_config.get('adam_betas', [0.5, 0.9]))

        opt_g = optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=lr, betas=betas
        )
        opt_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)

        scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=3, min_lr=1e-7)
        scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode='min', factor=0.5, patience=3, min_lr=1e-7)

        # ========================================
        # 4. Trainer
        # ========================================
        trainer = CausalTrainer(
            model=model,
            loss_fn=loss_fn,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
            grad_clip=training_config.get('grad_clip', 1.0),
            use_amp=training_config.get('use_amp', True),
            messenger=messenger
        )

        if resume_path and resume_path.exists():
            trainer.load_checkpoint(resume_path, new_lr=lr)

        # ========================================
        # 5. Training Loop
        # ========================================
        num_epochs = training_config.get('epochs', 100)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_snr = 0.0

        for epoch in range(trainer.current_epoch, num_epochs):
            train_metrics = trainer.train_epoch(train_loader, epoch, num_epochs)
            val_metrics = trainer.validate(val_loader)

            # Streaming validation (causal-specific)
            streaming_result = trainer.validate_streaming()
            
            trainer.history['train_loss'].append(train_metrics['loss_total'])
            trainer.history['val_loss'].append(val_metrics['loss'])
            trainer.history['train_ber'].append(train_metrics['ber'])
            trainer.history['val_ber'].append(val_metrics['ber'])

            scheduler_g.step(val_metrics['loss'])
            scheduler_d.step(val_metrics['loss'])
            current_lr = opt_g.param_groups[0]['lr']

            # Summary
            streaming_status = "✅ PASS" if streaming_result['passed'] else f"❌ FAIL (diff={streaming_result['max_diff']:.2e})"
            
            summary = (
                f"✅ **Epoch {epoch+1}/{num_epochs}** (CAUSAL)\n"
                f"📉 Val Loss: `{val_metrics['loss']:.4f}`\n"
                f"🎯 Val BER: `{val_metrics['ber']:.4f}`\n"
                f"🔊 Val SNR: `{val_metrics['snr']:.1f}dB`\n"
                f"🔄 Streaming: {streaming_status}\n"
                f"📊 LR: `{current_lr:.2e}`"
            )
            print(f"\n{summary.replace('**', '').replace('`', '')}")

            if messenger:
                messenger.send_message(summary)

            # Save checkpoints
            trainer.save_checkpoint(
                save_dir / f"causal_epoch{epoch+1}.pth",
                val_metrics, config, is_latest=True
            )

            if val_metrics['ber'] < trainer.best_ber:
                trainer.best_ber = val_metrics['ber']
                trainer.save_checkpoint(save_dir / "causal_best_ber.pth", val_metrics, config)
                print(f"  ★ New best BER: {trainer.best_ber:.4f}")

            if val_metrics['snr'] > best_val_snr:
                best_val_snr = val_metrics['snr']
                trainer.save_checkpoint(save_dir / "causal_best_snr.pth", val_metrics, config)
                print(f"  ★ New best SNR: {best_val_snr:.1f}dB")

        # Final
        trainer.save_checkpoint(save_dir / "causal_final.pth", val_metrics, config)
        print(f"\n{'='*60}")
        print("Causal Training Completed!")
        print(f"Best BER: {trainer.best_ber:.4f}")
        print(f"{'='*60}")

        if messenger:
            messenger.send_message(f"🎉 **Causal Training Completed**\nBest BER: `{trainer.best_ber:.4f}`")

    except Exception as e:
        error_msg = f"❌ **Causal Training Crashed!**\n```\n{traceback.format_exc()[-1000:]}\n```"
        print(traceback.format_exc())
        if messenger:
            messenger.send_message(error_msg)
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal CallCops Training")

    parser.add_argument('--train_dir', type=str, default='data/raw/training')
    parser.add_argument('--val_dir', type=str, default='data/raw/validation')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints/causal')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_amp', action='store_true')
    
    # Segmentation options
    parser.add_argument('--no_split', action='store_true',
                        help='Disable audio segmentation (use random crop instead)')
    parser.add_argument('--max_frames', type=int, default=128,
                        help='Max frames per segment (128=5.12s @ 8kHz)')
    parser.add_argument('--overlap_ratio', type=float, default=0.0,
                        help='Overlap ratio between segments (0.0-0.5)')

    args = parser.parse_args()

    # Config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'audio': {'sample_rate': 8000, 'segment_length': 40960},
            'watermark': {'payload_length': 128},
            'model': {'hidden_channels': [32, 64, 128, 256], 'num_residual_blocks': 4},
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'use_amp': not args.no_amp
            }
        }

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Data (with segmentation options)
    split_long_audio = not args.no_split
    print(f"[Data] split_long_audio={split_long_audio}, max_frames={args.max_frames}, overlap={args.overlap_ratio}")
    
    train_loader, val_loader = create_train_val_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        sample_rate=config['audio']['sample_rate'],
        pin_memory=True,
        max_frames=args.max_frames,
        split_long_audio=split_long_audio,
        overlap_ratio=args.overlap_ratio
    )

    # Train
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / timestamp

    train_causal(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=save_dir,
        device=device,
        resume_path=Path(args.resume) if args.resume else None
    )


if __name__ == "__main__":
    main()
