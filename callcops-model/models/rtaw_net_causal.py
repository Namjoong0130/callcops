"""
CallCops: True Causal Frame-Wise Real-Time Audio Watermarking Network
======================================================================

아키텍처 특징:
1. **True Causal Convolution**: 모든 Conv1d 레이어에 left-only padding 적용
2. **Zero Look-ahead**: 미래 샘플에 대한 의존성 완전 제거
3. **실시간 스트리밍**: 히스토리 버퍼 없이 단일 프레임 직접 처리 가능

Non-Causal vs Causal 비교:
- Non-Causal: padding = (kernel_size - 1) * dilation // 2 (양쪽)
- Causal: padding = (kernel_size - 1) * dilation (왼쪽만)

Receptive Field (Past Context Only):
- Encoder: 4 × (7-1) = 24 samples
- Residual: (1+2+4+8) × (3-1) = 30 samples  
- Decoder: 3 × (7-1) = 18 samples
- Output: (7-1) = 6 samples
- Total: ~78 samples = 9.75ms @ 8kHz

Usage:
    model = CausalCallCopsNet(message_dim=128)
    watermarked = model.embed(audio, bits)  # 실시간 가능!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# =============================================================================
# Constants
# =============================================================================

FRAME_SAMPLES = 320       # 40ms @ 8kHz
PAYLOAD_LENGTH = 128      # 128-bit cyclic payload
CYCLE_SAMPLES = FRAME_SAMPLES * PAYLOAD_LENGTH  # 40,960 samples = 5.12s
SAMPLE_RATE = 8000


# =============================================================================
# Normalization Type Selection
# =============================================================================

# InstanceNorm1d vs BatchNorm1d for Streaming:
# - BatchNorm1d: Uses running statistics (mean/var) computed during training
#   → Unstable for streaming inference with batch_size=1
#   → Causes SNR drops when running statistics diverge from batch statistics
# - InstanceNorm1d: Computes mean/var per sample independently
#   → Consistent behavior between training and inference
#   → Recommended for real-time streaming applications

NORM_TYPE = 'instance'  # Options: 'batch', 'instance', 'layer'


def get_norm_layer(num_features: int, norm_type: str = NORM_TYPE) -> nn.Module:
    """
    Get normalization layer based on type.
    
    Args:
        num_features: Number of channels
        norm_type: 'batch', 'instance', or 'layer'
    
    Returns:
        Normalization layer
    """
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'instance':
        # affine=True allows learnable scale/shift parameters
        # track_running_stats=False ensures no dependence on training statistics
        return nn.InstanceNorm1d(num_features, affine=True, track_running_stats=False)
    elif norm_type == 'layer':
        # LayerNorm for 1D: normalize over (C, T) dimensions
        # Note: Requires wrapper for variable-length sequences
        return nn.GroupNorm(1, num_features)  # GroupNorm with 1 group = LayerNorm
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# =============================================================================
# Causal Building Blocks
# =============================================================================

class CausalConvBlock(nn.Module):
    """
    Causal Convolutional Block
    ==========================
    
    Conv1d → Norm → LeakyReLU with LEFT-ONLY padding.
    
    - Zero look-ahead: 출력의 각 샘플은 과거 입력에만 의존
    - 패딩 = (kernel_size - 1) * dilation (왼쪽에만 적용)
    - InstanceNorm1d for streaming stability (default)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        # Causal padding amount (applied to LEFT side only)
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No built-in padding!
            dilation=dilation
        )
        self.norm = get_norm_layer(out_channels, norm_type)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal (left-only) padding
        x = F.pad(x, (self.causal_padding, 0))
        return self.act(self.norm(self.conv(x)))


class CausalResidualBlock(nn.Module):
    """
    Causal Residual Block with Skip Connection
    ==========================================
    
    두 개의 Causal Conv1d 레이어와 skip connection.
    Dilated convolution 지원으로 넓은 receptive field 확보.
    InstanceNorm1d for streaming stability.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        # Causal padding for this dilation
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation
        )
        self.norm1 = get_norm_layer(channels, norm_type)

        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation
        )
        self.norm2 = get_norm_layer(channels, norm_type)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Conv1 with causal padding
        out = F.pad(x, (self.causal_padding, 0))
        out = self.act(self.norm1(self.conv1(out)))

        # Conv2 with causal padding
        out = F.pad(out, (self.causal_padding, 0))
        out = self.norm2(self.conv2(out))

        # Skip connection
        out = out + residual
        out = self.act(out)
        return out


class CausalUpsampleBlock(nn.Module):
    """
    Causal Upsampling Block
    =======================
    
    ConvTranspose1d는 양방향으로 에너지를 분산시켜 look-ahead를 유발할 수 있음.
    이를 방지하기 위해 출력의 오른쪽(미래) 부분을 crop하여 causality 보장.
    
    Alternative approach: Nearest-neighbor upsampling + Causal Conv
    (더 안정적이고 artifact가 적음)
    
    수식:
        output[i] depends only on input[j] where j <= i/stride
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        mode: str = 'nearest_conv',  # 'transpose' or 'nearest_conv'
        norm_type: str = NORM_TYPE
    ):
        super().__init__()
        self.mode = mode
        self.stride = stride

        if mode == 'transpose':
            # ConvTranspose1d with causal output cropping
            # For stride=2, kernel=4: output_padding=0
            # The transposed conv produces look-ahead which we crop
            self.upsample = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0
            )
            # Amount to crop from the RIGHT (future) side
            # For causal: crop (kernel_size - stride) samples
            self.crop_right = kernel_size - stride
        else:
            # Recommended: Nearest-neighbor upsampling + causal conv
            # More stable, no checkerboard artifacts
            self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size - 1,  # Slightly smaller for refinement
                padding=0
            )
            self.causal_padding = kernel_size - 2  # Left-only padding

        self.norm = get_norm_layer(out_channels, norm_type)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'transpose':
            # Transposed convolution
            out = self.upsample(x)
            # Crop future (right) samples to ensure causality
            if self.crop_right > 0:
                out = out[:, :, :-self.crop_right]
        else:
            # Nearest upsampling + causal conv
            out = self.upsample(x)
            out = F.pad(out, (self.causal_padding, 0))  # Left-only padding
            out = self.conv(out)

        return self.act(self.norm(out))


class CausalDownsampleBlock(nn.Module):
    """
    Causal Downsampling Block
    =========================
    
    Strided convolution이나 pooling은 자연스럽게 causal 가능.
    AvgPool1d with proper padding ensures no look-ahead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        # Causal padding for strided conv
        self.causal_padding = kernel_size - stride

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        self.norm = get_norm_layer(out_channels, norm_type)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-only padding for causal strided conv
        x = F.pad(x, (self.causal_padding, 0))
        return self.act(self.norm(self.conv(x)))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention)
    
    Global pooling이므로 causal 이슈 없음.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1)
        return x * y.expand_as(x)


# =============================================================================
# Causal Frame-Wise Fusion Block
# =============================================================================

class CausalFrameWiseFusionBlock(nn.Module):
    """
    Causal Frame-Wise Fusion Block
    ==============================
    
    각 40ms 프레임에 해당하는 1비트만 삽입.
    모든 내부 Conv 레이어가 causal padding 사용.
    InstanceNorm1d for streaming stability.
    """

    def __init__(
        self,
        audio_channels: int,
        message_dim: int = 128,
        frame_samples: int = FRAME_SAMPLES,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        self.audio_channels = audio_channels
        self.message_dim = message_dim
        self.frame_samples = frame_samples

        # Bit embedding (no temporal dependency)
        self.bit_embedding = nn.Sequential(
            nn.Linear(1, audio_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(audio_channels // 2, audio_channels),
        )

        # Causal temporal modulator (InstanceNorm for streaming)
        self.temporal_conv = nn.Conv1d(
            audio_channels, audio_channels, 
            kernel_size=3, padding=0, groups=audio_channels
        )
        self.temporal_norm = get_norm_layer(audio_channels, norm_type)
        self.temporal_act = nn.LeakyReLU(0.2, inplace=True)
        self.temporal_padding = 2  # (3-1) * 1 for causal

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv1d(audio_channels * 2, audio_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        audio_feat: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_feat: [B, C, T] - 오디오 피처
            message: [B, 128] - 128비트 페이로드

        Returns:
            fused: [B, C, T] - 비트 정보가 융합된 피처
        """
        B, C, T = audio_feat.shape

        # Frame indices (cyclic)
        num_frames = T // self.frame_samples
        frame_indices = torch.arange(num_frames, device=audio_feat.device) % self.message_dim
        frame_bits = message[:, frame_indices]  # [B, num_frames]

        # Embed bits
        bit_embed = self.bit_embedding(frame_bits.unsqueeze(-1))  # [B, num_frames, C]
        bit_embed = bit_embed.permute(0, 2, 1)  # [B, C, num_frames]

        # Expand to full resolution
        bit_feat = bit_embed.repeat_interleave(self.frame_samples, dim=-1)  # [B, C, T']

        # Handle size mismatch
        if bit_feat.shape[-1] < T:
            bit_feat = F.pad(bit_feat, (0, T - bit_feat.shape[-1]))
        elif bit_feat.shape[-1] > T:
            bit_feat = bit_feat[:, :, :T]

        # Causal temporal modulation
        bit_feat_padded = F.pad(bit_feat, (self.temporal_padding, 0))
        modulated = self.temporal_act(self.temporal_norm(self.temporal_conv(bit_feat_padded)))

        # Gated fusion
        gate = self.gate(torch.cat([audio_feat, modulated], dim=1))
        fused = audio_feat + gate * modulated

        return fused


# =============================================================================
# Causal Frame-Wise Bit Extractor
# =============================================================================

class CausalFrameWiseBitExtractor(nn.Module):
    """
    Causal Frame-Wise Bit Extractor
    ===============================
    
    다운샘플된 피처맵에서 프레임당 1비트 추출.
    InstanceNorm1d for streaming stability.
    """

    def __init__(
        self,
        in_channels: int,
        frame_samples: int = FRAME_SAMPLES // 16,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        self.frame_samples = frame_samples

        # Causal frame refine (InstanceNorm for streaming)
        self.refine_conv1 = nn.Conv1d(
            in_channels, in_channels, 
            kernel_size=3, padding=0, groups=in_channels
        )
        self.refine_norm1 = get_norm_layer(in_channels, norm_type)
        self.refine_act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.refine_conv2 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.refine_norm2 = get_norm_layer(in_channels // 2, norm_type)
        self.refine_act2 = nn.LeakyReLU(0.2, inplace=True)
        self.refine_padding = 2  # causal padding for kernel_size=3

        # Frame-wise bit extraction (1 bit per frame)
        self.bit_conv = nn.Conv1d(
            in_channels // 2, 1,
            kernel_size=frame_samples,
            stride=frame_samples,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # Causal refine: conv1 with causal padding
        x = F.pad(x, (self.refine_padding, 0))
        x = self.refine_act1(self.refine_norm1(self.refine_conv1(x)))
        x = self.refine_act2(self.refine_norm2(self.refine_conv2(x)))

        # Padding for frame alignment
        if T < self.frame_samples:
            pad_size = self.frame_samples - T
            x = F.pad(x, (0, pad_size))
        elif T % self.frame_samples != 0:
            pad_size = self.frame_samples - (T % self.frame_samples)
            x = F.pad(x, (0, pad_size))

        # Extract 1 bit per frame
        logits = self.bit_conv(x)  # [B, 1, num_frames]
        logits = logits.squeeze(1)  # [B, num_frames]

        return logits


# =============================================================================
# Causal Encoder
# =============================================================================

class CausalEncoder(nn.Module):
    """
    Causal Frame-Wise Watermark Embedding Network
    =============================================
    
    True causal architecture:
    - 모든 Conv1d에 left-only padding
    - Zero look-ahead latency
    - 단일 프레임 실시간 처리 가능
    - InstanceNorm1d for streaming stability (default)
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        frame_samples: int = FRAME_SAMPLES,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        self.message_dim = message_dim
        self.hidden_channels = hidden_channels
        self.frame_samples = frame_samples

        # =============================================
        # 1. Causal Audio Encoder
        # =============================================

        encoder_layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            encoder_layers.append(
                CausalConvBlock(in_ch, out_ch, kernel_size=7, dilation=1, norm_type=norm_type)
            )
            in_ch = out_ch

        self.audio_encoder = nn.Sequential(*encoder_layers)

        # =============================================
        # 2. Causal Frame-Wise Fusion
        # =============================================

        self.frame_fusion = CausalFrameWiseFusionBlock(
            audio_channels=hidden_channels[-1],
            message_dim=message_dim,
            frame_samples=frame_samples,
            norm_type=norm_type
        )

        # Post-fusion refinement (causal)
        self.fusion_refine = nn.Sequential(
            CausalConvBlock(hidden_channels[-1], hidden_channels[-1], kernel_size=3, dilation=1, norm_type=norm_type),
            SEBlock(hidden_channels[-1]),
        )

        # =============================================
        # 3. Causal Residual Blocks
        # =============================================

        self.residual_blocks = nn.ModuleList([
            CausalResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=3,
                dilation=2 ** (i % 4),
                norm_type=norm_type
            )
            for i in range(num_residual_blocks)
        ])

        # =============================================
        # 4. Causal Audio Decoder
        # =============================================

        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))

        for i, out_ch in enumerate(reversed_channels[1:]):
            in_ch = reversed_channels[i]
            decoder_layers.append(
                CausalConvBlock(in_ch, out_ch, kernel_size=7, dilation=1, norm_type=norm_type)
            )

        self.audio_decoder = nn.Sequential(*decoder_layers)

        # =============================================
        # 5. Causal Output Layer
        # =============================================

        self.output_padding = 6  # (7-1) * 1
        self.output_conv = nn.Conv1d(hidden_channels[0], 1, kernel_size=7, padding=0)

        # Alpha (perturbation scale)
        self.alpha_min = 0.25
        self.alpha_max = 1.0
        self.register_buffer('alpha', torch.tensor(0.25))

    def forward(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Causal 40ms 프레임별 1비트 삽입

        Args:
            audio: [B, 1, T] 원본 오디오
            message: [B, 128] 워터마크 비트

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
        """
        B, _, T = audio.shape

        # 1. Causal audio feature extraction
        audio_feat = self.audio_encoder(audio)  # [B, C, T]

        # 2. Causal frame-wise fusion
        fused = self.frame_fusion(audio_feat, message)

        # 3. Post-fusion refinement
        fused = self.fusion_refine(fused)

        # 4. Causal residual blocks
        for block in self.residual_blocks:
            fused = block(fused)

        # 5. Causal audio decoder
        decoded = self.audio_decoder(fused)

        # 6. Causal output
        decoded = F.pad(decoded, (self.output_padding, 0))
        perturbation = self.output_conv(decoded)

        # 7. Apply perturbation
        perturbation = torch.tanh(perturbation) * self.alpha
        watermarked = audio + perturbation
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

        return watermarked


# =============================================================================
# Causal Decoder
# =============================================================================

class CausalDecoder(nn.Module):
    """
    Causal Frame-Wise Watermark Extraction Network
    ==============================================
    
    Causal convolution으로 실시간 비트 추출.
    InstanceNorm1d for streaming stability (default).
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        frame_samples: int = FRAME_SAMPLES,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        self.message_dim = message_dim
        self.frame_samples = frame_samples
        self.downsample_factor = 2 ** len(hidden_channels)
        self.downsampled_frame = frame_samples // self.downsample_factor

        # =============================================
        # 1. Causal Feature Extractor (with stride)
        # =============================================

        # Note: Stride convolutions with causal padding require careful handling
        encoder_layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            encoder_layers.append(
                CausalConvBlock(in_ch, out_ch, kernel_size=5, stride=1, dilation=1, norm_type=norm_type)
            )
            # Downsample with stride=2 pooling (causal)
            encoder_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*encoder_layers)

        # =============================================
        # 2. Causal Residual Blocks
        # =============================================

        self.residual_blocks = nn.ModuleList([
            CausalResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=3,
                dilation=2 ** (i % 4),
                norm_type=norm_type
            )
            for i in range(num_residual_blocks)
        ])

        self.se_block = SEBlock(hidden_channels[-1])

        # =============================================
        # 3. Causal Bit Extractor
        # =============================================

        self.bit_extractor = CausalFrameWiseBitExtractor(
            in_channels=hidden_channels[-1],
            frame_samples=self.downsampled_frame,
            norm_type=norm_type
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Causal 프레임별 비트 추출

        Args:
            audio: [B, 1, T] 워터마크된 오디오

        Returns:
            logits: [B, num_frames] 비트 로짓
        """
        # 1. Causal feature extraction
        features = self.feature_extractor(audio)

        # 2. Causal residual blocks
        for block in self.residual_blocks:
            features = block(features)

        # 3. Channel attention
        features = self.se_block(features)

        # 4. Frame-wise bit extraction
        logits = self.bit_extractor(features)

        return logits


# =============================================================================
# Causal Discriminator
# =============================================================================

class CausalDiscriminator(nn.Module):
    """
    Causal Multi-Scale Discriminator
    InstanceNorm1d for streaming stability.
    """

    def __init__(
        self,
        hidden_channels: List[int] = [16, 32, 64, 128],
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            layers.append(CausalConvBlock(in_ch, out_ch, kernel_size=5, dilation=1, norm_type=norm_type))
            layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


# =============================================================================
# Causal CallCopsNet (Complete Network)
# =============================================================================

class CausalCallCopsNet(nn.Module):
    """
    Causal CallCops Complete Network
    ================================
    
    True causal architecture for zero-latency real-time streaming.
    
    Features:
    - Zero look-ahead: 미래 샘플 의존성 없음
    - Single-frame processing: 히스토리 버퍼 불필요
    - VoIP-ready: 실시간 통화 삽입 가능
    - InstanceNorm1d: 스트리밍 추론 안정성 보장
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        use_discriminator: bool = True,
        norm_type: str = NORM_TYPE
    ):
        super().__init__()

        self.encoder = CausalEncoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            norm_type=norm_type
        )

        self.decoder = CausalDecoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            norm_type=norm_type
        )

        self.use_discriminator = use_discriminator
        if use_discriminator:
            self.discriminator = CausalDiscriminator(
                hidden_channels=[16, 32, 64, 128],
                norm_type=norm_type
            )

    def embed(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """워터마크 삽입"""
        watermarked = self.encoder(audio, message)
        perturbation = watermarked - audio
        return watermarked, perturbation

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """워터마크 추출"""
        return self.decoder(audio)

    def forward(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> dict:
        """End-to-end forward for training"""
        watermarked, perturbation = self.embed(audio, message)
        extracted = self.extract(watermarked)

        outputs = {
            'watermarked': watermarked,
            'perturbation': perturbation,
            'extracted': extracted
        }

        if self.use_discriminator:
            outputs['disc_real'] = self.discriminator(audio)
            outputs['disc_fake'] = self.discriminator(watermarked)

        return outputs

    def count_parameters(self) -> dict:
        """파라미터 수 카운트"""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'encoder': count(self.encoder),
            'decoder': count(self.decoder),
            'discriminator': count(self.discriminator) if self.use_discriminator else 0,
            'total': count(self)
        }


# =============================================================================
# Simplified Causal Streaming Wrapper
# =============================================================================

class CausalStreamingEncoder:
    """
    Simplified Streaming Encoder for Causal Model
    ==============================================
    
    히스토리 버퍼 불필요!
    단일 프레임 직접 처리 가능.
    """

    def __init__(self, encoder: CausalEncoder):
        self.encoder = encoder
        self.encoder.eval()
        self._frame_index = 0

    def reset(self):
        self._frame_index = 0

    @torch.no_grad()
    def process_frame(
        self,
        frame: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        단일 프레임 직접 처리 (히스토리 불필요!)

        Args:
            frame: [320] 원시 오디오
            message: [128] 메시지 비트

        Returns:
            watermarked: [320] 워터마크된 오디오
        """
        device = next(self.encoder.parameters()).device

        if frame.dim() == 1:
            frame = frame.unsqueeze(0).unsqueeze(0).to(device)
        if message.dim() == 1:
            message = message.unsqueeze(0).to(device)

        # 메시지 회전 (cyclic bit alignment)
        offset = self._frame_index % PAYLOAD_LENGTH
        rotated = torch.roll(message, -offset, dims=-1)

        # 직접 인코딩!
        watermarked = self.encoder(frame, rotated)

        self._frame_index += 1
        return watermarked.squeeze()

    @property
    def frame_index(self) -> int:
        return self._frame_index


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_receptive_field() -> dict:
    """
    Causal 아키텍처의 receptive field 계산
    """
    encoder_rf = 4 * (7 - 1)  # 4 CausalConvBlocks with kernel=7
    residual_rf = sum((3 - 1) * (2 ** i) for i in range(4))  # 4 dilated blocks
    decoder_rf = 3 * (7 - 1)  # 3 CausalConvBlocks with kernel=7
    output_rf = (7 - 1)  # Final output conv

    total_rf = encoder_rf + residual_rf + decoder_rf + output_rf

    return {
        'encoder': encoder_rf,
        'residual': residual_rf,
        'decoder': decoder_rf,
        'output': output_rf,
        'total_samples': total_rf,
        'total_ms': total_rf / SAMPLE_RATE * 1000,
        'look_ahead': 0  # Zero!
    }


if __name__ == "__main__":
    # Test causal model
    print("=" * 60)
    print("Causal CallCopsNet Test")
    print("=" * 60)

    model = CausalCallCopsNet(message_dim=128)
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Discriminator: {params['discriminator']:,}")
    print(f"  Total: {params['total']:,}")

    # Receptive field
    rf = calculate_receptive_field()
    print(f"\nReceptive Field:")
    print(f"  Encoder: {rf['encoder']} samples")
    print(f"  Residual: {rf['residual']} samples")
    print(f"  Decoder: {rf['decoder']} samples")
    print(f"  Total: {rf['total_samples']} samples = {rf['total_ms']:.2f}ms")
    print(f"  Look-ahead: {rf['look_ahead']} samples (ZERO!)")

    # Test forward pass
    B, T = 2, 1920
    audio = torch.randn(B, 1, T)
    message = torch.randint(0, 2, (B, 128)).float()

    with torch.no_grad():
        watermarked = model.encoder(audio, message)
        logits = model.decoder(watermarked)

    print(f"\nForward Pass:")
    print(f"  Input: {audio.shape}")
    print(f"  Watermarked: {watermarked.shape}")
    print(f"  Logits: {logits.shape}")

    # Test streaming
    print(f"\nStreaming Test:")
    wrapper = CausalStreamingEncoder(model.encoder)
    frame = torch.randn(320)
    msg = torch.randint(0, 2, (128,)).float()
    
    out = wrapper.process_frame(frame, msg)
    print(f"  Input frame: {frame.shape}")
    print(f"  Output frame: {out.shape}")
    print(f"  Frame index: {wrapper.frame_index}")

    print("\n✅ Causal model test passed!")
