# CallCops Models v2.0
# ====================
# Frame-Wise Real-Time Audio Watermarking Neural Network Components

from .rtaw_net import (
    # Main classes
    CallCopsNet,
    Encoder,
    Decoder,
    Discriminator,
    # Building blocks
    ConvBlock,
    ResidualBlock,
    SEBlock,
    FrameWiseFusionBlock,
    FrameWiseBitExtractor,
    # Constants
    FRAME_SAMPLES,
    PAYLOAD_LENGTH,
    CYCLE_SAMPLES,
    SAMPLE_RATE,
    # Utilities
    align_to_frames,
    get_cyclic_bit_indices,
)

from .codec_simulator import (
    DifferentiableCodecSimulator,
    G711Simulator,
    G729Simulator,
)

from .streaming import (
    StreamingEncoderWrapper,
    validate_streaming_vs_batch,
)

# Causal Architecture (Zero Look-ahead)
from .rtaw_net_causal import (
    CausalCallCopsNet,
    CausalEncoder,
    CausalDecoder,
    CausalDiscriminator,
    CausalConvBlock,
    CausalResidualBlock,
    CausalFrameWiseFusionBlock,
    CausalFrameWiseBitExtractor,
    CausalStreamingEncoder,
    calculate_receptive_field,
)

from .losses import (
    CallCopsLoss,
    CallShieldLoss,
    MultiResolutionMelLoss,
    BitAccuracyLoss,
    AdversarialLoss,
)

__all__ = [
    # Main network
    "CallCopsNet",
    "Encoder",
    "Decoder",
    "Discriminator",
    # Building blocks
    "ConvBlock",
    "ResidualBlock",
    "SEBlock",
    "FrameWiseFusionBlock",
    "FrameWiseBitExtractor",
    # Constants
    "FRAME_SAMPLES",
    "PAYLOAD_LENGTH",
    "CYCLE_SAMPLES",
    "SAMPLE_RATE",
    # Utilities
    "align_to_frames",
    "get_cyclic_bit_indices",
    # Codec simulation
    "DifferentiableCodecSimulator",
    "G711Simulator",
    "G729Simulator",
    # Streaming
    "StreamingEncoderWrapper",
    "validate_streaming_vs_batch",
    # Causal Architecture
    "CausalCallCopsNet",
    "CausalEncoder",
    "CausalDecoder",
    "CausalDiscriminator",
    "CausalConvBlock",
    "CausalResidualBlock",
    "CausalFrameWiseFusionBlock",
    "CausalFrameWiseBitExtractor",
    "CausalStreamingEncoder",
    "calculate_receptive_field",
    # Losses
    "CallCopsLoss",
    "CallShieldLoss",
    "MultiResolutionMelLoss",
    "BitAccuracyLoss",
    "AdversarialLoss",
]
