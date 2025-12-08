import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OobleckConfig:
    in_channels: int = 2
    channels: int = 128
    c_mults: List[int] = (1, 2, 4, 8, 16)
    strides: List[int] = (2, 4, 4, 8, 8)
    latent_dim: int = 64
    use_snake: bool = True
    final_tanh: bool = False  # Stable Audio Open 1.0 uses False

class Snake1d(nn.Module):
    """Snake activation with log-scale alpha/beta parameters.
    
    The original implementation uses alpha_logscale=True, meaning the
    alpha and beta parameters are stored in log-space and must be
    exponentiated before use.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Parameters stored in log-space (initialized to 0 = exp(0) = 1)
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        # x: (N, L, C)
        # alpha, beta: (C) - stored in log-space, must exp() before use!
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.square(mx.sin(alpha * x))

class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        # Structure: Act -> Conv -> Act -> Conv
        self.layers = nn.Sequential(
            Snake1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=7, dilation=dilation, padding=((7-1)*dilation)//2),
            Snake1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        )

    def __call__(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # ResUnits -> Act -> Conv
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1),
            ResidualUnit(in_channels, in_channels, dilation=3),
            ResidualUnit(in_channels, in_channels, dilation=9),
            Snake1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=2*stride, stride=stride, padding=(2*stride - stride)//2)
        )

    def __call__(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Act -> Upsample -> ResUnits
        self.layers = nn.Sequential(
            Snake1d(in_channels),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2*stride, stride=stride, padding=(2*stride - stride)//2),
            ResidualUnit(out_channels, out_channels, dilation=1),
            ResidualUnit(out_channels, out_channels, dilation=3),
            ResidualUnit(out_channels, out_channels, dilation=9)
        )

    def __call__(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, config: OobleckConfig):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv1d(config.in_channels, config.channels, kernel_size=7, padding=3))
        
        c_in = config.channels
        for i, stride in enumerate(config.strides):
            c_out = config.channels * config.c_mults[i]
            layers.append(EncoderBlock(c_in, c_out, stride))
            c_in = c_out
            
        layers.append(Snake1d(c_in))
        
        # The encoder outputs 2 * latent_dim (mean and log_var) for VAE bottleneck
        # But OobleckConfig sets latent_dim=64, and weights show 128 output channels.
        layers.append(nn.Conv1d(c_in, config.latent_dim * 2, kernel_size=3, padding=1))
        
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)

class OobleckDecoder(nn.Module):
    def __init__(self, config: OobleckConfig):
        super().__init__()
        
        layers = []
        c_in = config.latent_dim
        c_hidden = config.channels * config.c_mults[-1] # Highest mult
        
        layers.append(nn.Conv1d(c_in, c_hidden, kernel_size=7, padding=3))
        
        strides = list(reversed(config.strides))
        c_mults = list(reversed(config.c_mults))
        c_in = c_hidden
        
        for i, stride in enumerate(strides):
            if i < len(c_mults) - 1:
                c_out = config.channels * c_mults[i+1]
            else:
                c_out = config.channels
            
            layers.append(DecoderBlock(c_in, c_out, stride))
            c_in = c_out
            
        layers.append(Snake1d(c_in))
        # Final layer has no bias in Stable Audio Open 1.0 weights
        layers.append(nn.Conv1d(c_in, config.in_channels, kernel_size=7, padding=3, bias=False))
        # Stable Audio Open 1.0 uses final_tanh=False
        if config.final_tanh:
            layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)

class AutoencoderOobleck(nn.Module):
    def __init__(self, config: OobleckConfig = OobleckConfig()):
        super().__init__()
        self.encoder = OobleckEncoder(config)
        self.decoder = OobleckDecoder(config)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def __call__(self, x):
        return self.decode(self.encode(x))
