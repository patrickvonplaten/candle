//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! https://arxiv.org/abs/1512.03385
use super::utils::{conv2d, Conv2d};
use candle::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

/// Configuration for a ResNet block.
#[derive(Debug, Clone, Copy)]
pub struct ResnetBlock2DConfig {
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    pub output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: Conv2d,
    norm2: nn::GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
    span: tracing::Span,
}

impl ResnetBlock2D {
    pub fn new(vs: nn::VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            stride: 1,
            padding: 1,
            groups: 1,
            dilation: 1,
        };
        let eps = 1e-6;

        let norm1 = nn::group_norm(32, in_channels, eps, vs.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg, vs.pp("conv1"))?;
        let norm2 = nn::group_norm(32, out_channels, eps, vs.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg, vs.pp("conv2"))?;

        let conv_shortcut = if in_channels != out_channels {
            let conv_cfg = nn::Conv2dConfig {
                stride: 1,
                padding: 0,
                groups: 1,
                dilation: 1,
            };
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                conv_cfg,
                vs.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "resnet2d");
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
            span,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = xs;

        let xs = self.norm1.forward(xs)?;
        let xs = nn::ops::silu(&xs)?;
        let xs = self.conv1.forward(&xs)?;

        let xs = self.norm2.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        let xs = self.conv2.forward(&xs)?;

        let residual = match &self.conv_shortcut {
            Some(conv_shortcut) => conv_shortcut.forward(residual)?,
            None => residual.clone(),
        };

        xs + residual
    }
}
