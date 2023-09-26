//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use super::embeddings::{TimestepEmbedding, Timesteps};
use super::unet_2d_blocks::*;
use super::utils::{conv2d, Conv2d};
use candle::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub out_channels: usize,
    /// When `None` no cross-attn is used, when `Some(d)` then cross-attn is used and `d` is the
    /// number of transformer blocks to be used.
    pub use_cross_attn: Option<usize>,
    pub attention_head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct UViTTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub layers_per_block: usize,
    pub attn_head_dim: usize,
    pub context_dim: usize,
    pub condition_input: usize,
    pub encoder_hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
}

impl Default for UViTTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 768,
            out_channels: 768,
            layers_per_block: 3,
            attn_head_dim: 64,
            context_dim: 768,
            condition_input: 128,
            encoder_hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 16,
            vocab_size: 8256,
            hidden_size: 1024,
            norm_num_groups: 32,
            norm_eps: 1e-6,
        }
    }
}

#[derive(Debug)]
pub(crate) enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct ConditionEmbedding {
    linear_in: nn::Linear,
    linear_out: nn::Linear,
}

impl ConditionEmbedding {
    pub fn new(input_dim: usize, hidden_dim: usize, vs: nn::VarBuilder) -> Result<Self> {
        let linear_in = nn::linear(input_dim, hidden_dim, vs.pp("0"))?;
        // add a SiLU activation function here
        let linear_out = nn::linear(input_dim, hidden_dim, vs.pp("2"))?;
        Ok(Self {
            linear_in,
            linear_out,
        })
    }
}

#[derive(Debug)]
pub struct CrossAttention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    heads: usize,
    scale: f64,
    span: tracing::Span,
    span_attn: tracing::Span,
    span_softmax: tracing::Span,
    use_flash_attn: bool,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = nn::linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_k = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_k"))?;
        let to_v = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_v"))?;
        let to_out = nn::linear(inner_dim, query_dim, vs.pp("to_out.0"))?;
        let span = tracing::span!(tracing::Level::TRACE, "xa");
        let span_attn = tracing::span!(tracing::Level::TRACE, "xa-attn");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "xa-softmax");
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            scale,
            span,
            span_attn,
            span_softmax,
            use_flash_attn,
        })
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }
}

/// A basic Transformer block.
#[derive(Debug)]
struct AttentionBlock2D {
    attn1: CrossAttention,
    attn2: CrossAttention,
    norm1: nn::RmsNorm,
    norm2: nn::RmsNorm,
    span: tracing::Span,
}

impl AttentionBlock2D {
    fn new(
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        use_flash_attn: bool,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let attn1 =
            CrossAttention::new(vs.pp("attn1"), dim, None, n_heads, d_head, use_flash_attn)?;
        let attn2 = CrossAttention::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            use_flash_attn,
        )?;
        let norm1 = nn::rms_norm(dim, 1e-5, vs.pp("norm1"))?;
        let norm2 = nn::rms_norm(dim, 1e-5, vs.pp("norm2"))?;
        let span = tracing::span!(tracing::Level::TRACE, "basic-transformer");
        Ok(Self {
            attn1,
            attn2,
            norm1,
            norm2,
            span,
        })
    }
}

#[derive(Debug)]
pub struct ConvEmbedding {
    embedding: nn::Embedding,
    layer_norm: nn::RmsNorm,
    conv: nn::Conv2d,
}

impl ConvEmbedding {
    pub fn new(
        vocab_size: usize,
        in_channels: usize,
        out_channels: usize,
        eps: f64,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let embedding = nn::embedding(vocab_size, in_channels, vs.pp("embeddings"))?;
        let layer_norm = nn::rms_norm(in_channels, eps, vs.pp("layer_norm"))?;
        let conv_cfg = Default::default();
        let conv = nn::conv2d(in_channels, out_channels, 1, conv_cfg, vs.pp("conv"))?;
        Ok(Self {
            embedding,
            layer_norm,
            conv,
        })
    }
}

#[derive(Debug)]
pub struct ResBlock {
    depthwise: nn::Conv2d,
    norm: nn::RmsNorm,
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    adaLN_modulation: AdaLNModulation,
}

impl ResBlock {
    pub fn new(
        in_channels: usize,
        res_ffn_factor: usize,
        norm_eps: f64,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Default::default();
        let depthwise = nn::conv2d(in_channels, in_channels, 3, conv_cfg, vs.pp("depthwise"))?;
        let norm = nn::rms_norm(in_channels, norm_eps, vs.pp("attn_layer_norm"))?;

        let linear_hid_size = (res_ffn_factor * in_channels) as usize;
        let seq_vs = vs.pp("channelwise");
        let linear_1 = nn::linear(in_channels, linear_hid_size, seq_vs.pp("0"))?;
        let linear_2 = nn::linear(in_channels, linear_hid_size, seq_vs.pp("4"))?;

        let adaLN_modulation = AdaLNModulation::new(in_channels, vs.pp("adaLN_modulation"))?;

        Ok(Self {
            depthwise,
            norm,
            linear_1,
            linear_2,
            adaLN_modulation,
        })
    }
}

#[derive(Debug)]
pub struct DownsampleBlock {
    resnets: Vec<ResBlock>,
    attentions: Vec<AttentionBlock2D>,
    span: tracing::Span,
}

impl DownsampleBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_downblock_layers: usize,
        attn_num_head_channels: usize,
        context_dim: usize,
        use_flash_attn: bool,
        norm_eps: f64,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let n_heads = attn_num_head_channels;

        let vs_resnets = vs.pp("res_blocks");
        let resnets = (0..num_downblock_layers)
            .map(|i| ResBlock::new(out_channels, 4, norm_eps, vs_resnets.pp(&i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        let vs_attn = vs.pp("attention_blocks");

        let attentions = (0..num_downblock_layers)
            .map(|i| {
                AttentionBlock2D::new(
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    Some(context_dim),
                    use_flash_attn,
                    vs_attn.pp(&i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "xa-down2d");
        Ok(Self {
            resnets,
            attentions,
            span,
        })
    }
}

#[derive(Debug)]
pub struct UpsampleBlock {
    resnets: Vec<ResBlock>,
    attentions: Vec<AttentionBlock2D>,
    span: tracing::Span,
}

impl UpsampleBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_upblock_layers: usize,
        attn_num_head_channels: usize,
        context_dim: usize,
        use_flash_attn: bool,
        norm_eps: f64,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let n_heads = attn_num_head_channels;

        let vs_resnets = vs.pp("res_blocks");
        let resnets = (0..num_upblock_layers)
            .map(|i| ResBlock::new(out_channels, 4, norm_eps, vs_resnets.pp(&i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        let vs_attn = vs.pp("attention_blocks");

        let attentions = (0..num_upblock_layers)
            .map(|i| {
                AttentionBlock2D::new(
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    Some(context_dim),
                    use_flash_attn,
                    vs_attn.pp(&i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "xa-down2d");
        Ok(Self {
            resnets,
            attentions,
            span,
        })
    }
}

#[derive(Debug)]
pub struct AdaLNModulation {
    linear: nn::Linear,
}

impl AdaLNModulation {
    pub fn new(hidden_size: usize, vs: nn::VarBuilder) -> Result<Self> {
        let linear = nn::linear(hidden_size, 2 * hidden_size, vs.pp("mapper"))?;
        Ok(Self { linear })
    }
}

/// A feed-forward layer.
#[derive(Debug)]
struct FeedForward {
    wi_0: nn::Linear,
    wi_1: nn::Linear,
    wo: nn::Linear,
    span: tracing::Span,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(hidden_size: usize, inner_dim: Option<usize>, vs: nn::VarBuilder) -> Result<Self> {
        let inner_dim = inner_dim.unwrap_or(4 * hidden_size);

        let wi_0 = nn::linear(hidden_size, inner_dim, vs.pp("wi_0"))?;
        let wi_1 = nn::linear(hidden_size, inner_dim, vs.pp("wi_1"))?;
        let wo = nn::linear(inner_dim, hidden_size, vs.pp("wo"))?;

        let span = tracing::span!(tracing::Level::TRACE, "ff");
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            span,
        })
    }
}

#[derive(Debug)]
struct ConvMLMLayer {
    conv1: nn::Conv2d,
    layer_norm: nn::RmsNorm,
    conv2: nn::Conv2d,
}

impl ConvMLMLayer {
    fn new(
        in_channels: usize,
        vocab_size: usize,
        out_channels: usize,
        eps: f64,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Default::default();
        let conv1 = nn::conv2d(out_channels, in_channels, 1, conv_cfg, vs.pp("conv1"))?;
        let conv2 = nn::conv2d(in_channels, out_channels, 1, conv_cfg, vs.pp("conv2"))?;
        let layer_norm = nn::rms_norm(in_channels, eps, vs.pp("layer_norm"))?;

        Ok(Self {
            conv1,
            layer_norm,
            conv2,
        })
    }
}

#[derive(Debug)]
pub struct TransformerLayer {
    attn_layer_norm: nn::RmsNorm,
    attention: CrossAttention,
    self_attn_adaLN_modulation: AdaLNModulation,
    cross_attn_layer_norm: nn::RmsNorm,
    cross_attention: CrossAttention,
    cross_attn_adaLN_modulation: AdaLNModulation,
    ffn: FeedForward,
}

impl TransformerLayer {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        attn_head_dim: usize,
        context_dim: usize,
        norm_eps: f64,
        use_flash_attn: bool,
        vs: nn::VarBuilder,
    ) -> Result<Self> {
        let attn_layer_norm = nn::rms_norm(hidden_size, norm_eps, vs.pp("attn_layer_norm"))?;
        let attention = CrossAttention::new(
            vs.pp("attention"),
            hidden_size,
            None,
            num_attention_heads,
            attn_head_dim,
            use_flash_attn,
        )?;
        let self_attn_adaLN_modulation =
            AdaLNModulation::new(hidden_size, vs.pp("self_attn_adaLN_modulation"))?;
        let cross_attn_layer_norm =
            nn::rms_norm(hidden_size, norm_eps, vs.pp("crossattn_layer_norm"))?;
        let cross_attention = CrossAttention::new(
            vs.pp("crossattention"),
            hidden_size,
            Some(context_dim),
            num_attention_heads,
            attn_head_dim,
            use_flash_attn,
        )?;
        let cross_attn_adaLN_modulation =
            AdaLNModulation::new(hidden_size, vs.pp("cross_attn_adaLN_modulation"))?;

        let ffn = FeedForward::new(hidden_size, None, vs.pp("ffn"))?;

        Ok(Self {
            attn_layer_norm,
            attention,
            self_attn_adaLN_modulation,
            cross_attn_layer_norm,
            cross_attention,
            cross_attn_adaLN_modulation,
            ffn,
        })
    }
}

#[derive(Debug)]
pub struct UViTTransformer {
    encoder_proj: nn::Linear,
    encoder_proj_layer_norm: nn::RmsNorm,
    embed: ConvEmbedding,
    cond_embed: ConditionEmbedding,
    down_blocks: Vec<DownsampleBlock>,
    up_blocks: Vec<UpsampleBlock>,
    project_to_hidden_norm: nn::RmsNorm,
    project_to_hidden: nn::Linear,
    transformer_layers: Vec<TransformerLayer>,
    project_from_hidden_norm: nn::RmsNorm,
    project_from_hidden: nn::Linear,
    mlm_layer: ConvMLMLayer,
    config: UViTTransformerConfig,
}

impl UViTTransformer {
    pub fn new(config: UViTTransformerConfig, vs: nn::VarBuilder) -> Result<Self> {
        let use_flash_attn = false;
        let encoder_proj = nn::linear(
            config.encoder_hidden_size,
            config.context_dim,
            vs.pp("encoder_proj"),
        )?;
        println!("suh-du");
        let encoder_proj_layer_norm = nn::rms_norm(
            config.hidden_size,
            config.norm_eps,
            vs.pp("encoder_proj_layer_norm "),
        )?;
        let embed = ConvEmbedding::new(
            config.vocab_size,
            config.in_channels,
            config.out_channels,
            config.norm_eps,
            vs.pp("embed"),
        )?;

        let cond_embed = ConditionEmbedding::new(
            config.condition_input,
            config.hidden_size,
            vs.pp("cond_embed"),
        )?;

        let down_blocks_vs = vs.pp("down_block");
        let down_blocks = (0..1)
            .map(|i| {
                DownsampleBlock::new(
                    config.in_channels,
                    config.out_channels,
                    config.layers_per_block,
                    config.attn_head_dim,
                    config.context_dim,
                    use_flash_attn,
                    config.norm_eps,
                    down_blocks_vs.pp(&i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let project_to_hidden_norm = nn::rms_norm(
            config.out_channels,
            config.norm_eps,
            vs.pp("project_to_hidden_norm"),
        )?;
        let project_to_hidden = nn::linear(
            config.out_channels,
            config.hidden_size,
            vs.pp("project_to_hidden"),
        )?;

        let transformer_layers_vs = vs.pp("transformer_layers");
        let transformer_layers = (0..config.num_hidden_layers)
            .map(|i| {
                TransformerLayer::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.attn_head_dim,
                    config.context_dim,
                    config.norm_eps,
                    false,
                    transformer_layers_vs.pp(&i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let project_from_hidden_norm = nn::rms_norm(
            config.hidden_size,
            config.norm_eps,
            vs.pp("project_from_hidden_norm"),
        )?;
        let project_from_hidden = nn::linear(
            config.hidden_size,
            config.out_channels,
            vs.pp("project_from_hidden"),
        )?;

        let up_blocks_vs = vs.pp("up_blocks");
        let up_blocks = (0..1)
            .map(|i| {
                UpsampleBlock::new(
                    config.in_channels,
                    config.out_channels,
                    config.layers_per_block,
                    config.attn_head_dim,
                    config.context_dim,
                    use_flash_attn,
                    config.norm_eps,
                    up_blocks_vs.pp(&i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mlm_layer = ConvMLMLayer::new(
            config.in_channels,
            config.vocab_size,
            config.out_channels,
            config.norm_eps,
            vs.pp("mlm_layer"),
        )?;

        Ok(Self {
            encoder_proj,
            encoder_proj_layer_norm,
            embed,
            cond_embed,
            down_blocks,
            up_blocks,
            project_to_hidden_norm,
            project_to_hidden,
            transformer_layers,
            project_from_hidden_norm,
            project_from_hidden,
            mlm_layer,
            config,
        })
    }
}
