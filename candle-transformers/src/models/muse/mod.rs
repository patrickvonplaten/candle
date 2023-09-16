pub mod attention;
pub mod clip;
pub mod embeddings;
pub mod resnet;
pub mod unet_2d;
pub mod unet_2d_blocks;
pub mod utils;
pub mod vqgan;

use candle::{DType, Device, Result};
use candle_nn as nn;

#[derive(Clone, Debug)]
pub struct MuseConfig {
    pub width: usize,
    pub height: usize,
    pub clip: clip::Config,
    vqgan: vqgan::VQGANModelConfig,
}

impl MuseConfig {
    pub fn v256(height: Option<usize>, width: Option<usize>) -> Self {
        let height = if let Some(height) = height {
            assert_eq!(height % 16, 0, "height has to be divisible by 16");
            height
        } else {
            256
        };
        let width = if let Some(width) = width {
            assert_eq!(width % 16, 0, "width has to be divisible by 16");
            width
        } else {
            256
        };
        let vqgan = vqgan::VQGANModelConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };

        Self {
            width,
            height,
            clip: clip::Config::v1_5(),
            vqgan,
        }
    }

    pub fn build_vqgan<P: AsRef<std::path::Path>>(
        &self,
        vae_weights: P,
        device: &Device,
        dtype: DType,
    ) -> Result<vqgan::VQGANModel> {
        let weights = unsafe { candle::safetensors::MmapedFile::new(vae_weights)? };
        let weights = weights.deserialize()?;
        let vs_vqgan = nn::VarBuilder::from_safetensors(vec![weights], dtype, device);

        println!("{:?}", vs_vqgan.state_dict());
        let vqgan = vqgan::VQGANModel::new(vs_vqgan, 3, 3, self.vqgan.clone())?;
        Ok(vqgan)
    }
}
