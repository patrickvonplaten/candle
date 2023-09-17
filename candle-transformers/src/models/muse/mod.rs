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
use nn::var_builder;

#[derive(Clone, Debug)]
pub struct MuseConfig {
    pub width: usize,
    pub height: usize,
    pub clip: clip::Config,
    vqgan_config: vqgan::VQGANModelConfig,
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
        let vqgan_config = vqgan::VQGANModelConfig {
            block_out_channels: vec![768, 512, 256, 256, 128],
            layers_per_block: 2,
            latent_channels: 64,
            norm_num_groups: 32,
        };

        Self {
            width,
            height,
            clip: clip::Config::v1_5(),
            vqgan_config,
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
        let tensor_dict = vs_vqgan.tensors();

        // println!("{:?}", vs_vqgan.tensors());
        // let var_builder = nn::VarBuilder::from_varmap(&nn::VarMap::new(), dtype, device);
        let var_builder = nn::VarBuilder::shapes(dtype, device);
        let vqgan = vqgan::VQGANModel::new(var_builder.clone(), 3, 3, self.vqgan_config.clone())?;

        let var_builder = var_builder.pp("decoder");
        let mut entries: Vec<_> = var_builder.tensors().into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        for (key, value) in entries.iter() {
            match map.get(key) {
                Some(&value2) => {
                    if value2 

                }println!("The value for key '{}' is {}", key, value),
                None => println!("No value found for key '{}'", key),
            }

            if tensor_dict.contains_key(key) and tensor_dict

            } else {
                println!("{}: {:?}", key, value.shape());
            }
        }
        Ok(vqgan)
    }
}
