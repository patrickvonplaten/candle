pub mod attention;
pub mod clip;
pub mod embeddings;
pub mod resnet;
pub mod unet_2d_blocks;
pub mod utils;
pub mod uvit;
pub mod vqgan;

use candle::{DType, Device, Result};
use candle_nn as nn;
use nn::var_builder;

#[derive(Clone, Debug)]
pub struct MuseConfig {
    pub width: usize,
    pub height: usize,
    pub clip_config: clip::Config,
    pub vqgan_config: vqgan::VQGANModelConfig,
    pub uvit_config: uvit::UViTTransformerConfig,
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
            num_embeddings: 8192,
        };
        let uvit_config = uvit::UViTTransformerConfig::default();

        Self {
            width,
            height,
            clip_config: clip::Config::v1_5(),
            vqgan_config,
            uvit_config,
        }
    }

    pub fn build_clip_transformer<P: AsRef<std::path::Path>>(
        &self,
        clip_weights: P,
        device: &Device,
        dtype: DType,
    ) -> Result<clip::ClipTextTransformer> {
        let weights = unsafe { candle::safetensors::MmapedFile::new(clip_weights)? };
        let weights = weights.deserialize()?;
        let vs = nn::VarBuilder::from_safetensors(vec![weights], dtype, device);
        let text_model = clip::ClipTextTransformer::new(vs, &self.clip_config.clone())?;
        Ok(text_model)
    }

    pub fn build_uvit_transformer<P: AsRef<std::path::Path>>(
        &self,
        uvit_weights: P,
        device: &Device,
        dtype: DType,
    ) -> Result<uvit::UViTTransformer> {
        let weights = unsafe { candle::safetensors::MmapedFile::new(uvit_weights)? };
        let weights = weights.deserialize()?;
        let vs = nn::VarBuilder::from_safetensors(vec![weights], dtype, device);
        let uvit_model = uvit::UViTTransformer::new(self.uvit_config.clone(), vs.clone())?;
        Ok(uvit_model)
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
        let vqgan = vqgan::VQGANModel::new(vs_vqgan.clone(), 3, 3, self.vqgan_config.clone())?;
        Ok(vqgan)
        // // println!("{:?}", vs_vqgan.tensors());
        // // let var_builder = nn::VarBuilder::from_varmap(&nn::VarMap::new(), dtype, device);
        // let var_builder = nn::VarBuilder::shapes(dtype, device);
        // let vqgan = vqgan::VQGANModel::new(var_builder.clone(), 3, 3, self.vqgan_config.clone())?;

        // let tensor_dict_new = var_builder.tensors();

        // for (key, value) in entries.iter() {
        //     match tensor_dict_new.get(key) {
        //         Some(&ref value2) => {
        //             if value2.shape() != value.shape() {
        //                 println!(
        //                     "For {} original model has shape {:?}, but new model has {:?}",
        //                     key,
        //                     value.shape(),
        //                     value2.shape()
        //                 );
        //             }
        //         }
        //         None => println!("No value found for key '{}'", key),
        //     }
        // }
        // let mut entries: Vec<_> = var_builder.tensors().into_iter().collect();
        // entries.sort_by(|a, b| a.0.cmp(&b.0));

        // for (key, value) in entries.iter() {
        //     match vs_vqgan.tensors().get(key) {
        //         Some(&ref value2) => {
        //             if value2.shape() != value.shape() {
        //                 println!(
        //                     "For {} original model has shape {:?}, but new model has {:?}",
        //                     key,
        //                     value.shape(),
        //                     value2.shape()
        //                 );
        //             }
        //         }
        //         None => println!("No value found for key '{}'", key),
        //     }
        // }
        // Ok(vqgan)
    }
}
