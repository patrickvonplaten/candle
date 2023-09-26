#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::muse;

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use clap::Parser;
use tokenizers::Tokenizer;

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Clip,
    VQGan,
    UViT,
}

impl ModelFile {
    fn get(&self, filename: Option<String>) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => ("openai/clip-vit-large-patch14", "tokenizer.json"),
                    Self::Clip => ("openai/clip-vit-large-patch14", "model.safetensors"),
                    Self::VQGan => ("openMUSE/vqgan-f16-8192-laion", "model.safetensors"),
                    Self::UViT => ("valhalla/muse-research-run", "ema_model/model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn run() -> Result<()> {
    let device = Device::Cpu;

    let muse_config = muse::MuseConfig::v256(None, None);

    println!("Building the VQGan.");
    let vqgan_weights = ModelFile::VQGan.get(None)?;
    let vqgan = muse_config.build_vqgan(&vqgan_weights, &device, DType::F32);

    println!("Building CLIP.");
    let clip_weights = ModelFile::Clip.get(None)?;
    let clip = muse_config.build_clip_transformer(&clip_weights, &device, DType::F32);

    println!("Building CLIP Tokenizer.");
    let tokenizer_file = ModelFile::Tokenizer.get(None)?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    println!("Building the UViT.");
    let uvit_weights = ModelFile::UViT.get(None)?;
    let uvit = muse_config.build_uvit_transformer(&uvit_weights, &device, DType::F32);
    println!("Hey");
    Ok(())
}

fn main() -> Result<()> {
    run()
}
