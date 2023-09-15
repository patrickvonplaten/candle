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
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn run() -> Result<()> {
    println!("Building the autoencoder.");
    let vae_weights = ModelFile::VQGan.get(None)?;
    println!("{:?}", vae_weights);
    Ok(())
}

fn main() -> Result<()> {
    run()
}
