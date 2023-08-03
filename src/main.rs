use std::env;
use std::fmt::format;
use std::fs::read;
use std::ptr::slice_from_raw_parts;

#[derive(Debug)]
pub struct Config {
    pub dim: u32, // transformer dimension// transformer dimension
    pub hidden_fim: u32, // for ffn layers
    pub n_layers: u32, // number of layers
    pub n_heads: u32, // number of query heads
    pub n_kv_heads: u32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: u32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: u32 // max sequence length
}

impl Config {
    pub fn new(box_data: Box<[u32]>) -> Config {
        let ref_data = box_data.as_ref();
        Config {
            dim: ref_data[0],
            hidden_fim: ref_data[1],
            n_layers: ref_data[2],
            n_heads: ref_data[3],
            n_kv_heads: ref_data[4],
            vocab_size: ref_data[5],
            seq_len: ref_data[6]
        }
    }
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Box<[f32]>,    // (vocab_size * dim)
    // weights for rmsnorms
    rms_att_weight: Box<[f32]>, // (layer * dim) rmsnorm weights
    rms_ffn_weight: Box<[f32]>, // (layer * dim)
    // weights for matmuls
    wq: Box<[f32]>, // (layer * dim * dim)
    wk: Box<[f32]>, // (layer * dim * dim)
    wv: Box<[f32]>, // (layer * dim * dim)
    wo: Box<[f32]>, // (layer * dim * dim)
    // weights for ffn
    w1: Box<[f32]>, // (layer * hidden_dim * dim)
    w2: Box<[f32]>, // (layer * dim * hidden_dim)
    w3: Box<[f32]>, // (layer * hidden_dim * dim)
    // final rmsnorm
    rms_final_weight: Box<[f32]>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Box<[f32]>, // (seq_len * dim/2)
    freq_cis_imag: Box<[f32]>, // (seq_len * dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Option<Box<[f32]>>,
}

fn bytes_to_box<T: Clone>(bytes: &[u8]) -> Result<Box<[T]>, String> {
    let type_size = std::mem::size_of::<T>();
    if bytes.len() % type_size != 0 {
        return Err(format!("Size mismatch, bytearray size {}, type size {}", bytes.len(), type_size));
    }
    let num_elements = bytes.len() / type_size;
    let mut data = Vec::with_capacity(num_elements);
    let ptr = bytes.as_ptr() as *const T;
    let slice = unsafe { &*slice_from_raw_parts(ptr, num_elements)};
    data.extend_from_slice(slice);
    Ok(data.into_boxed_slice())
}

fn read_config(path: &str) {
    let mut vec = read(path).expect("error reading file");
    let config_size = std::mem::size_of::<Config>();
    let config_data: Box<[u32]> = bytes_to_box(&vec[0..config_size]).unwrap();
    let config = Config::new(config_data);
    println!("Config: {:?}", config);
}

fn main() {
    println!("Rust implementation of LLAMA2 inference in C by Andrej Kapathy!");

    //TODO: Get from cmd
    let checkpoint_file = "out/model.bin";
    read_config(checkpoint_file);
}
