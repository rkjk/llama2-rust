use std::fs::read;
use std::rc::Rc;
use std::ptr::slice_from_raw_parts;

#[derive(Debug)]
pub struct Config {
    pub dim: i32, // transformer dimension// transformer dimension
    pub hidden_fim: i32, // for ffn layers
    pub n_layers: i32, // number of layers
    pub n_heads: i32, // number of query heads
    pub n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32 // max sequence length
}

impl Config {
    pub fn new(box_data: Box<[i32]>) -> Config {
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
    token_embedding_table: Rc<Vec<f32>>,    // (vocab_size * dim)
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
    wcls: Rc<Vec<f32>>,
}

impl TransformerWeights {
    pub fn new(bytes: &[u8], config: &Config, shared_weights: bool) -> TransformerWeights {
        let dim: usize = config.dim as usize;
        let hidden_dim = config.hidden_fim as usize;
        let n_layers = config.n_layers as usize;
        let n_heads = config.n_heads as usize;
        let vocab_size = config.vocab_size as usize;
        let seq_len = config.seq_len as usize;
        let mut offset: usize = 0;
        let f32_size = std::mem::size_of::<f32>();
        let token_embedding_table: Rc<Vec<f32>> = Rc::new(bytes_to_box(&bytes[offset..vocab_size * dim * f32_size])
            .expect("Error reading token_embedding_table").to_vec());
        offset += vocab_size * dim * f32_size;

        // Initialize rms_att_weight
        let rms_att_weight: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * f32_size])
            .expect("Error reading rms_att_weight");
        offset += n_layers * dim * f32_size;

        // Initialize wq, wk, wv, wo
        let wq: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * dim * f32_size])
            .expect("Error reading wq");
        offset += n_layers * dim * dim * f32_size;

        let wk: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * dim * f32_size])
            .expect("Error reading wk");
        offset += n_layers * dim * dim * f32_size;

        let wv: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * dim * f32_size])
            .expect("Error reading wv");
        offset += n_layers * dim * dim * f32_size;

        let wo: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * dim * f32_size])
            .expect("Error reading wo");
        offset += n_layers * dim * dim * f32_size;

        // Initialize rms_ffn_weight
        let rms_ffn_weight: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * f32_size])
            .expect("Error reading rms_ffn_weight");
        offset += n_layers * dim * f32_size;

        // Initialize w1, w2, w3
        let w1: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * hidden_dim * dim * f32_size])
            .expect("Error reading w1");
        offset += n_layers * hidden_dim * dim * f32_size;

        let w2: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * dim * hidden_dim * f32_size])
            .expect("Error reading w2");
        offset += n_layers * dim * hidden_dim * f32_size;

        let w3: Box<[f32]> = bytes_to_box(&bytes[offset..offset + n_layers * hidden_dim * dim * f32_size])
            .expect("Error reading w3");
        offset += n_layers * hidden_dim * dim * f32_size;

        // Initialize rms_final_weight
        let rms_final_weight: Box<[f32]> = bytes_to_box(&bytes[offset..offset + dim * f32_size])
            .expect("Error reading rms_final_weight");
        offset += dim * f32_size;
        let head_size = dim / n_heads;
        // Initialize freq_cis_real and freq_cis_imag
        let freq_cis_real: Box<[f32]> = bytes_to_box(&bytes[offset..offset + seq_len * head_size * f32_size / 2])
            .expect("Error reading freq_cis_real");
        offset += seq_len * head_size * f32_size / 2;

        let freq_cis_imag: Box<[f32]> = bytes_to_box(&bytes[offset..offset + seq_len * head_size  * f32_size / 2])
            .expect("Error reading freq_cis_imag");
        offset += seq_len * head_size * f32_size / 2;

        // Initialize wcls if it exists (optional)
        let wcls = match shared_weights {
            true => Rc::clone(&token_embedding_table),
            false => Rc::new(bytes_to_box(&bytes[offset..])
                .expect("Error reading wcls").to_vec())
        };

        assert!(offset == bytes.len());
        println!("offset: {}, bytes len: {}, shared_weights: {}", offset, bytes.len(), shared_weights);

        TransformerWeights {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        }
    }
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

fn read_config(path: &str) -> (Config, TransformerWeights){
    let vec = read(path).expect("error reading file");
    let config_size = std::mem::size_of::<Config>();
    let config_data: Box<[i32]> = bytes_to_box(&vec[0..config_size]).unwrap();
    let config = Config::new(config_data);
    let shared_weights = config.vocab_size > 0;
    config.vocab_size - config.vocab_size.abs();
    let transformer_weights = TransformerWeights::new(&vec[config_size..], &config, shared_weights);
    println!("Config: {:?}", config);
    println!("TransformerWeights size: {}", std::mem::size_of::<TransformerWeights>());
    (config, transformer_weights)
}

fn main() {
    println!("Rust implementation of LLAMA2 inference in C by Andrej Kapathy!");

    //TODO: Get from cmd
    let checkpoint_file = "out/model.bin";
    let (config, transformer_weights) = read_config(checkpoint_file);
}
