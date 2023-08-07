
use std::fs::read;
use std::io;
use std::io::Write;

use std::ptr;
use std::rc::Rc;
use std::ptr::slice_from_raw_parts;


#[derive(Debug, Copy, Clone)]
pub struct Config {
    pub dim: i32, // transformer dimension// transformer dimension
    pub hidden_dim: i32, // for ffn layers
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
            hidden_dim: ref_data[1],
            n_layers: ref_data[2],
            n_heads: ref_data[3],
            n_kv_heads: ref_data[4],
            vocab_size: ref_data[5],
            seq_len: ref_data[6]
        }
    }
}

pub struct TransformerWeights {
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
        let hidden_dim = config.hidden_dim as usize;
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

pub struct RunState {
    x: Box<[f32]>,
    // activation at current time stamp (dim,)
    xb: Box<[f32]>,
    // same, but inside a residual branch (dim,)
    xb2: Box<[f32]>,
    // an additional buffer just for convenience (dim,)
    hb: Box<[f32]>,
    // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Box<[f32]>,
    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Box<[f32]>,
    // query (dim,)
    k: Box<[f32]>,
    // key (dim,)
    v: Box<[f32]>,
    // value (dim,)
    att: Box<[f32]>,
    // buffer for scores/attention values (n_heads, seq_len)
    logits: Box<[f32]>,
    // output logits

    // kv cache
    key_cache: Box<[f32]>,
    // (layer, seq_len, dim)
    value_cache: Box<[f32]>, // (layer, seq_len, dim)},
    config: Config,
    transformer_weights: TransformerWeights
}

impl RunState {
    pub fn new(config: &Config, transformer_weights: TransformerWeights) -> RunState {
        RunState {
            x: vec![0.0; config.dim as usize].into_boxed_slice(),
            xb: vec![0.0; config.dim as usize].into_boxed_slice(),
            xb2: vec![0.0; config.dim as usize].into_boxed_slice(),
            hb: vec![0.0; config.hidden_dim as usize].into_boxed_slice(),
            hb2: vec![0.0; config.hidden_dim as usize].into_boxed_slice(),
            q: vec![0.0; config.dim as usize].into_boxed_slice(),
            k: vec![0.0; config.dim as usize].into_boxed_slice(),
            v: vec![0.0; config.dim as usize].into_boxed_slice(),
            att: vec![0.0; (config.n_heads * config.seq_len) as usize].into_boxed_slice(),
            logits: vec![0.0; config.vocab_size as usize].into_boxed_slice(),
            key_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim) as usize].into_boxed_slice(),
            value_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim) as usize].into_boxed_slice(),
            config: config.clone(),
            transformer_weights: transformer_weights,
        }
    }

    fn transformer(&mut self, token: usize, pos: usize) {
        let dim = self.config.dim as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let head_size = dim / self.config.n_heads as usize;
        let seq_len: usize = self.config.seq_len as usize;
        let n_layers = self.config.n_layers as usize;
        let n_heads = self.config.n_heads as usize;

        // copy the token embedding into x
        let offset = token * dim;
        self.x.copy_from_slice(&self.transformer_weights.token_embedding_table[offset..offset + dim]);
        //println!("Activation at start: {:?}", self.x);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        let freq_cis_real_row: &[f32] = &self.transformer_weights.freq_cis_real[(pos * head_size / 2)..((pos + 1) * head_size / 2)];
        let freq_cis_imag_row: &[f32] = &self.transformer_weights.freq_cis_imag[(pos * head_size / 2)..((pos + 1) * head_size / 2)];

        // forward all the layers
        for l in 0..n_layers {
            // attention rmsnorm
            rmsnorm(
                self.xb.as_mut_ptr(),
                self.x.as_ref(),
                self.transformer_weights.rms_att_weight[l * dim..(l + 1) * dim].as_ref());
            //println!("xb for layer {} -> {:?}", l, self.xb);
            // qkv matmuls for this position
            matmul(
                self.q.as_mut_ptr(),
                self.xb.as_ref(),
                self.transformer_weights.wq[(l * dim * dim)..((l + 1) * dim * dim)].as_ref(),
            dim,
            dim);
            //println!("Query for layer {} -> {:?}", l, self.q);
            matmul(
                self.k.as_mut_ptr(),
                self.xb.as_ref(),
                self.transformer_weights.wk[(l * dim * dim)..((l + 1) * dim * dim)].as_ref(),
                dim,
                dim
            );
            //println!("Key for layer {} -> {:?}", l, self.k);
            matmul(
                self.v.as_mut_ptr(),
                self.xb.as_ref(),
                self.transformer_weights.wv[(l * dim * dim)..((l + 1) * dim * dim)].as_ref(),
                dim,
                dim
            );
            //println!("Value for layer {} -> {:?}", l, self.v);
            // apply RoPE rotation to the q and k vectors for each head
            for h in 0..n_heads as usize {
                // get the q and k vectors for this head
                let q: &mut [f32] = self.q[(h * head_size)..((h + 1) * head_size)].as_mut();
                let k: &mut [f32] = self.k[(h * head_size)..((h + 1) * head_size)].as_mut();

                for i in (0..head_size).step_by(2) {
                    let q0 = q[i];
                    let q1 = q[i + 1];
                    let k0 = k[i];
                    let k1 = k[i + 1];
                    let fcr = freq_cis_real_row[i/2];
                    let fci = freq_cis_imag_row[i/2];
                    q[i] = q0 * fcr - q1 * fci;
                    q[i+1] = q0 * fci + q1 * fcr;
                    k[i] = k0 * fcr - k1 * fci;
                    k[i+1] = k0 * fci + k1 * fcr;
                }
            }
            //println!("Query after  RoPE for layer {} -> {:?}", l, self.q);
            //println!("Value after RoPE for layer {} -> {:?}", l, self.k);
            // save key,value at this time step (pos) to our kv cache
            let loff: usize = l * seq_len * dim; // kv cache layer offset for convenience
            let nloff = loff + pos * dim;
            self.key_cache[nloff..nloff + dim].copy_from_slice(self.k.as_ref());
            self.value_cache[nloff..nloff + dim].copy_from_slice(self.v.as_ref());
            //println!("Key cache at layer {} -> {:?}", l, &self.key_cache[nloff..nloff + dim]);
            //println!("Value cache at layer {} -> {:?}", l, &self.value_cache[nloff..nloff + dim]);

            // multihead attention. iterate over all heads
            // TODO: Make parallel
            for h in 0..n_heads {
                // get the query vector for this head
                let q: &[f32] = &self.q[h * head_size..(h + 1) * head_size];
                // attention scores for this head
                // TODO: Check length = pos + 1
                let att: &mut [f32] = &mut self.att[(h * seq_len)..(h * seq_len + pos + 1)];
                // iterate over all timesteps, including the current one
                for t in 0..pos+1 {
                    // get the key vector for this head and at this timestep
                    let offset = loff + t * dim;
                    let k = &self.key_cache[(offset + h * head_size)..(offset + (h + 1) * head_size)];
                    // calculate the attention score as the dot product of q and k
                    let mut score: f32 = 0.0;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }
                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att);
                //println!("Attention after softmax for head {} -> {:?}", h, att);

                // weighted sum of the values, store back into xb
                let xb = &mut self.xb[h * head_size..((h + 1) * head_size)];
                xb.fill(0.0);
                for t in 0..pos+1 {
                    // get the value vector for this head and at this timestep
                    let o = loff + t * dim;
                    let v = &self.value_cache[o + h * head_size..(o + (h + 1) * head_size)];
                    // get the attention weight for this timestep
                    let a = att[t];
                    // accumulate the weighted value into xb
                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            }
            //println!("xb after multi-head layer {} -> {:?}", l, self.xb);
            // final matmul to get the output of the attention
            matmul(
                self.xb2.as_mut_ptr(),
                self.xb.as_ref(),
                &self.transformer_weights.wo[l * dim * dim..((l + 1) * dim * dim)],
                dim,
                dim
            );
            //println!("xb2 after multi-head layer {} -> {:?}", l, self.xb2);
            // residual connection back into x
            accum(self.x.as_mut(), self.xb2.as_ref());
            //println!("Activation after residual conn layer {} -> {:?}", l, self.x);

            // ffn rmsnorm
            rmsnorm(
                self.xb.as_mut_ptr(),
                self.x.as_ref(),
                &self.transformer_weights.rms_ffn_weight[l * dim..((l + 1) * dim)]
            );
            //println!("xb after ffn rmsnorm layer {} -> {:?}", l, self.xb);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(
                self.hb.as_mut_ptr(),
                self.xb.as_ref(),
                &self.transformer_weights.w1[l * dim * hidden_dim..((l + 1) * dim * hidden_dim)],
                dim,
                hidden_dim
            );
            //println!("hb at layer {} -> {:?}", l, self.hb);
            matmul(
                self.hb2.as_mut_ptr(),
                self.xb.as_ref(),
                &self.transformer_weights.w3[l * dim * hidden_dim..((l + 1) * dim * hidden_dim)],
                dim,
                hidden_dim
            );
            //println!("hb2 at layer {} -> {:?}", l, self.hb2);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for i in 0..hidden_dim {
                self.hb[i] = self.hb[i] * (1.0 / (1.0 + (-1.0 * self.hb[i]).exp()));
            }

            // elementwise multiply with w3(x)
            for i in 0..hidden_dim {
                self.hb[i] *= self.hb2[i];
            }
            //println!("hb after elemenwise ops at layer {} -> {:?}", l, self.hb);
            //println!("w2 here {:?}", &self.transformer_weights.w2[l * dim * hidden_dim..((l + 1) * dim * hidden_dim)]);
            // final matmul to get the output of the ffn
            matmul(
                self.xb.as_mut_ptr(),
                self.hb.as_ref(),
                &self.transformer_weights.w2[l * dim * hidden_dim..((l + 1) * dim * hidden_dim)],
                hidden_dim,
                dim
            );
            //println!("xb after final matmul at layer {} -> {:?}", l, self.xb);
            // residual connection
            accum(self.x.as_mut(), self.xb.as_ref());
        }
        // final rmsnorm
        rmsnorm(self.x.as_mut_ptr(), self.x.as_ref(), &self.transformer_weights.rms_final_weight);

        // classifier into logits
        matmul(
            self.logits.as_mut_ptr(),
            self.x.as_ref(),
            self.transformer_weights.wcls.as_slice(),
            dim,
            self.config.vocab_size as usize
        );
        //println!("x: {:?}", self.x);
        //println!("wcls: {:?}", self.transformer_weights.wcls);
        //println!("logits: {:?}", self.logits);
    }
}

#[derive(Debug)]
pub struct Token {
    score: f32,
    token: String
}

impl Token {
    pub fn new(score: f32, token: String) -> Token {
        Token {
            score,
            token
        }
    }
}

#[derive(Debug)]
pub struct Tokenizer {
    toks: Box<[Token]>
}

impl Tokenizer {
    pub fn new(toks: Vec<Token>) -> Tokenizer {
        Tokenizer {
            toks: toks.into_boxed_slice()
        }
    }
}

fn bpe_encode(prompt: &str, tokenizer: &Tokenizer) -> Result<Vec<usize>, String> {
    let _tokens: Vec<usize> = Vec::with_capacity(prompt.len());
    let toks = tokenizer.toks.as_ref();
    let str_lookup = |c: String| -> Option<usize> {
        let c_string = c.to_string();
        for i in 0..toks.len() {
            if toks[i].token == c_string {
                //println!("Found match: {}", toks[i].token);
                return Some(i);
            }
        }
        None
    };
    // encode every byte in the prompt
    let mut tokens: Vec<usize> = vec![];
    for c in prompt.chars() {
        if let Some(v) =  str_lookup(c.to_string()) {
            //println!("Tokenizing {} to {}", c, &*tokenizer.toks[v].token);
            tokens.push(v);
        } else {
            return Err(format!("Could not tokenize {}", c));
        }
    }
    loop {
        let mut best_score: f32 = -1e10;
        let mut best_id: usize = usize::MAX;
        let mut best_idx: usize = usize::MAX;
        for i in 0..tokens.len() - 1 {
            let idx = tokens[i];
            let nex_idx = tokens[i + 1];
            let merged = toks[idx].token.clone() + toks[nex_idx].token.as_str();
            if let Some(v) = str_lookup(merged) {
                let score = toks[v].score;
                if score > best_score {
                    best_score = score;
                    best_id = v;
                    best_idx = i;
                }
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        tokens[best_idx] = best_id;
        for i in best_idx + 1..tokens.len() - 1 {
            tokens[i] = tokens[i + 1];
        }
        tokens.pop();
    }
    Ok(tokens)
}

fn accum(a: &mut [f32], b: &[f32]) {
    assert!(a.len() == b.len(), "a and b should be equal length slices");
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn rmsnorm(out: *mut f32, x: &[f32], w: &[f32]) {
    assert!(x.len() == w.len(), "out and x should be equal length slices");
    let mut ss: f32 = 0.0;
    for i in 0..x.len() {
        ss += x[i] * x[i];
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    unsafe {
        for i in 0..x.len() {
            *out.offset(i as isize) = x[i] * w[i] * ss;
        }
    }
}

fn softmax(x: &mut[f32]) {
    let mut max_val = x[0];
    for i in 1..x.len() {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    let mut sum: f32 = 0.0;
    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    for i in 0..x.len() {
        x[i] /= sum;
    }
}

fn matmul(xout: *mut f32, x: &[f32], w: &[f32], n: usize, d: usize) {
    // Multiply W (d, n) * X(n, 1) and store in xout (d, 1)
    /*
    unsafe {
        sgemm(
            d,
            n,
            1,
            1.0,
            w.as_ptr(),
            d as isize,
            1,
            x.as_ptr(),
            n as isize,
            1,
            0.0,
            xout,
            1,
            1);
    }
     */
    for i in 0..d {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        unsafe {
            *xout.offset(i as isize) = val;
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut max_val = v[0];
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i].ge(&max_val) {
            max_val = v[i];
            max_idx = i;
        }
    }
    max_idx
}

fn bytes_to_box<T: Clone>(bytes: &[u8]) -> Result<Box<[T]>, String> {
    let type_size = std::mem::size_of::<T>();
    if bytes.len() % type_size != 0 {
        return Err(format!("Size mismatch, bytearray size {}, type size {}", bytes.len(), type_size));
    }
    let num_elements = bytes.len() / type_size;
    //println!("num_elements: {}", num_elements);
    let mut data = Vec::with_capacity(num_elements);
    let ptr = bytes.as_ptr() as *const T;
    let slice = unsafe { &*slice_from_raw_parts(ptr, num_elements)};
    data.extend_from_slice(slice);
    Ok(data.into_boxed_slice())
}

fn unaligned_bytes_to_box<T: Clone>(bytes: &[u8]) -> Result<Box<[T]>, String> {
    let type_size = std::mem::size_of::<T>();
    if bytes.len() % type_size != 0 {
        return Err(format!("Size mismatch, bytearray size {}, type size {}", bytes.len(), type_size));
    }
    let num_elements = bytes.len() / type_size;
    let mut data = Vec::with_capacity(num_elements);
    let ptr = bytes.as_ptr() as *const T;
    unsafe {
        for _ in 0..num_elements {
            data.push(ptr::read_unaligned(ptr));
        }
    }
    Ok(data.into_boxed_slice())
}

fn read_config(path: &str) -> (Config, TransformerWeights){
    let vec = read(path).expect("error reading model");
    let config_size = std::mem::size_of::<Config>();
    let config_data: Box<[i32]> = bytes_to_box(&vec[0..config_size]).unwrap();
    let mut config = Config::new(config_data);
    let shared_weights = config.vocab_size > 0;
    config.vocab_size = config.vocab_size.abs();
    let transformer_weights = TransformerWeights::new(&vec[config_size..], &config, shared_weights);
    (config, transformer_weights)
}

fn read_tokenizer(path: &str, vocab_size: i32) -> (u32, Tokenizer) {
    let f32_size = std::mem::size_of::<f32>();
    let u32_size = std::mem::size_of::<u32>();
    let vec = read(path).expect("error reading tokenizer");
    let mut tokens: Vec<Token> = Vec::new();
    let mut offset: usize = 0; // byte offset
    let max_token_size: u32 = bytes_to_box::<u32>(&vec[offset..offset + u32_size]).unwrap()[0];
    offset += u32_size;

    for _i in 0..vocab_size as usize {
        let token_score: f32 = unaligned_bytes_to_box::<f32>(&vec[offset..offset + f32_size]).unwrap()[0];
        offset += f32_size;
        let num_chars: u32 = unaligned_bytes_to_box::<u32>(&vec[offset..offset + u32_size]).unwrap()[0];
        offset += u32_size;
        let str = std::str::from_utf8(&vec[offset..offset + num_chars as usize]).unwrap();
        let token = Token::new(token_score, str.to_owned());
        tokens.push(token);
        offset += num_chars as usize;
    }
    (max_token_size, Tokenizer::new(tokens))
}

fn main() {
    println!("Rust implementation of LLAMA2 inference in C by Andrej Kapathy!");

    //TODO: Get from cmd
    let checkpoint_file = "out/model.bin";
    let tokenizer_file = "tokenizer.bin";
    let (config, transformer_weights) = read_config(checkpoint_file);
    let (max_token_size, tokenizer) = read_tokenizer(tokenizer_file, config.vocab_size);
    println!("Config: {:?}", config);
    println!("Max token size: {}", max_token_size);

    //TODO: Get from cmd
    //let steps = config.seq_len as usize;
    let steps = config.seq_len as usize;
    let mut runstate = RunState::new(&config, transformer_weights);

    // TODO: Get from cmd
    let prompt = "Hello, my name is Raghav. Who are you?";
    let prompt_tokens = bpe_encode(prompt, &tokenizer)
        .expect("Could not encode provided prompt");

    // TODO: Get from cmd
    let temperature: f32 = 0.0;
    let tokens = &*tokenizer.toks;
    let mut cur_token_idx: usize = 1;
    let mut next: usize = 0;
    for pos in 0..steps {
        runstate.transformer(cur_token_idx, pos);
        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            if temperature == 0.0 {
                //println!("logits at 6754: {}", runstate.logits[6574]);
                next = argmax(&runstate.logits);
                //println!("Next: {}", next);
            } else {

            }
        }
        let nex_tok = match cur_token_idx == 1 && tokens[next].token.starts_with(' ') {
            true => tokens[next].token[1..].to_string(),
            false => tokens[next].token.to_string()
        };
        print!("{}", nex_tok);
        io::stdout().flush().unwrap();
        cur_token_idx = next;
    }
    println!();
}
