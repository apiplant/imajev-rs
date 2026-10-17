use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, ops};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use super::gdn;

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

struct Projection {
    weight: Tensor,
    bias: Option<Tensor>,
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
    lora_scale: f64,
}

impl Projection {
    fn new(base: VarBuilder, adapter: Option<VarBuilder>, input: usize, output: usize) -> Result<Self> {
        let weight = base.get((output, input), "weight")?;
        let bias = if base.contains_tensor("bias") { Some(base.get(output, "bias")?) } else { None };
        let (lora_a, lora_b, lora_scale) = if let Some(adapter) = adapter {
            if adapter.contains_tensor("lora_A.weight") {
                let a = adapter.get((64, input), "lora_A.weight")?.to_dtype(DType::F32)?;
                let b = adapter.get((output, 64), "lora_B.weight")?.to_dtype(DType::F32)?;
                (Some(a), Some(b), 2.0) // Imajev-4B: alpha 128 / rank 64
            } else {
                (None, None, 0.0)
            }
        } else {
            (None, None, 0.0)
        };
        Ok(Self { weight, bias, lora_a, lora_b, lora_scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_base = x.to_dtype(self.weight.dtype())?;
        let mut y = x_base.matmul(&self.weight.t()?)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.to_dtype(y.dtype())?)?;
        }
        y = y.to_dtype(DType::F32)?;
        if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let low_rank = x.to_dtype(DType::F32)?.matmul(&a.t()?)?;
            let update = low_rank.matmul(&b.t()?)? * self.lora_scale;
            y = (y + update?)?;
        }
        Ok(y)
    }
}

struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        #[cfg(feature = "cuda")]
        let activated = if x.device().is_cuda() { gdn::swiglu(&gate, &up)? } else { (gate.silu()? * up)? };
        #[cfg(not(feature = "cuda"))]
        let activated = (gate.silu()? * up)?;
        self.down.forward(&activated)
    }
}

struct DeltaNet {
    qkv: Projection,
    gate_z: Projection,
    beta: Projection,
    alpha: Projection,
    dt_bias: Tensor,
    a: Tensor,
    conv_w: Tensor,
    norm: Tensor,
    out: Projection,
    conv_state: Option<Tensor>,
    state: Option<Tensor>,
}

struct Attention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Tensor,
    k_norm: Tensor,
}

enum Mixer {
    Linear(DeltaNet),
    Full(Attention),
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    mlp: Mlp,
}

pub struct Qwen35Text {
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    config: TextConfig,
    device: Device,
}

impl Qwen35Text {
    pub fn new(base_root: &VarBuilder, adapter_root: &VarBuilder, config: TextConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let text = base_root.pp("model").pp("language_model");
        let adapter_text = adapter_root.pp("base_model").pp("model").pp("model").pp("language_model");
        let embed = text.pp("embed_tokens").get((config.vocab_size, hidden), "weight")?.to_dtype(DType::F32)?;
        let norm = (text.pp("norm").get(hidden, "weight")?.to_dtype(DType::F32)? + 1.0)?;
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_key_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for index in 0..config.num_hidden_layers {
            let base_layer = text.pp("layers").pp(index);
            let adapter_layer = adapter_text.pp("layers").pp(index);
            let mlp_base = base_layer.pp("mlp");
            let mlp_adapter = adapter_layer.pp("mlp");
            let mlp = Mlp {
                gate: Projection::new(mlp_base.pp("gate_proj"), Some(mlp_adapter.pp("gate_proj")), hidden, config.intermediate_size)?,
                up: Projection::new(mlp_base.pp("up_proj"), Some(mlp_adapter.pp("up_proj")), hidden, config.intermediate_size)?,
                down: Projection::new(mlp_base.pp("down_proj"), Some(mlp_adapter.pp("down_proj")), config.intermediate_size, hidden)?,
            };
            let mixer = match config.layer_types.get(index).map(String::as_str) {
                Some("linear_attention") => {
                    let linear = base_layer.pp("linear_attn");
                    let linear_adapter = adapter_layer.pp("linear_attn");
                    let conv = linear.pp("conv1d").get((conv_dim, 1, config.linear_conv_kernel_dim), "weight")?
                        .reshape((conv_dim, config.linear_conv_kernel_dim))?.to_dtype(DType::F32)?;
                    let a_log = linear.get(config.linear_num_value_heads, "A_log")?.to_dtype(DType::F32)?;
                    Mixer::Linear(DeltaNet {
                        qkv: Projection::new(linear.pp("in_proj_qkv"), Some(linear_adapter.pp("in_proj_qkv")), hidden, conv_dim)?,
                        gate_z: Projection::new(linear.pp("in_proj_z"), Some(linear_adapter.pp("in_proj_z")), hidden, value_dim)?,
                        beta: Projection::new(linear.pp("in_proj_b"), None, hidden, config.linear_num_value_heads)?,
                        alpha: Projection::new(linear.pp("in_proj_a"), None, hidden, config.linear_num_value_heads)?,
                        dt_bias: linear.get(config.linear_num_value_heads, "dt_bias")?.to_dtype(DType::F32)?,
                        a: a_log.exp()?.neg()?,
                        conv_w: conv,
                        norm: linear.pp("norm").get(config.linear_key_head_dim, "weight")?.to_dtype(DType::F32)?,
                        out: Projection::new(linear.pp("out_proj"), Some(linear_adapter.pp("out_proj")), value_dim, hidden)?,
                        conv_state: None,
                        state: None,
                    })
                }
                Some("full_attention") => {
                    let attention = base_layer.pp("self_attn");
                    let adapter_attention = adapter_layer.pp("self_attn");
                    let n_heads = config.num_attention_heads;
                    let n_kv_heads = config.num_key_value_heads;
                    let head_dim = config.head_dim;
                    Mixer::Full(Attention {
                        q: Projection::new(attention.pp("q_proj"), Some(adapter_attention.pp("q_proj")), hidden, n_heads * 2 * head_dim)?,
                        k: Projection::new(attention.pp("k_proj"), Some(adapter_attention.pp("k_proj")), hidden, n_kv_heads * head_dim)?,
                        v: Projection::new(attention.pp("v_proj"), Some(adapter_attention.pp("v_proj")), hidden, n_kv_heads * head_dim)?,
                        o: Projection::new(attention.pp("o_proj"), Some(adapter_attention.pp("o_proj")), n_heads * head_dim, hidden)?,
                        q_norm: (attention.pp("q_norm").get(head_dim, "weight")?.to_dtype(DType::F32)? + 1.0)?,
                        k_norm: (attention.pp("k_norm").get(head_dim, "weight")?.to_dtype(DType::F32)? + 1.0)?,
                    })
                }
                other => bail!("unknown Qwen3.5 layer type at {index}: {other:?}"),
            };
            layers.push(Layer {
                input_norm: (base_layer.pp("input_layernorm").get(hidden, "weight")?.to_dtype(DType::F32)? + 1.0)?,
                post_norm: (base_layer.pp("post_attention_layernorm").get(hidden, "weight")?.to_dtype(DType::F32)? + 1.0)?,
                mixer,
                mlp,
            });
        }
        Ok(Self { embed, layers, norm, config, device: base_root.device().clone() })
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.device)?;
        Ok(self.embed.index_select(&ids, 0)?)
    }

    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            if let Mixer::Linear(mixer) = &mut layer.mixer {
                mixer.conv_state = None;
                mixer.state = None;
            }
        }
    }

    pub fn forward_hidden(&mut self, mut h: Tensor, positions: &[[usize; 3]]) -> Result<Tensor> {
        let (seq, hidden) = h.dims2()?;
        ensure!(hidden == self.config.hidden_size, "unexpected input embedding width {hidden}");
        ensure!(seq > 0 && seq <= 4096, "processed page request must contain 1 to 4096 tokens (found {seq})");
        ensure!(positions.len() == seq, "expected one rotary position per token");
        h = h.to_dtype(DType::F32)?;
        let (cos, sin) = self.rope_tables(positions)?;
        let mask = self.causal_mask(seq)?;
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let params = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor).round() as usize,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
        );
        for layer in &mut self.layers {
            let x = rms(&h, &layer.input_norm, eps)?;
            let mixed = match &mut layer.mixer {
                Mixer::Linear(m) => delta_net(m, &x, eps, params)?,
                Mixer::Full(m) => attention(m, &x, &cos, &sin, &mask, eps, params)?,
            };
            h = (h + mixed)?;
            let x = rms(&h, &layer.post_norm, eps)?;
            h = (h + layer.mlp.forward(&x)?)?;
        }
        let last = h.narrow(0, seq - 1, 1)?;
        Ok(rms(&last, &self.norm, eps)?.squeeze(0)?)
    }

    fn rope_tables(&self, positions: &[[usize; 3]]) -> Result<(Tensor, Tensor)> {
        let n_rot = (self.config.head_dim as f64 * self.config.rope_parameters.partial_rotary_factor).round() as usize;
        let half = n_rot / 2;
        let inv: Vec<f32> = (0..half).map(|i| {
            1.0 / self.config.rope_parameters.rope_theta.powf(2.0 * i as f64 / n_rot as f64) as f32
        }).collect();
        let sections = if self.config.rope_parameters.mrope_section.is_empty() {
            vec![half / 3, half / 3, half - 2 * (half / 3)]
        } else {
            self.config.rope_parameters.mrope_section.clone()
        };
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for position in positions {
            for (i, frequency) in inv.iter().enumerate() {
                let axis = if self.config.rope_parameters.mrope_interleaved {
                    if i % 3 == 1 && i < sections[1] * 3 { 1 }
                    else if i % 3 == 2 && i < sections[2] * 3 { 2 }
                    else { 0 }
                } else {
                    // Non-interleaved Qwen M-RoPE repeats the configured axis chunks.
                    let h_end = sections[1] * 2;
                    let w_end = h_end + sections[2] * 2;
                    if i < h_end { 1 } else if i < w_end { 2 } else { 0 }
                };
                let angle = position[axis] as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (positions.len(), half), &self.device)?,
            Tensor::from_vec(sin, (positions.len(), half), &self.device)?,
        ))
    }

    fn causal_mask(&self, seq: usize) -> Result<Tensor> {
        let values: Vec<f32> = (0..seq).flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
        Ok(Tensor::from_vec(values, (seq, seq), &self.device)?)
    }
}

fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(ops::rms_norm(&x.contiguous()?, weight, eps as f32)?)
}

type LayerParams = (usize, usize, usize, usize, usize, usize, usize);

fn attention(a: &mut Attention, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor, eps: f64, cfg: LayerParams) -> Result<Tensor> {
    let (n_head, n_kv, head_dim, n_rot, ..) = cfg;
    let len = x.dim(0)?;
    let qg = a.q.forward(x)?.reshape((len, n_head, 2 * head_dim))?;
    let q = qg.narrow(2, 0, head_dim)?;
    let gate = qg.narrow(2, head_dim, head_dim)?.reshape((len, n_head * head_dim))?;
    let q = rms(&q, &a.q_norm, eps)?;
    let k = rms(&a.k.forward(x)?.reshape((len, n_kv, head_dim))?, &a.k_norm, eps)?;
    let v = a.v.forward(x)?.reshape((len, n_kv, head_dim))?;
    let rope = |t: &Tensor| -> Result<Tensor> {
        let t = t.transpose(0, 1)?.unsqueeze(0)?;
        let rotated = candle_nn::rotary_emb::rope(&t.narrow(3, 0, n_rot)?.contiguous()?, cos, sin)?;
        Ok(Tensor::cat(&[rotated, t.narrow(3, n_rot, head_dim - n_rot)?], 3)?)
    };
    let q = rope(&q)?;
    let k = rope(&k)?;
    let v = v.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    let repeat = n_head / n_kv;
    let q = q.reshape((1, n_kv, repeat * len, head_dim))?.contiguous()?;
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (head_dim as f64).sqrt()))?;
    let scores = scores.reshape((1, n_head, len, len))?.broadcast_add(mask)?.reshape((1, n_kv, repeat * len, len))?;
    let probs = ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v)?.reshape((1, n_head, len, head_dim))?.squeeze(0)?.transpose(0, 1)?.reshape((len, n_head * head_dim))?;
    a.o.forward(&(out * ops::sigmoid(&gate)?)?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok((x.relu()? + x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?)
}

fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.clamp(eps, f64::INFINITY)?;
    Ok(x.broadcast_div(&norm)?)
}

fn delta_net(m: &mut DeltaNet, x: &Tensor, eps: f64, cfg: LayerParams) -> Result<Tensor> {
    let (.., key_heads, value_heads, dim) = cfg;
    let len = x.dim(0)?;
    let key_dim = key_heads * dim;
    let value_dim = value_heads * dim;
    let conv_dim = 2 * key_dim + value_dim;
    let kernel = m.conv_w.dim(1)?;
    let qkv = m.qkv.forward(x)?;
    let z = m.gate_z.forward(x)?;
    let beta_alpha = Tensor::cat(&[&m.beta.forward(x)?, &m.alpha.forward(x)?], 1)?;
    let prev = match &m.conv_state {
        Some(state) => state.clone(),
        None => Tensor::zeros((conv_dim, kernel - 1), DType::F32, x.device())?,
    };

    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && dim == gdn::D {
        let (conv, next) = gdn::conv(&qkv, &prev, &m.conv_w)?;
        m.conv_state = Some(next);
        let qk = gdn::qk_norm(&conv, key_heads, eps)?;
        let (beta, gate) = gdn::gates(&beta_alpha, &m.dt_bias, &m.a, value_heads)?;
        let state = match &m.state {
            Some(state) => state.clone(),
            None => Tensor::zeros((value_heads, dim, dim), DType::F32, x.device())?,
        };
        let (out, next) = gdn::run(&qk, &conv, &gate, &beta, &state, value_heads, key_heads, 2 * key_dim)?;
        m.state = Some(next);
        return finish(m, out, &z, eps, len, value_heads, dim);
    }

    let beta = ops::sigmoid(&beta_alpha.narrow(1, 0, value_heads)?)?;
    let gate = softplus(&beta_alpha.narrow(1, value_heads, value_heads)?.broadcast_add(&m.dt_bias)?)?.broadcast_mul(&m.a)?;
    let seq = Tensor::cat(&[&prev, &qkv.t()?], 1)?.contiguous()?;
    m.conv_state = Some(seq.narrow(1, len, kernel - 1)?.contiguous()?);
    let mut conv = seq.narrow(1, 0, len)?.broadcast_mul(&m.conv_w.narrow(1, 0, 1)?)?;
    for i in 1..kernel {
        conv = (conv + seq.narrow(1, i, len)?.broadcast_mul(&m.conv_w.narrow(1, i, 1)?)?)?;
    }
    let conv = conv.silu()?.t()?.contiguous()?;
    let q = l2_norm(&conv.narrow(1, 0, key_dim)?.reshape((len, key_heads, dim))?, eps)?;
    let k = l2_norm(&conv.narrow(1, key_dim, key_dim)?.reshape((len, key_heads, dim))?, eps)?;
    let v = conv.narrow(1, 2 * key_dim, value_dim)?.reshape((len, value_heads, dim))?;
    let q = (q * (1.0 / (dim as f64).sqrt()))?;
    // Value head h reads key head h / (value_heads / key_heads): repeat_interleave, not a tiled repeat.
    let repeat = value_heads / key_heads;
    let widen = |t: &Tensor| -> Result<Tensor> { Ok(t.unsqueeze(2)?.expand((len, key_heads, repeat, dim))?.reshape((len, value_heads, dim))?) };
    let q = widen(&q)?;
    let k = widen(&k)?;
    let decay = gate.exp()?;
    let mut state = match &m.state {
        Some(state) => state.clone(),
        None => Tensor::zeros((value_heads, dim, dim), DType::F32, x.device())?,
    };
    let mut outs = Vec::with_capacity(len);
    for t in 0..len {
        let qt = q.get(t)?.unsqueeze(2)?;
        let kt = k.get(t)?.unsqueeze(2)?;
        let vt = v.get(t)?.unsqueeze(2)?;
        let beta_t = beta.get(t)?.reshape((value_heads, 1, 1))?;
        let decay_t = decay.get(t)?.reshape((value_heads, 1, 1))?;
        state = state.broadcast_mul(&decay_t)?;
        let correction = (vt - state.matmul(&kt)?)?.broadcast_mul(&beta_t)?;
        state = (state + correction.matmul(&kt.t()?)?)?;
        outs.push(state.matmul(&qt)?.squeeze(2)?);
    }
    m.state = Some(state);
    finish(m, Tensor::stack(&outs, 0)?, &z, eps, len, value_heads, dim)
}

fn finish(m: &DeltaNet, output: Tensor, z: &Tensor, eps: f64, len: usize, heads: usize, dim: usize) -> Result<Tensor> {
    let output = rms(&output, &m.norm, eps)?;
    let gated = (output * z.reshape((len, heads, dim))?.silu()?)?.reshape((len, heads * dim))?;
    m.out.forward(&gated)
}
