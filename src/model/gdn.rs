//! Fused Gated DeltaNet recurrence as one CUDA launch per layer (llama.cpp: ggml_gated_delta_net).
//! Per head h and token t, with S (dv x dk):
//!   S = exp(g) * S;  d = beta * (v - S k);  S += d k^T;  o = S q

use candle_core::backend::BackendStorage;
use candle_core::cuda::{cudarc, CudaStorage};
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

pub const D: usize = 128;

const SRC: &str = r#"
#define D 128
extern "C" __global__ void gdn(const float* __restrict__ qk,
                               const float* __restrict__ v, const float* __restrict__ g,
                               const float* __restrict__ beta, const float* __restrict__ s_in,
                               float* __restrict__ out, int L, int H, int HK,
                               int qk_stride, int v_stride, int v_off) {
    const int h = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    __shared__ float sq[D], sk[D];
    float row[D];
    const float* sr = s_in + ((size_t)h * D + i) * D;
    #pragma unroll
    for (int j = 0; j < D; ++j) row[j] = sr[j];
    for (int t = 0; t < L; ++t) {
        const size_t base = ((size_t)t * H + h) * D;                       // output layout
        // packed qk: [q heads | k heads]; value head h reads key head h / (H / HK) (HF repeat_interleave)
        const size_t qbase = (size_t)t * qk_stride + (size_t)(h / (H / HK)) * D;
        const size_t kbase = qbase + (size_t)HK * D;
        const size_t vbase = (size_t)t * v_stride + v_off + (size_t)h * D;
        __syncthreads();
        for (int j = threadIdx.x; j < D; j += blockDim.x) { sq[j] = qk[qbase + j]; sk[j] = qk[kbase + j]; }
        __syncthreads();
        const float dec = expf(g[t * H + h]);
        const float b = beta[t * H + h];
        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < D; ++j) { row[j] *= dec; acc += row[j] * sk[j]; }
        const float d = (v[vbase + i] - acc) * b;
        float o = 0.f;
        #pragma unroll
        for (int j = 0; j < D; ++j) { row[j] += d * sk[j]; o += row[j] * sq[j]; }
        out[base + i] = o;
    }
    float* so = out + (size_t)L * H * D + ((size_t)h * D + i) * D;
    #pragma unroll
    for (int j = 0; j < D; ++j) so[j] = row[j];
}
"#;

const SRC_PRE: &str = r#"
#define D 128
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// causal depthwise conv over [state | x] + silu; also emits the next conv state
extern "C" __global__ void gdn_conv(const float* __restrict__ x, const float* __restrict__ st,
                                    const float* __restrict__ w, float* __restrict__ out,
                                    int L, int C, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * C) return;
    const int c = idx % C, t = idx / C;
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        const int ti = t + j - (K - 1);
        const float val = ti < 0 ? st[c * (K - 1) + (K - 1 + ti)] : x[(size_t)ti * C + c];
        acc += val * w[c * K + j];
    }
    out[(size_t)t * C + c] = silu(acc);
    if (t == 0) {
        float* so = out + (size_t)L * C + (size_t)c * (K - 1);
        for (int j = 0; j < K - 1; ++j) {
            const int ti = L - (K - 1) + j;
            so[j] = ti < 0 ? st[c * (K - 1) + (K - 1 + ti)] : x[(size_t)ti * C + c];
        }
    }
}

// l2-normalise each q and k head (llama.cpp build_gdn_l2_norm), scaling q by 1/sqrt(D)
extern "C" __global__ void gdn_qk(const float* __restrict__ conv, float* __restrict__ out,
                                  int L, int C, int HK, float eps, float qscale) {
    const int head = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x;                 // D == blockDim.x
    const int is_k = head >= HK;
    const int h = is_k ? head - HK : head;
    const float* src = conv + (size_t)t * C + (size_t)(is_k ? HK * D : 0) + (size_t)h * D;
    const float x = src[lane];
    __shared__ float red[D];
    red[lane] = x * x;
    __syncthreads();
    for (int s = D / 2; s > 0; s >>= 1) { if (lane < s) red[lane] += red[lane + s]; __syncthreads(); }
    const float n = sqrtf(red[0]);
    const float scale = (is_k ? 1.f : qscale) / fmaxf(n, eps);
    out[((size_t)t * 2 * HK + head) * D + lane] = x * scale;
}

// swiglu: silu(gate) * up, in one pass
extern "C" __global__ void swiglu(const float* __restrict__ gate, const float* __restrict__ up,
                                  float* __restrict__ out, int n, int unused) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = silu(gate[i]) * up[i];
}

// beta = sigmoid(.), g = softplus(. + dt_bias) * a
extern "C" __global__ void gdn_gate(const float* __restrict__ ba, const float* __restrict__ dt,
                                    const float* __restrict__ a, float* __restrict__ out,
                                    int L, int H) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * H) return;
    const int h = idx % H, t = idx / H;
    const float b = ba[(size_t)t * 2 * H + h];
    out[idx] = 1.f / (1.f + expf(-b));
    const float z = ba[(size_t)t * 2 * H + H + h] + dt[h];
    const float sp = z > 20.f ? z : log1pf(expf(z));
    out[(size_t)L * H + idx] = sp * a[h];
}
"#;

/// Applied to the state (H, D, D); q, k, v: (L, H, D), g, beta: (L, H), all f32 contiguous.
/// Output is flat: [o (L*H*D) | new state (H*D*D)].
struct Gdn {
    qk: Tensor,   // (L, 2*HK, D)
    conv: Tensor, // (L, C) — v read in place at v_off
    g: Tensor,
    beta: Tensor,
    h: usize,
    hk: usize,
    v_off: usize,
}

impl CustomOp1 for Gdn {
    fn name(&self) -> &'static str {
        "gated-delta-net"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gated-delta-net kernel is CUDA only")
    }

    fn cuda_fwd(&self, state: &CudaStorage, sl: &Layout) -> Result<(CudaStorage, Shape)> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let dev = state.device().clone();
        let (l, _, _) = self.qk.dims3()?;
        let (h, hk) = (self.h, self.hk);
        let c = self.conv.dim(1)?;
        let func = dev.get_or_load_custom_func("gdn", "semif_gdn", &ptx()?)?;

        macro_rules! view {
            ($t:expr) => {{
                let (st, lay) = $t.storage_and_layout();
                let s = match &*st {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.clone(),
                    _ => candle_core::bail!("expected cuda storage"),
                };
                (s, lay.start_offset())
            }};
        }
        let (q, qo) = view!(self.qk);
        let (v, vo) = view!(self.conv);
        let (g, go) = view!(self.g);
        let (b, bo) = view!(self.beta);
        let s = state.as_cuda_slice::<f32>()?;
        let n = l * h * D + h * D * D;
        let out = unsafe { dev.alloc::<f32>(n)? };
        let (li, hi, hki) = (l as i32, h as i32, hk as i32);
        let (qks, vs, vo_i) = ((2 * hk * D) as i32, c as i32, self.v_off as i32);
        let threads = 32u32;
        let cfg = LaunchConfig {
            grid_dim: (h as u32, (D as u32) / threads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        let (qv, vv, gv, bv, sv) = (
            q.slice(qo..),
            v.slice(vo..),
            g.slice(go..),
            b.slice(bo..),
            s.slice(sl.start_offset()..),
        );
        builder
            .arg(&qv)
            .arg(&vv)
            .arg(&gv)
            .arg(&bv)
            .arg(&sv)
            .arg(&out)
            .arg(&li)
            .arg(&hi)
            .arg(&hki)
            .arg(&qks)
            .arg(&vs)
            .arg(&vo_i);
        unsafe { builder.launch(cfg) }.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n)))
    }
}

fn ptx() -> Result<String> {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(p) = PTX.get() {
        return Ok(p.clone());
    }
    let opts = cudarc::nvrtc::CompileOptions { use_fast_math: Some(false), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(PTX.get_or_init(|| ptx.to_src()).clone())
}

/// Returns (o: (L, H, D), new_state: (H, D, D)). `conv` is the (L, C) conv output; v starts at v_off.
pub fn run(qk: &Tensor, conv: &Tensor, g: &Tensor, beta: &Tensor, state: &Tensor, h: usize, hk: usize, v_off: usize) -> Result<(Tensor, Tensor)> {
    let l = qk.dim(0)?;
    let op = Gdn {
        qk: qk.contiguous()?,
        conv: conv.contiguous()?,
        g: g.contiguous()?,
        beta: beta.contiguous()?,
        h,
        hk,
        v_off,
    };
    let flat = state.contiguous()?.apply_op1_no_bwd(&op)?;
    let o = flat.narrow(0, 0, l * h * D)?.reshape((l, h, D))?;
    let s = flat.narrow(0, l * h * D, h * D * D)?.reshape((h, D, D))?;
    Ok((o, s))
}


struct Pre {
    kind: u8, // 0 conv, 1 qk, 2 gate
    a: Tensor,
    b: Tensor,
    dims: (usize, usize, usize),
    fargs: (f32, f32),
}

impl CustomOp1 for Pre {
    fn name(&self) -> &'static str {
        "gdn-pre"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gdn-pre kernels are CUDA only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let dev = x.device().clone();
        let name = match self.kind {
            0 => "gdn_conv",
            1 => "gdn_qk",
            2 => "gdn_gate",
            _ => "swiglu",
        };
        let func = dev.get_or_load_custom_func(name, "semif_gdn_pre", &ptx_pre()?)?;
        macro_rules! view {
            ($t:expr) => {{
                let (st, lay) = $t.storage_and_layout();
                let s = match &*st {
                    candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.clone(),
                    _ => candle_core::bail!("expected cuda storage"),
                };
                (s, lay.start_offset())
            }};
        }
        let (a, ao) = view!(self.a);
        let (b, bo) = view!(self.b);
        let xs = x.as_cuda_slice::<f32>()?.slice(xl.start_offset()..);
        let (av, bv) = (a.slice(ao..), b.slice(bo..));
        let (d0, d1, d2) = self.dims;
        let (n, out_len) = match self.kind {
            0 => (d0 * d1, d0 * d1 + d1 * (d2 - 1)),
            1 => (d0 * 2 * d2, d0 * 2 * d2 * D),
            2 => (d0 * d1, 2 * d0 * d1),
            _ => (d0, d0),
        };
        let out = unsafe { dev.alloc::<f32>(out_len)? };
        let (i0, i1, i2) = (d0 as i32, d1 as i32, d2 as i32);
        let mut builder = func.builder();
        let cfg = if self.kind == 1 {
            LaunchConfig { grid_dim: (2 * d2 as u32, d0 as u32, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: 0 }
        } else {
            LaunchConfig::for_num_elems(n as u32)
        };
        let (f0, f1) = self.fargs;
        match self.kind {
            0 => {
                builder.arg(&xs).arg(&av).arg(&bv).arg(&out).arg(&i0).arg(&i1).arg(&i2);
            }
            1 => {
                builder.arg(&xs).arg(&out).arg(&i0).arg(&i1).arg(&i2).arg(&f0).arg(&f1);
            }
            2 => {
                builder.arg(&xs).arg(&av).arg(&bv).arg(&out).arg(&i0).arg(&i1);
            }
            _ => {
                builder.arg(&xs).arg(&av).arg(&out).arg(&i0).arg(&i1);
            }
        }
        unsafe { builder.launch(cfg) }.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(out_len)))
    }
}

fn ptx_pre() -> Result<String> {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(p) = PTX.get() {
        return Ok(p.clone());
    }
    let opts = cudarc::nvrtc::CompileOptions { use_fast_math: Some(false), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC_PRE, opts).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(PTX.get_or_init(|| ptx.to_src()).clone())
}

/// Causal depthwise conv + silu. Returns ((L, C) output, (C, K-1) next state).
pub fn conv(x: &Tensor, state: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (l, c) = x.dims2()?;
    let k = w.dim(1)?;
    let flat = x.contiguous()?.apply_op1_no_bwd(&Pre {
        kind: 0,
        a: state.contiguous()?,
        b: w.contiguous()?,
        dims: (l, c, k),
        fargs: (0., 0.),
    })?;
    Ok((flat.narrow(0, 0, l * c)?.reshape((l, c))?, flat.narrow(0, l * c, c * (k - 1))?.reshape((c, k - 1))?))
}

/// l2-normalised q and k heads packed as (L, 2*HK, D); q pre-scaled by 1/sqrt(D).
pub fn qk_norm(conv: &Tensor, hk: usize, eps: f64) -> Result<Tensor> {
    let (l, c) = conv.dims2()?;
    let flat = conv.contiguous()?.apply_op1_no_bwd(&Pre {
        kind: 1,
        a: conv.clone(),
        b: conv.clone(),
        dims: (l, c, hk),
        fargs: (eps as f32, 1.0 / (D as f32).sqrt()),
    })?;
    flat.reshape((l, 2 * hk, D))
}

/// silu(gate) * up.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let shape = gate.shape().clone();
    let n = shape.elem_count();
    let flat = gate.contiguous()?.apply_op1_no_bwd(&Pre {
        kind: 3,
        a: up.contiguous()?,
        b: up.contiguous()?,
        dims: (n, 0, 0),
        fargs: (0., 0.),
    })?;
    flat.reshape(shape)
}

/// (beta, g) from the packed beta|alpha projection.
pub fn gates(ba: &Tensor, dt_bias: &Tensor, a: &Tensor, h: usize) -> Result<(Tensor, Tensor)> {
    let l = ba.dim(0)?;
    let flat = ba.contiguous()?.apply_op1_no_bwd(&Pre {
        kind: 2,
        a: dt_bias.contiguous()?,
        b: a.contiguous()?,
        dims: (l, h, 0),
        fargs: (0., 0.),
    })?;
    Ok((flat.narrow(0, 0, l * h)?.reshape((l, h))?, flat.narrow(0, l * h, l * h)?.reshape((l, h))?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D as Dim};

    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, b: &Tensor, s0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (l, h, _) = q.dims3()?;
        let mut s = s0.clone();
        let mut outs = vec![];
        for t in 0..l {
            let qt = q.get(t)?.unsqueeze(2)?;
            let kt = k.get(t)?.unsqueeze(2)?;
            let vt = v.get(t)?.unsqueeze(2)?;
            s = s.broadcast_mul(&g.get(t)?.exp()?.reshape((h, 1, 1))?)?;
            let d = (vt - s.matmul(&kt)?)?.broadcast_mul(&b.get(t)?.reshape((h, 1, 1))?)?;
            s = (s + d.matmul(&kt.t()?)?)?;
            outs.push(s.matmul(&qt)?.squeeze(2)?);
        }
        Ok((Tensor::stack(&outs, 0)?, s))
    }

    #[test]
    fn kernel_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (l, h) = (37, 4);
        let unit = |t: Tensor| -> Result<Tensor> { t.broadcast_div(&t.sqr()?.sum_keepdim(Dim::Minus1)?.sqrt()?) };
        let q = unit(Tensor::randn(0f32, 1., (l, h, D), &dev)?)?;
        let k = unit(Tensor::randn(0f32, 1., (l, h, D), &dev)?)?;
        let v = Tensor::randn(0f32, 1., (l, h, D), &dev)?;
        let g = (Tensor::rand(0f32, 1., (l, h), &dev)? * -0.5)?;
        let b = Tensor::rand(0f32, 1., (l, h), &dev)?;
        let s0 = (Tensor::randn(0f32, 1., (h, D, D), &dev)? * 0.1)?;
        let qk = Tensor::cat(&[&q, &k], 1)?.contiguous()?; // (l, 2h, D)
        let conv = v.reshape((l, h * D))?.contiguous()?;
        let (o1, s1) = run(&qk, &conv, &g, &b, &s0, h, h, 0)?;
        let (o2, s2) = reference(&q, &k, &v, &g, &b, &s0)?;
        let eo = (o1 - o2)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let es = (s1 - s2)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("max err out {eo:e} state {es:e}");
        assert!(eo < 1e-3 && es < 1e-3);
        Ok(())
    }
}
