# Mathematical Foundations of LoRA Kernel

> A self-contained reference for all mathematics implemented in the LoRA Kernel C++
> LLM training/serving framework. Each section presents the formula, its derivation,
> numerical considerations, the actual C++ implementation, and why the technique exists.

---

## Table of Contents

1. [Tensor Core Operations](#1-tensor-core-operations)
2. [Normalization](#2-normalization)
3. [Activation Functions](#3-activation-functions)
4. [Attention Mechanisms](#4-attention-mechanisms)
5. [Position Encodings](#5-position-encodings)
6. [Loss Functions](#6-loss-functions)
7. [Optimizers](#7-optimizers)
8. [Learning Rate Scheduling](#8-learning-rate-scheduling)
9. [Weight Initialization](#9-weight-initialization)
10. [Training Enhancements](#10-training-enhancements)
11. [Mixed Precision & Loss Scaling](#11-mixed-precision--loss-scaling)
12. [Quantization](#12-quantization)
13. [Numerical Gradient Checking](#13-numerical-gradient-checking)
14. [Memory Management](#14-memory-management)
15. [Model Merging & Pruning](#15-model-merging--pruning)
16. [Reinforcement Learning from Human Feedback](#16-reinforcement-learning-from-human-feedback)
17. [Knowledge Distillation](#17-knowledge-distillation)
18. [References](#18-references)

---

## 1. Tensor Core Operations

### 1.1 Strided Indexing

All tensors are stored as flat `float*` arrays with shape vector $\mathbf{s} \in \mathbb{Z}^d$
and stride vector $\mathbf{\sigma} \in \mathbb{Z}^d$.

**Formula:**
$$
\text{offset}(\mathbf{i}) = \sum_{k=0}^{d-1} i_k \cdot \sigma_k, \qquad
\sigma_k = \prod_{j=k+1}^{d-1} s_j
$$

**Implementation (`include/core/tensor.hpp:232`):**
```cpp
int64_t offset(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size())
        throw std::runtime_error("offset: dimension mismatch");
    int64_t off = 0;
    for (size_t i = 0; i < indices.size(); ++i)
        off += indices[i] * strides_[i];
    return off;
}
```

**Why row-major:** Contiguous memory = cache-friendly sequential access. C++ arrays are row-major natively.

### 1.2 Reshape

Changes shape without reallocating data (if element count matches):

```cpp
void reshape(const std::vector<int64_t>& new_shape) {
    int64_t n = 1;
    for (auto s : new_shape) n *= s;
    if (n != (int64_t)num_elements_) {
        deallocate_storage();
        shape_ = new_shape;
        compute_strides();
        allocate_storage((size_t)n);
    } else {
        shape_ = new_shape;
        strides_.resize(new_shape.size());
        if (!strides_.empty()) {
            strides_.back() = 1;
            for (int i = (int)new_shape.size() - 2; i >= 0; --i)
                strides_[i] = strides_[i + 1] * new_shape[i + 1];
        }
    }
}
```

**Why:** Avoids `memcpy` when logical shape changes but physical layout doesn't (e.g., batch × seq × dim).

### 1.3 Tiled Matrix Multiplication (GEMM)

CPU-optimized matmul with multi-level tiling for cache efficiency (`include/core/gemm_engine.hpp`):

| Level | Tile Size | Target Cache |
|-------|-----------|--------------|
| L3    | 256       | L3 (LLC)     |
| L2    | 128       | L2           |
| L1    | 32        | L1           |
| Register | 4      | Registers    |

```
for (int i3 = 0; i3 < M; i3 += L3)
  for (int j3 = 0; j3 < N; j3 += L3)
    for (int i2 = i3; i2 < min(i3+L3, M); i2 += L2)
      for (int j2 = j3; j2 < min(j3+L3, N); j2 += L2)
        for (int i1 = i2; i1 < min(i2+L2, M); i1 += L1)
          for (int j1 = j2; j1 < min(j2+L2, N); j1 += L1)
            for (int k = 0; k < K; ++k)
              for (int ri = i1; ri < min(i1+L1, M); ri += REG)
                for (int rj = j1; rj < min(j1+L1, N); rj += REG)
                  C[ri..ri+REG][rj..rj+REG] += A[ri..ri+REG][k] * B[k][rj..rj+REG]
```

**Why:** Naive triple-loop only uses ~5% of peak FLOPS on modern CPUs. Tiling keeps data in L1/L2 cache, achieving 60-80% of theoretical peak.

---

## 2. Normalization

### 2.1 RMSNorm (Root Mean Square Normalization)

#### Forward

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon},
\qquad y_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot w_i
$$

**Implementation (`include/core/tensor.hpp:394`):**
```cpp
inline void rmsnorm_forward(const float* x, float* y, int n,
                            const float* weight, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += (double)x[i] * x[i];
    float rms = (float)(std::sqrt(sum_sq / n + eps));
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; ++i)
        y[i] = (x[i] * inv_rms) * weight[i];
}
```

Note: uses `double` accumulator for `sum_sq` to avoid float32 overflow
($x_i^2$ can overflow for $|x_i| > 10^4$ in float32, whereas double handles $10^{154}$).

#### Backward Derivation

Let $r = \text{RMS}(\mathbf{x})$, $\hat{x}_i = x_i / r$, $y_i = \hat{x}_i \cdot w_i$.
Loss $L$ produces incoming gradient $g_i = \partial L / \partial y_i$.

We need $\partial L / \partial x_i$:

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

For $j \neq i$:
$$
y_j = \frac{x_j w_j}{r}, \quad
\frac{\partial y_j}{\partial x_i} = -\frac{x_j w_j}{r^2} \cdot \frac{\partial r}{\partial x_i}
$$

For $j = i$:
$$
\frac{\partial y_i}{\partial x_i} = \frac{w_i}{r} - \frac{x_i w_i}{r^2} \cdot \frac{\partial r}{\partial x_i}
$$

Now $\partial r / \partial x_i$:
$$
r^2 = \frac{1}{n}\sum_k x_k^2 + \epsilon \implies
2r\frac{\partial r}{\partial x_i} = \frac{2x_i}{n} \implies
\frac{\partial r}{\partial x_i} = \frac{x_i}{n r}
$$

Substituting:
$$
\begin{aligned}
\frac{\partial L}{\partial x_i} &= \frac{g_i w_i}{r}
- \frac{x_i}{n r^3} \sum_j g_j w_j x_j \\[4pt]
&= \frac{w_i}{r}\left(g_i - \frac{x_i}{n r^2} \sum_j g_j x_j\right) \\[4pt]
&= \frac{w_i}{r}\left(g_i - \frac{x_i \cdot \sum_j g_j x_j}{\sum_j x_j^2 + \epsilon n}\right)
\end{aligned}
$$

**Implementation (`include/core/tensor.hpp:402`):**
```cpp
inline void rmsnorm_backward(const float* grad_output, const float* x,
                             float* grad_x, const float* weight,
                             int n, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += (double)x[i] * x[i];
    float rms = (float)(std::sqrt(sum_sq / n + eps));
    float inv_rms = 1.0f / rms;
    double sum_gx = 0.0;
    for (int i = 0; i < n; ++i)
        sum_gx += (double)grad_output[i] * x[i];
    float scale = (float)(sum_gx / (sum_sq + eps * n));
    for (int i = 0; i < n; ++i)
        grad_x[i] = inv_rms * (grad_output[i] - x[i] * scale) * weight[i];
}
```

**Why RMSNorm vs LayerNorm:** RMSNorm omits the mean subtraction, saving $O(n)$ flops per normalization.
Empirically (Zhang & Sennrich, 2019), the mean-centering is unnecessary for Transformer models
— the RMS alone provides sufficient normalization. RMSNorm is used in Llama 2/3, Mistral, and Gemma.

#### Weight Gradient

For the learnable scale weight $w_d$:
$$
\frac{\partial L}{\partial w_d} = \sum_{b=1}^B \sum_{s=1}^S
\frac{\partial L}{\partial y_{bsd}} \cdot \hat{x}_{bsd}
$$

### 2.2 Softmax

#### Forward (Online Safe Variant)

$$
m = \max_j s_j, \quad
p_i = \frac{\exp(s_i - m)}{\sum_j \exp(s_j - m)}
$$

**Implementation (`include/core/tensor.hpp:365`):**
```cpp
inline void softmax_inplace(Tensor& X) {
    int64_t rows = X.numel() / X.size(X.ndim() - 1);
    int64_t cols = X.size(X.ndim() - 1);
    for (int64_t r = 0; r < rows; ++r) {
        float* row = X.data() + r * cols;
        float max_val = *std::max_element(row, row + cols);
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c)
            sum += std::exp((double)(row[c] - max_val));
        double inv_sum = 1.0 / (sum + 1e-12);
        for (int64_t c = 0; c < cols; ++c)
            row[c] = (float)(std::exp((double)(row[c] - max_val)) * inv_sum);
    }
}
```

**Numerical considerations:**
- Max-shift prevents `exp(1000)` overflow (float32 max ~ $3.4 \times 10^{38}$, `exp(89)` ≈ $10^{38}$).
- `double` accumulator in sum prevents catastrophic cancellation when many small probabilities sum to 1.
- `1e-12` epsilon prevents division by zero.

#### Backward

$$
d = \sum_j p_j \cdot g_j, \qquad
\frac{\partial L}{\partial s_i} = p_i \cdot (g_i - d)
$$

**Implementation (`include/core/tensor.hpp:380`):**
```cpp
inline void softmax_backward_inplace(const Tensor& softmax_out, Tensor& grad) {
    int64_t rows = softmax_out.numel() / softmax_out.size(softmax_out.ndim() - 1);
    int64_t cols = softmax_out.size(softmax_out.ndim() - 1);
    for (int64_t r = 0; r < rows; ++r) {
        const float* sm_row = softmax_out.data() + r * cols;
        float* grad_row = grad.data() + r * cols;
        double dot = 0.0;
        for (int64_t c = 0; c < cols; ++c)
            dot += (double)sm_row[c] * grad_row[c];
        for (int64_t c = 0; c < cols; ++c)
            grad_row[c] = sm_row[c] * (grad_row[c] - (float)dot);
    }
}
```

**Derivation:** For softmax output $p_i = e^{s_i} / \sum_j e^{s_j}$:

$$
\frac{\partial p_i}{\partial s_j} = p_i (\delta_{ij} - p_j)
$$

By the chain rule:
$$
\frac{\partial L}{\partial s_i} = \sum_j g_j \frac{\partial p_j}{\partial s_i}
= \sum_j g_j \cdot p_j (\delta_{ij} - p_i)
= p_i g_i - p_i \sum_j p_j g_j
= p_i(g_i - d)
$$

**Why:** The backward through softmax + cross-entropy simplifies to $p_i - y_i$
(when $g_i = \partial L_{\text{CE}} / \partial p_i$), which is why the training pipeline
computes `grad_logits[b,s,v] = softmax - one_hot(target)` directly.

### 2.3 LayerNorm (Fused Kernel)

$$
\mu = \frac{1}{n}\sum_i x_i, \qquad
\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2, \qquad
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

**When used:** As an alternative normalization in fused GEMM+Bias+Norm kernels for GPU-style execution
(`include/nn/fused_kernels.hpp`). RMSNorm is preferred for new models.

---

## 3. Activation Functions

### 3.1 GELU (Gaussian Error Linear Unit)

#### Motivation

The Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) is a smooth approximation of
$x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}(x / \sqrt{2})\right]
$$

The $\text{erf}$ function is expensive, so we use the tanh approximation (used in GPT-2, BERT):

#### Forward

$$
\text{GELU}(x) \approx 0.5\,x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)
$$

where $\sqrt{2/\pi} \approx 0.7978845608$.

**Implementation (`include/core/tensor.hpp:416`):**
```cpp
inline void gelu_forward(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    for (int i = 0; i < n; ++i) {
        float xv = x[i];
        float x3 = xv * xv * xv;
        float tanh_arg = sqrt_2_over_pi * (xv + 0.044715f * x3);
        y[i] = 0.5f * xv * (1.0f + std::tanh(tanh_arg));
    }
}
```

#### Backward Derivation

Let $a(x) = \sqrt{2/\pi}(x + 0.044715\,x^3)$:

$$
a'(x) = \sqrt{\frac{2}{\pi}} (1 + 3 \cdot 0.044715 \cdot x^2) = \sqrt{\frac{2}{\pi}} (1 + 0.134145\,x^2)
$$

$$
\frac{d}{dx}\text{GELU} = 0.5(1 + \tanh a) + 0.5\,x \cdot \text{sech}^2(a) \cdot a'(x)
$$

where $\text{sech}^2(a) = 1 - \tanh^2(a)$.

**Implementation (`include/core/tensor.hpp:425`):**
```cpp
inline void gelu_backward(const float* grad_output, const float* x,
                          float* grad_x, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    for (int i = 0; i < n; ++i) {
        float xv = x[i];
        float x3 = xv * xv * xv;
        float tanh_arg = sqrt_2_over_pi * (xv + coeff * x3);
        float tanh_val = std::tanh(tanh_arg);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float dtanh = sqrt_2_over_pi * (1.0f + 3.0f * coeff * xv * xv) * sech2;
        float dgelu = 0.5f * (1.0f + tanh_val) + 0.5f * xv * dtanh;
        grad_x[i] = grad_output[i] * dgelu;
    }
}
```

**Why GELU over ReLU:** GELU is smooth (differentiable everywhere) and has negative
values for $x < 0$, which allows gradient flow through negative activations. In practice,
GELU consistently outperforms ReLU by ~0.5-1% perplexity in LMs.

### 3.2 SwiGLU (Swish-Gated Linear Unit)

#### Forward

SwiGLU (Shazeer, 2020) gates one linear transform by another:

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \quad \text{(sigmoid)}
$$
$$
\text{Swish}_\beta(x) = x \cdot \sigma(\beta x) \quad (\beta = 1 \text{ in our impl})
$$
$$
\text{SwiGLU}(\mathbf{x}, \mathbf{y}) = \text{Swish}(\mathbf{x}) \odot \mathbf{y}
$$

In the FeedForward network ($D \to 4D \to D$ with SwiGLU, used in Llama, PaLM, Mistral):

$$
\mathbf{h} = \text{Swish}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})
$$
$$
\text{FFN}(\mathbf{x}) = \mathbf{h} \mathbf{W}_{\text{down}}
$$

**Implementation (`include/core/tensor.hpp:440`):**
```cpp
inline void swiglu_forward(const float* x, const float* y, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x[i]));
        float swish = x[i] * sig;
        out[i] = swish * y[i];
    }
}
```

#### Backward

$$
\frac{d\sigma}{dx} = \sigma(1 - \sigma), \quad
\frac{d}{dx}\text{Swish}(x) = \sigma + x \cdot \sigma(1 - \sigma)
$$

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \text{out}_i} \cdot y_i \cdot \frac{d}{dx}\text{Swish}(x_i)
\qquad
\frac{\partial L}{\partial y_i} = \frac{\partial L}{\partial \text{out}_i} \cdot \text{Swish}(x_i)
$$

**Implementation (`include/core/tensor.hpp:448`):**
```cpp
inline void swiglu_backward(const float* grad_out, const float* x,
                            const float* y, float* grad_x, float* grad_y, int n) {
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x[i]));
        float swish = x[i] * sig;
        float dsig = sig * (1.0f - sig);
        float dswish = sig + x[i] * dsig;
        grad_x[i] = grad_out[i] * y[i] * dswish;
        grad_y[i] = grad_out[i] * swish;
    }
}
```

**Why SwiGLU over plain GELU:** For the same hidden dimension, SwiGLU has ~50% more
parameters but significantly better quality. When matching parameter count (by reducing
FFN dimension from $4D$ to $\sim 2.7D$), SwiGLU still outperforms ReLU/GELU
(Shazeer, 2020; Llama paper).

### 3.3 ReLU and ELU

$$
\text{ReLU}(x) = \max(0, x), \qquad
\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

$$
\phi_{\text{ELU}+1}(x) = \begin{cases} x + 1 & x > 0 \\ e^x & x \leq 0 \end{cases}
$$

ELU+1 is used as a feature map for linear (Performer-style) attention — it ensures
strictly positive outputs (necessary for the denominator in linear attention).

---

## 4. Attention Mechanisms

### 4.1 Scaled Dot-Product Multi-Head Attention

#### Forward

Given input $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$:

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X}\mathbf{W}_q + \mathbf{b}_q, &
\mathbf{W}_q &\in \mathbb{R}^{D \times D} \\
\mathbf{K} &= \mathbf{X}\mathbf{W}_k + \mathbf{b}_k, &
\mathbf{W}_k &\in \mathbb{R}^{D \times D} \\
\mathbf{V} &= \mathbf{X}\mathbf{W}_v + \mathbf{b}_v, &
\mathbf{W}_v &\in \mathbb{R}^{D \times D}
\end{aligned}
$$

Split into $H$ heads: each head gets $D_h = D / H$ dimensions.

$$
\begin{aligned}
\mathbf{Q}_{bh} &\in \mathbb{R}^{S \times D_h}, \quad
\mathbf{K}_{bh} \in \mathbb{R}^{S \times D_h}, \quad
\mathbf{V}_{bh} \in \mathbb{R}^{S \times D_h} \\[4pt]
\mathbf{S}_{bh} &= \frac{\mathbf{Q}_{bh} \mathbf{K}_{bh}^\top}{\sqrt{D_h}} \in \mathbb{R}^{S \times S} \\[4pt]
\mathbf{S}_{bh} &\gets \mathbf{S}_{bh} + \text{causal\_mask} + \text{RoPE}(\mathbf{Q}, \mathbf{K}) \\[4pt]
\mathbf{A}_{bh} &= \text{softmax}(\mathbf{S}_{bh}) \in \mathbb{R}^{S \times S} \\[4pt]
\mathbf{C}_{bh} &= \mathbf{A}_{bh} \mathbf{V}_{bh} \in \mathbb{R}^{S \times D_h}
\end{aligned}
$$

Concatenate heads and project:
$$
\text{MHA}(\mathbf{X}) = \text{Concat}(\mathbf{C}_{b1}, \dots, \mathbf{C}_{bH})\,\mathbf{W}_o + \mathbf{b}_o
$$

**Implementation (simplified from `src/nn/transformer_blocks.cpp:9`):**
```cpp
void MultiHeadAttention::forward(const Tensor& input, Tensor& output) {
    int B = input.size(0), S = input.size(1), D = hidden_dim_;
    int H = num_heads_, Dh = head_dim_;
    // Q, K, V projections
    last_Q_.reshape({B, S, D}); last_K_.reshape({B, S, D}); last_V_.reshape({B, S, D});
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < D; ++d) {
                float sum_q = 0, sum_k = 0, sum_v = 0;
                for (int k = 0; k < D; ++k) {
                    float x = input.at({b, s, k});
                    sum_q += x * Wq_.at({k, d});
                    sum_k += x * Wk_.at({k, d});
                    sum_v += x * Wv_.at({k, d});
                }
                last_Q_.at({b, s, d}) = sum_q + bias_q_[d];
                last_K_.at({b, s, d}) = sum_k + bias_k_[d];
                last_V_.at({b, s, d}) = sum_v + bias_v_[d];
            }
    // Apply RoPE
    for (int b = 0; b < B; ++b)
        rope_forward(last_Q_.data() + b*S*D, last_K_.data() + b*S*D, S, D, 10000);
    // Split into heads and compute scaled dot-product
    Tensor Q_heads({B*H, S, Dh}), K_heads({B*H, S, Dh}), V_heads({B*H, S, Dh});
    // ... reshape + batch matmul + softmax + output projection
}
```

#### Scaling Factor $\sqrt{D_h}$

**Why $1/\sqrt{D_h}$?** Consider the dot product of two random vectors
$\mathbf{q}, \mathbf{k} \in \mathbb{R}^{D_h}$ with i.i.d. $\mathcal{N}(0, 1)$ entries:

$$
\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0, \quad
\text{Var}[\mathbf{q} \cdot \mathbf{k}] = D_h
$$

Without scaling, the variance grows with $D_h$, pushing scores into the flat tails of softmax
(vanishing gradients). Scaling by $1/\sqrt{D_h}$ restores unit variance:

$$
\text{Var}\!\left[\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{D_h}}\right] = 1
$$

#### MHA Backward (Gradient Flow)

For a complete Transformer block, gradients flow through:
1. Output projection $\mathbf{W}_o$: $\partial L / \partial \mathbf{C} = \partial L / \partial \text{MHA} \cdot \mathbf{W}_o^\top$
2. Softmax: $\partial L / \partial \mathbf{S} = \text{softmax} \cdot (\partial L / \partial \mathbf{C} \cdot \mathbf{V}^\top - \text{diag}(\text{softmax}^\top \cdot (\partial L / \partial \mathbf{C} \cdot \mathbf{V}^\top)))$
3. Scores: $\partial L / \partial \mathbf{Q} = (1/\sqrt{D_h}) \cdot \partial L / \partial \mathbf{S} \cdot \mathbf{K}$, similarly for $\mathbf{K}, \mathbf{V}$
4. Input projection: $\partial L / \partial \mathbf{X} = \partial L / \partial \mathbf{Q} \cdot \mathbf{W}_q^\top + \partial L / \partial \mathbf{K} \cdot \mathbf{W}_k^\top + \partial L / \partial \mathbf{V} \cdot \mathbf{W}_v^\top$

### 4.2 Causal Mask

Prevents each token from attending to future tokens (autoregressive constraint):

$$
\text{mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

**Implementation (`include/core/tensor.hpp:460`):**
```cpp
inline void apply_causal_mask(float* scores, int seq_len) {
    for (int i = 0; i < seq_len; ++i)
        for (int j = i + 1; j < seq_len; ++j)
            scores[i * seq_len + j] = -1e9f;
}
```

Setting masked positions to `-1e9` ensures $\exp(-1e9) \approx 0$ in softmax.

### 4.3 FlashAttention V2 (Online Safe Softmax)

#### Motivation

Standard attention materializes the $S \times S$ score matrix, requiring $O(S^2)$ memory.
For $S = 8192$, that's 256 MB per head — prohibitive. FlashAttention (Dao et al., 2022)
computes attention in tiles without materializing the full matrix.

#### Algorithm

Break $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into blocks of size $B_r, B_c$.
Process each $\mathbf{Q}_i$ block:

$$
\begin{aligned}
\mathbf{S}_{ij} &= \mathbf{Q}_i \mathbf{K}_j^\top / \sqrt{D_h} \\[4pt]
m^{\text{new}} &= \max(m^{\text{old}},\; \max(\mathbf{S}_{ij})) \\[4pt]
\ell^{\text{new}} &= \ell^{\text{old}} \cdot e^{m^{\text{old}} - m^{\text{new}}}
+ \sum_{k} e^{\mathbf{S}_{ij,k} - m^{\text{new}}} \\[4pt]
\alpha &= \ell^{\text{old}} \cdot e^{m^{\text{old}} - m^{\text{new}}} \\[4pt]
\text{norm} &= 1 / (\alpha + \ell^{\text{new}}) \\[4pt]
\mathbf{O} &\gets \big(\mathbf{O} \cdot \alpha + e^{\mathbf{S}_{ij} - m^{\text{new}}} \mathbf{V}_j\big) \cdot \text{norm} \\[4pt]
m^{\text{old}} &\gets m^{\text{new}}, \quad \ell^{\text{old}} \gets \ell^{\text{new}}
\end{aligned}
$$

**Implementation (`include/nn/flash_attention.hpp`):**
```cpp
inline void flash_attention_forward(const Tensor& Q, const Tensor& K,
                                    const Tensor& V, Tensor& O) {
    int B = ..., H = ..., S = ..., D = ...;
    const int Br = 32, Bc = 32;
    for (int ib = 0; ib < B; ++ib)
        for (int ih = 0; ih < H; ++ih)
            for (int i = 0; i < S; i += Br) {
                float m_old = -INFINITY, l_old = 0.0f;
                std::vector<float> O_i(Br * D, 0.0f);
                int q_end = std::min(i + Br, S);
                for (int j = 0; j < S; j += Bc) {
                    int k_end = std::min(j + Bc, S);
                    // S_ij = Q_i * K_j^T / sqrt(D)
                    for (int ri = i; ri < q_end; ++ri)
                        for (int cj = j; cj < k_end; ++cj) {
                            float s = 0;
                            for (int d = 0; d < D; ++d)
                                s += Q.at({ib, ih, ri, d}) * K.at({ib, ih, cj, d});
                            s /= std::sqrt((float)D);
                            // Apply causal mask
                            if (cj > ri) s = -INFINITY;
                            // Online safe softmax
                            float m_new = std::max(m_old, s);
                            float l_new = l_old * std::exp(m_old - m_new)
                                        + std::exp(s - m_new);
                            float rescale = l_old * std::exp(m_old - m_new);
                            float norm = 1.0f / (rescale + l_new);
                            for (int d = 0; d < D; ++d)
                                O_i[(ri - i) * D + d] =
                                    (O_i[(ri - i) * D + d] * rescale
                                     + std::exp(s - m_new) * V.at({ib, ih, cj, d})) * norm;
                            m_old = m_new;
                            l_old = l_new;
                        }
                }
                // Write O_i back to O
                for (int ri = i; ri < q_end; ++ri)
                    for (int d = 0; d < D; ++d)
                        O.at({ib, ih, ri, d}) = O_i[(ri - i) * D + d];
            }
}
```

**Memory:** $O(S \cdot D_h)$ per head (just Q, K, V blocks in SRAM), vs $O(S^2)$ for standard
attention. Enables 8× longer sequences on the same hardware.

**V2 improvement:** Reversed loop order (outer over K, inner over Q) reduces
SRAM reads/writes of O and ℓ by 2× vs v1.

### 4.4 Multi-Query & Grouped-Query Attention

#### MQA (Multi-Query Attention)

All $H$ query heads share one KV head:

$$
\mathbf{K} \in \mathbb{R}^{B \times 1 \times S \times D_h}, \quad
\mathbf{V} \in \mathbb{R}^{B \times 1 \times S \times D_h}
$$

$$
\text{scores}_{b,h,i,j} = \frac{\mathbf{Q}_{b,h,i} \cdot \mathbf{K}_{b,0,j}}{\sqrt{D_h}}
$$

#### GQA (Grouped-Query Attention)

$H$ query heads, $G$ KV heads (where $G < H$, typically $G \in \{2, 4, 8\}$).
Group $g = h \cdot G / H$. Each KV head serves $H/G$ query heads.

**Why:** MQA = minimal KV cache (reduces memory by $H$×). GQA = tunable middle-ground
— reduces KV cache by $H/G$× while preserving most of MHA's quality.
Used in Llama 2 70B (GQA), Llama 3 (GQA), Mistral (GQA), Falcon (MQA).

### 4.5 ALiBi (Attention with Linear Biases)

Rather than adding position embeddings to the input, ALiBi (Press et al., 2022)
adds a bias directly to the attention scores:

$$
\text{bias}_{h,i,j} = m_h \cdot (j - i), \qquad
m_h = \frac{1}{2^{h+1}}
$$

$$
\text{scores}'_{h,i,j} = \text{scores}_{h,i,j} + \text{bias}_{h,i,j}
$$

where $m_h$ is a head-specific slope. For $H=8$: $m = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]$.

**Advantage:** Extrapolates to sequences longer than training length without additional
training. No learned position parameters — purely inductive bias.

### 4.6 Linear Attention (Performer)

Uses kernel trick to avoid quadratic attention:

$$
\phi(x) = \text{ELU}(x) + 1 = \begin{cases} x + 1 & x > 0 \\ e^x & x \leq 0 \end{cases}
$$

$$
\mathbf{O} = \frac{\phi(\mathbf{Q})\big(\phi(\mathbf{K})^\top \mathbf{V}\big)}
{\phi(\mathbf{Q}) \sum_j \phi(\mathbf{K}_j)}
$$

By associativity: $\phi(\mathbf{K})^\top \mathbf{V}$ is $(D_h \times S) \times (S \times D_h) = O(D_h^2)$,
then $\phi(\mathbf{Q}) \times (D_h \times D_h) = O(S D_h^2)$, vs $O(S^2 D_h)$ for standard attention.

### 4.7 Attention Variants Summary

| Variant | Complexity | KV Cache | Quality | Use Case |
|---------|-----------|----------|---------|----------|
| MHA | $O(S^2 D)$ | $H \times S \times D_h$ | Best | Default |
| MQA | $O(S^2 D)$ | $1 \times S \times D_h$ | ~OK | High throughput |
| GQA | $O(S^2 D)$ | $G \times S \times D_h$ | Good | Balanced |
| FlashAttention | $O(S^2 D)$, $O(S)$ memory | N/A | Same as MHA | Long sequences |
| Sliding Window | $O(S W D)$, $W \ll S$ | Same | Good locally | Extreme length |
| Linear (Performer) | $O(S D^2)$ | N/A | Lower quality | Very long sequences |
| Sparse (Top-k) | $O(S k D)$ | N/A | Tunable | Custom |

---

## 5. Position Encodings

### 5.1 RoPE (Rotary Position Embedding)

#### Forward

RoPE (Su et al., 2021) encodes position by rotating query and key vectors:

For position $p$ and dimension pair $(d, d+1)$ with $d$ even:

$$
\theta_{p,d} = \frac{p}{\text{base}^{2d/D}}, \qquad \text{base} = 10000
$$

$$
\begin{pmatrix}
q'_{p,d} \\ q'_{p,d+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta_{p,d} & -\sin\theta_{p,d} \\
\sin\theta_{p,d} & \cos\theta_{p,d}
\end{pmatrix}
\begin{pmatrix}
q_{p,d} \\ q_{p,d+1}
\end{pmatrix}
$$

Same transformation applied to $\mathbf{K}$.

**Key property:** The dot product between query at position $i$ and key at position $j$
depends only on relative position $(i - j)$:

$$
\mathbf{Q}_i \cdot \mathbf{K}_j = \mathbf{q}_i^\top \mathbf{R}_{i-j} \mathbf{k}_j
$$

where $\mathbf{R}_{i-j}$ is a block-diagonal rotation matrix.

**Implementation (`include/core/tensor.hpp:466`):**
```cpp
inline void rope_forward(float* q, float* k, int seq_len, int head_dim, int base_freq) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int d = 0; d < head_dim; d += 2) {
            float theta = (float)pos / std::pow((float)base_freq,
                                                (float)d / (float)head_dim);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            float q0 = q[pos * head_dim + d];
            float q1 = q[pos * head_dim + d + 1];
            q[pos * head_dim + d]     = q0 * cos_t - q1 * sin_t;
            q[pos * head_dim + d + 1] = q0 * sin_t + q1 * cos_t;
            float k0 = k[pos * head_dim + d];
            float k1 = k[pos * head_dim + d + 1];
            k[pos * head_dim + d]     = k0 * cos_t - k1 * sin_t;
            k[pos * head_dim + d + 1] = k0 * sin_t + k1 * cos_t;
        }
    }
}
```

#### Backward

The rotation is orthogonal, so the backward pass applies the inverse rotation
(transpose = inverse for rotation matrices):

$$
\begin{pmatrix}
\frac{\partial L}{\partial q_d} \\ \frac{\partial L}{\partial q_{d+1}}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
g_d \\ g_{d+1}
\end{pmatrix}
$$

**Why RoPE over learned position embeddings:**
1. **Extrapolation:** RoPE can handle longer sequences than trained on.
2. **Relative bias:** The attention automatically decays with distance (rotation frequency varies by dimension).
3. **No extra parameters:** Unlike learned absolute PE.

### 5.2 ALiBi (Alternative)

See [Section 4.5](#45-alibi-attention-with-linear-biases).

---

## 6. Loss Functions

### 6.1 Cross-Entropy Loss

For logits $\mathbf{x} \in \mathbb{R}^V$ and target class $t \in \{1, \dots, V\}$:

$$
\mathcal{L}_{\text{CE}}(\mathbf{x}, t) = -\log\frac{\exp(x_t)}{\sum_{j=1}^V \exp(x_j)}
= -x_t + \log\sum_{j=1}^V \exp(x_j)
$$

**Gradient:**
$$
\frac{\partial \mathcal{L}}{\partial x_i} = \text{softmax}(x_i) - \delta_{i,t}
= p_i - \mathbf{1}_{\{i = t\}}
$$

**In the training pipeline (`include/nn/training_pipeline.hpp:70`):**
```cpp
// Forward: compute softmax and loss
for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float* row = logits.data() + (b * S + s) * V;
        float max_val = *std::max_element(row, row + V);
        double sum_exp = 0.0;
        for (int v = 0; v < V; ++v)
            sum_exp += std::exp((double)(row[v] - max_val));
        float inv_sum = (float)(1.0 / (sum_exp + 1e-12));
        // grad_logits = softmax
        for (int v = 0; v < V; ++v)
            grad_logits.at({b, s, v}) = std::exp((double)(row[v] - max_val)) * inv_sum;
        // subtract 1 from the correct class: grad = softmax - one_hot(target)
        int t = (int)targets.at({b, s});
        grad_logits.at({b, s, t}) -= 1.0f;
        // Negative log-likelihood (for monitoring)
        total_loss -= (double)(row[t] - max_val) - std::log(sum_exp + 1e-12);
    }
```

**Numerical stability:** Computing $-\log\sum\exp$ directly can overflow. We use
the identity:

$$
\log\sum\exp(x_j) = m + \log\sum\exp(x_j - m), \quad m = \max_j x_j
$$

### 6.2 Label Smoothing

Standard cross-entropy encourages the model to assign probability 1 to the correct
class, which can cause overfitting. Label smoothing (Szegedy et al., 2016) softens
the target distribution:

$$
q_i = \begin{cases}
1 - \alpha + \alpha / V & i = t \\
\alpha / V & i \neq t
\end{cases}
$$

$$
\mathcal{L}_{\text{LS}} = -\sum_{i=1}^V q_i \log p_i
$$

This is equivalent to adding KL divergence with a uniform distribution:

$$
\mathcal{L}_{\text{LS}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}} +
\alpha \cdot \text{KL}(\text{uniform} \parallel \mathbf{p})
$$

### 6.3 Knowledge Distillation

Distillation (Hinton et al., 2015) trains a student model to match a teacher's soft
outputs:

$$
\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot \text{KL}(
\text{softmax}(\mathbf{z}_t / T) \;\|\;
\text{softmax}(\mathbf{z}_s / T)
) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{z}_s, t)
$$

**Temperature $T$:** Higher $T$ produces softer probability distributions, revealing
more information about the teacher's "dark knowledge" (e.g., which classes are similar).

**Factor $T^2$:** Gradients scale as $1/T^2$, so multiplying by $T^2$ keeps the
gradient magnitude independent of temperature.

### 6.4 PPO (Proximal Policy Optimization) for RLHF

PPO (Schulman et al., 2017) is used for fine-tuning LMs with human feedback.

**Surrogate objective:**
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(
r_t \hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t
\right)\right]
$$

where $\hat{A}_t$ is the advantage estimate (how much better action $a_t$ is than average).

**PPO loss components:**
1. **Policy loss:** Clipped surrogate objective (prevents destructive large updates)
2. **Value loss:** MSE between value predictions and returns
3. **Entropy bonus:** $-\beta \cdot \sum \pi(a|s) \log \pi(a|s)$ (encourages exploration)

### 6.5 GRPO (Group Relative Policy Optimization)

GRPO (Shao et al., 2024) removes the value network by normalizing rewards within
a group:

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}, \quad
\mu_G = \frac{1}{G}\sum_{g=1}^G r_g, \quad
\sigma_G^2 = \frac{1}{G}\sum_{g=1}^G (r_g - \mu_G)^2
$$

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{g=1}^G \min\left(
\frac{\pi_\theta(y_g|x)}{\pi_{\theta_{\text{old}}}(y_g|x)} \hat{A}_g,\;
\text{clip}(\dots, 1-\epsilon, 1+\epsilon) \hat{A}_g
\right)
$$

**Why:** No need for a separate value model (saves ~1× model parameters in memory).

---

## 7. Optimizers

### 7.1 Adam (Adaptive Moment Estimation)

**Algorithm (Kingma & Ba, 2015):**

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Bias correction:** $m_0 = 0$, $v_0 = 0$ → initial estimates are biased toward zero.
Dividing by $1 - \beta^t$ corrects this.

**Why adaptive:** Each parameter has its own learning rate ($\eta / \sqrt{v_t}$),
so frequently updated parameters get smaller updates.

### 7.2 AdamW (Decoupled Weight Decay)

AdamW (Loshchilov & Hutter, 2019) fixes a bug in Adam's L2 regularization:

$$\boxed{\theta_t = \theta_{t-1} - \eta \left(\lambda \theta_{t-1} + \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right)}$$

**Key insight:** In standard Adam with L2 regularization ($\mathcal{L}' = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$),
the gradient includes $g_t + \lambda \theta_{t-1}$, which interacts poorly with the adaptive
learning rate (the regularization gets adapted too). AdamW decouples them:
weight decay is applied **after** the adaptive update, not mixed into the gradient.

**Implementation (`include/nn/training_pipeline.hpp:245`):**
```cpp
for (int64_t i = 0; i < n; ++i) {
    float g = grad.data()[i];
    int64_t idx = flat_offset + i;
    opt.m[idx] = beta1 * opt.m[idx] + (1.0f - beta1) * g;
    opt.v[idx] = beta2 * opt.v[idx] + (1.0f - beta2) * g * g;
    float m_hat = opt.m[idx] / bias_corr1;
    float v_hat = opt.v[idx] / bias_corr2;
    float param_val = param->data()[i];
    param_val -= lr_ * (weight_decay_ * param_val + m_hat / (std::sqrt(v_hat) + eps));
    param->data()[i] = param_val;
}
```

### 7.3 Gradient Clipping

Prevents gradient explosion by scaling down large gradients:

$$
\text{norm}_2 = \sqrt{\sum_i g_i^2}, \qquad
g_i' = \begin{cases}
g_i \cdot \frac{\text{max\_norm}}{\text{norm}_2 + \epsilon}
& \text{if } \text{norm}_2 > \text{max\_norm} \\
g_i & \text{otherwise}
\end{cases}
$$

**Why:** Without clipping, a single large gradient can destroy the model (parameters
jump to NaN). Typical max_norm values: 0.5–1.0 for Transformers.

### 7.4 Cosine LR Schedule with Warmup

**Warmup ($t < t_{\text{warm}}$):**
$$
\eta(t) = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{t_{\text{warm}}}
$$

**Cosine decay ($t \geq t_{\text{warm}}$):**
$$
\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
\left(1 + \cos\left(\pi \frac{t - t_{\text{warm}}}{T - t_{\text{warm}}}\right)\right)
$$

**Implementation (`include/nn/optimizer_and_loss.hpp:58`):**
```cpp
float get_lr(int step) {
    if (step < warmup_steps) {
        return min_lr + (max_lr - min_lr) * step / warmup_steps;
    }
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(M_PI * progress));
}
```

**Why warmup:** At the start of training, the Adam moments ($m, v$) are zero,
so the first few steps use un-corrected biased estimates. Warmup allows the
optimizer to stabilize. Without warmup, Transformers often diverge immediately.

### 7.5 SGD (Stochastic Gradient Descent)

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

Used only in the convergence test (`tests/test_convergence.cpp`) for verification.

---

## 8. Learning Rate Scheduling

### 8.1 Available Schedules

From `include/nn/training_enhancements.hpp`:

| Schedule | Formula |
|----------|---------|
| Linear | $\eta = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot (1 - p)$ |
| Cosine | $\eta = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi p))$ |
| Cosine Restarts | $\eta = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \cdot \gamma^{\lfloor p \cdot R \rfloor} (1 + \cos(\pi (p \cdot R - \lfloor p \cdot R \rfloor)))$ |
| Polynomial (p=2) | $\eta = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot (1 - p)^2$ |
| Constant | $\eta = \eta_{\max}$ |

where $p = (t - t_{\text{warm}}) / (T - t_{\text{warm}})$, $R$ = number of restarts, $\gamma$ = decay factor.

---

## 9. Weight Initialization

### 9.1 Kaiming (He) Normal

Designed for ReLU-family activations (He et al., 2015):

$$
\mathbf{W} \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)
$$

**Derivation:** For a layer $y = \mathbf{W}\mathbf{x}$ with ReLU, approximately half the
neurons are active. To keep $\text{Var}(y) = \text{Var}(x)$, we need
$\text{Var}(W) = 2 / \text{fan\_in}$.

**Implementation (`include/nn/weight_init.hpp:17`):**
```cpp
inline void kaiming_normal(Tensor& t, int fan_in) {
    float std = std::sqrt(2.0f / fan_in);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, std);
    for (int64_t i = 0; i < t.numel(); ++i)
        t.data()[i] = dist(gen);
}
```

### 9.2 Kaiming Uniform

$$
\mathbf{W} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{\text{fan\_in}}}, +\sqrt{\frac{6}{\text{fan\_in}}}\right)
$$

### 9.3 Xavier (Glorot) Normal

Optimal for tanh/sigmoid activations (Glorot & Bengio, 2010):

$$
\mathbf{W} \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)
$$

### 9.4 Xavier Uniform

$$
\mathbf{W} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}},
+\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)
$$

### 9.5 LoRA Initialization

For low-rank adaptation $\mathbf{W}' = \mathbf{W} + \mathbf{B} \cdot \mathbf{A} \cdot \text{scale}$:

- $\mathbf{A}$: Kaiming normal with $\text{fan\_in} = \text{in\_features}$
- $\mathbf{B}$: Zeros (so $\Delta \mathbf{W} = 0$ at initialization, preserving the base model)

---

## 10. Training Enhancements

### 10.1 Gradient Accumulation

Simulates larger batch size when memory limits micro-batch size:

```python
for step in range(num_micro_batches):
    loss = model(micro_batch)
    loss.backward()  # accumulates into .grad
if step % accumulation_steps == 0:
    optimizer.step()    # update with accumulated gradient
    optimizer.zero_grad()
```

$$
\mathbf{g}_{\text{total}} = \frac{1}{M}\sum_{m=1}^{M} \mathbf{g}_m,
\qquad \theta \gets \theta - \eta \cdot \text{optimizer}(\mathbf{g}_{\text{total}})
$$

**Why:** A single large batch may not fit in memory. Accumulation approximates
a batch of size $M \cdot \text{micro\_batch\_size}$.

### 10.2 SWA (Stochastic Weight Averaging)

SWA (Izmailov et al., 2018) averages weights from the last $k$ epochs/steps:

$$
\theta_{\text{SWA}} = \frac{1}{k}\sum_{i=N-k+1}^{N} \theta_i
$$

Incremental update (O(1) memory):
$$
n \gets n + 1, \qquad
\theta_{\text{SWA}} \gets \theta_{\text{SWA}} + \frac{\theta - \theta_{\text{SWA}}}{n}
$$

**Why:** Averaging traverses the loss landscape to find flatter minima, which
generalize better. SWA often matches or beats cosine annealing without extra
training cost.

### 10.3 EMA (Exponential Moving Average)

$$
\bar{\theta}_t = \beta \cdot \bar{\theta}_{t-1} + (1 - \beta) \cdot \theta_t
$$

Typical $\beta = 0.999$ (effective window of ~1000 steps).

**Why:** EMA acts as a temporal ensemble. The smoothed weights tend to be more
robust than the latest checkpoint, especially near the end of training.

### 10.4 Dropout (Inverted)

$$
\mathbf{m} \sim \text{Bernoulli}(1 - p), \qquad
\mathbf{y} = \frac{\mathbf{x} \odot \mathbf{m}}{1 - p}
$$

The $1/(1-p)$ scaling ensures $\mathbb{E}[y_i] = x_i$ at both train and test time
(test uses no mask but the same scale).

**Why dropout:** Prevents co-adaptation of neurons. Each neuron must learn
features that are useful even when other neurons are randomly dropped.

### 10.5 Stochastic Depth (Layer Dropout)

Randomly drop entire layers during training (Huang et al., 2016):

$$
p_{\text{survive}} = p_{\text{base}} \cdot \left(1 - \frac{\ell}{L}\right), \qquad
\mathbf{y} = \text{Bernoulli}(p_{\text{survive}}) \cdot \text{Layer}(\mathbf{x}) + \mathbf{x}
$$

Deeper layers are dropped more often (they learn redundant higher-level features).
At test time, all layers are active but scaled by $p_{\text{survive}}$.

### 10.6 Gradient Checkpointing (Activation Checkpointing)

Trade compute for memory: save only inputs at checkpointed layers; during backward,
recompute the forward pass from the nearest checkpoint.

**Memory:** $O(\sqrt{L})$ instead of $O(L)$ for $L$ layers.
**Compute:** ~33% overhead (one extra forward pass for each checkpoint segment).

### 10.7 Gradient Noise

Added Gaussian noise to gradients for exploration (Neelakantan et al., 2015):

$$
\mathbf{g}' = \mathbf{g} + \mathcal{N}\!\left(0, \frac{\sigma}{\sqrt{t+1}}\right)
$$

**Why:** Helps escape sharp minima and improves generalization, especially in RL.

### 10.8 Weight Tying

Share weights between input token embedding and output projection:

$$
\mathbf{W}_{\text{lm\_head}} = \mathbf{W}_{\text{embed}}^\top
$$

**Why:** Reduces parameters by $\sim V \cdot D$ (often millions) with minimal quality loss.
Used in ALBERT and TransformerXL.

---

## 11. Mixed Precision & Loss Scaling

### 11.1 FP16 Format

IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa bits.

| Format | Range | Precision |
|--------|-------|-----------|
| FP32 | $\pm 3.4 \times 10^{38}$ | ~7.2 decimal digits |
| FP16 | $\pm 6.5 \times 10^{4}$ | ~3.3 digits |
| BF16 | $\pm 3.4 \times 10^{38}$ | ~2.4 digits |

### 11.2 Dynamic Loss Scaling

FP16 can only represent values up to $6.5 \times 10^4$. Gradients smaller than
$6 \times 10^{-8}$ underflow to zero. Loss scaling amplifies the loss before
backward to keep gradients in the FP16 representable range.

**Algorithm:**
```
scale = INITIAL_SCALE (e.g., 2^16 = 65536)
for each step:
    loss = model(input)
    scaled_loss = loss * scale
    scaled_loss.backward()  # grads = grads * scale
    if any(grad has NaN/Inf):
        scale /= 2
        skip optimizer step (recompute with smaller scale)
    else:
        optimizer.step()  # optimizer unscales internally
        if no overflow for N steps:
            scale *= 2
```

**Implementation (`include/nn/loss_scaling.hpp`):**
```cpp
class DynamicLossScaler {
    float scale_{65536.0f};
    int steps_since_overflow_{0};
    static constexpr int GROWTH_INTERVAL = 2000;
public:
    float get_scale() const { return scale_; }
    void update(bool overflow) {
        if (overflow) {
            scale_ = std::max(1.0f, scale_ / 2.0f);
            steps_since_overflow_ = 0;
        } else {
            steps_since_overflow_++;
            if (steps_since_overflow_ >= GROWTH_INTERVAL) {
                scale_ *= 2.0f;
                steps_since_overflow_ = 0;
            }
        }
    }
};
```

### 11.3 FP8 E4M3 / E5M2

Two FP8 formats standardized for training/inference:

| Format | Sign | Exponent | Mantissa | Range | Best For |
|--------|------|----------|----------|-------|----------|
| E4M3 | 1 | 4 | 3 | $\pm 448$ | Weights, activations |
| E5M2 | 1 | 5 | 2 | $\pm 57344$ | Gradients |

FP8 training uses dual formats: E4M3 for forward (wider range for weights),
E5M2 for backward (wider range needed for gradients).

---

## 12. Quantization

### 12.1 Symmetric INT8 Quantization

$$
s = \frac{\max(|\mathbf{x}|)}{127}, \qquad
q = \text{round}\!\left(\frac{x}{s}\right), \qquad
\hat{x} = q \cdot s
$$

**Why 127 (not 128):** For symmetric signed 8-bit, the range is $[-128, 127]$.
Using 127 ensures symmetric mapping: $[-127, 127] \leftrightarrow [-s, s]$.

### 12.2 Per-Channel Quantization

Different scale per row of weight matrix:

$$
s_{r} = \frac{\max_c(|W_{rc}|)}{127}, \qquad
Q_{rc} = \text{round}(W_{rc} / s_r)
$$

**Why:** Each output channel has different sensitivity. Per-channel gives ~0.5%
better perplexity than per-tensor at INT8.

### 12.3 SmoothQuant (W8A8 Smoothing)

Activations have outliers that make INT8 quantization hard. SmoothQuant (Xiao et al., 2023)
migrates the quantization difficulty from activations to weights:

$$
s_c = \frac{\max(|\mathbf{A}_{:c}|)^\alpha}{\max(|\mathbf{W}_{c:}|)^{1-\alpha}}, \quad
s_c \in [0.1, 10] \text{ (clamped)}
$$

$$
\mathbf{W}'_{c:} = \frac{\mathbf{W}_{c:}}{s_c}, \qquad
\mathbf{A}'_{:c} = \mathbf{A}_{:c} \cdot s_c
$$

After smoothing, both $\mathbf{W}'$ and $\mathbf{A}'$ are easier to quantize
(no extreme outliers). The matmul result is unchanged:

$$
\mathbf{Y} = \mathbf{A} \mathbf{W}^\top = \mathbf{A}' \mathbf{W}'^\top
$$

### 12.4 GPTQ (Hessian-Based Quantization)

GPTQ (Frantar et al., 2023) performs layer-wise quantization using the Hessian
of the reconstruction error:

$$
\mathbf{H} = 2 \mathbf{X}^\top \mathbf{X} \quad \text{(from calibration data)}
$$

For each column $\mathbf{w}_{:q}$ being quantized:

1. Find $\text{Hessian}_{qq}^{-1}$ (from Cholesky decomposition of $\mathbf{H}$)
2. Quantize: $w_q = \text{round}(w_q / s)$
3. Compute error: $\delta = (w_q - w_q^{\text{quant}}) / \mathbf{H}_{qq}^{-1}$
4. Compensate: $\mathbf{w}' = \mathbf{w} + \delta \cdot \mathbf{H}_{:q}^{-1}$

This compensates the quantization error locally using the inverse Hessian,
making GPTQ near-optimal for 4-bit quantization.

### 12.5 AWQ (Activation-Aware Weight Quantization)

AWQ (Lin et al., 2024) observes that ~1% of weight channels are "salient"
(they handle important features). It scales weights by importance before quantization:

$$
s = \max(|\mathbf{X}|)^\alpha, \qquad
\mathbf{W}' = \mathbf{W} \cdot \text{diag}(s)^{-1}
$$

The scaling redistributes quantization error: less important channels get more error,
salient channels get less. After dequantization, the scaling is reversed.

### 12.6 INT4 Group-Wise Quantization

Group weights into blocks of size 32 and share scale:

$$
s_g = \frac{\max(|\mathbf{w}_g|)}{7}, \qquad
q_{g,i} = \text{round}(w_{g,i} / s_g), \quad q_{g,i} \in [-7, 7]
$$

Two 4-bit values packed per byte: `byte = (q[0] << 4) | q[1]`.

**Why 7 (not 8):** Symmetric INT4 range is $[-7, 7]$ (only 15 values, not 16).
Using 8 would make the range asymmetric.

### 12.7 KV Cache Quantization

During long-sequence inference, the KV cache dominates memory. Quantizing K/V
to FP8 or INT8 reduces memory by $4\times$:

```
k_scale = max_abs(k_block) / FP8_MAX
k_quant = round(k_block / k_scale)
```

Per-layer dynamic ranges (attention layers have different value distributions).

### 12.8 Mixed-Precision Quantization

Different layers get different bit widths based on sensitivity:

```
layer 0-3:   BF16  (early layers, very sensitive)
layer 4-7:   FP8   (middle layers)
layer 8-11:  INT4  (deep layers, most robust)
```

**Why:** Early layers process raw input → more sensitive. Deep layers process
abstract features → more quantization-tolerant.

---

## 13. Numerical Gradient Checking

### 13.1 Central Difference

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}
$$

**Error analysis:**
- **Truncation error:** $O(h^2)$ (central difference cancels $O(h)$ term)
- **Rounding error:** $O(\epsilon_{\text{mach}} / h)$ (subtraction of nearby values)

Total error ≈ $C_1 h^2 + C_2 \epsilon / h$. Optimal $h = (C_2 \epsilon / (2 C_1))^{1/3}$.
For float32 ($\epsilon \approx 10^{-7}$), $h \approx 10^{-4}$ balances both sources.

**Implementation (`tests/test_gradient_checker.cpp:45`):**
```cpp
for (int64_t i = 0; i < n && i < 50; ++i) {
    std::vector<float> x_plus = params;
    std::vector<float> x_minus = params;
    x_plus[i] += (float)h_;
    x_minus[i] -= (float)h_;
    float f_plus = f(x_plus);
    float f_minus = f(x_minus);
    float numerical = (f_plus - f_minus) / (2.0f * (float)h_);
    float analytical = analytical_grad[i];
    float abs_err = std::abs(numerical - analytical);
    float rel_err = abs_err / (std::abs(numerical) + 1e-8f);
    if (rel_err > rel_tol_ && abs_err > abs_tol_)
        failures++;  // gradient mismatch detected
}
```

### 13.2 When Is Gradient Checking Trustworthy?

| Situation | Expected Error | Notes |
|-----------|---------------|-------|
| Quadratic function | $\sim 10^{-5}$ | Central diff is exact for quadratics |
| Smooth activation (GELU) | $\sim 10^{-4}$ | Higher-order derivatives matter |
| At inflection points | $\sim 10^{-2}$ | $f'''(x)$ large, truncation error dominates |
| Near zero output | $\sim 10^{-1}$ | Float32 catastrophic cancellation |
| Very small gradients | High rel, low abs | Use absolute tolerance |

**Tolerances used:** `rel_tol = 5e-2`, `abs_tol = 1e-4`, `h = 1e-3`.

---

## 14. Memory Management

### 14.1 LRU Weight Cache

**Algorithm:** Hash map + doubly-linked list for O(1) get/put/evict.

```
On weight lookup (key):
    if key in cache:
        move node to head of list
        return value
    else:
        if cache full:
            evict tail node (least recently used)
        create new node at head
        insert into hash map
        return value
```

**Why:** Inference servers often have many concurrent requests with different
model versions. LRU keeps frequently used weights in memory while evicting
stale ones.

### 14.2 Memory Pool (Linear Allocator)

Pre-allocates a large region and sub-allocates linearly:

```cpp
class MemoryPool {
    char* base_;
    size_t offset_{0};
    size_t capacity_;
public:
    void* alloc(size_t n) {
        n = (n + 63) & ~63;  // 64-byte alignment
        if (offset_ + n > capacity_) return nullptr;
        void* ptr = base_ + offset_;
        offset_ += n;
        return ptr;
    }
    void reset() { offset_ = 0; }
    void free(void*) {}  // no-op, reset when done
};
```

**Why:** `malloc`/`free` are slow and cause fragmentation. Linear allocators
are O(1) and fragmentation-free.

### 14.3 NUMA-Aware Allocation

Non-Uniform Memory Access (NUMA) means accessing memory on a remote socket is
slower than local memory. We detect topology and bind allocations:

```cpp
#ifdef _WIN32
    // Windows: GetLogicalProcessorInformationEx + VirtualAllocExNuma
    hProcess = GetCurrentProcess();
    GetLogicalProcessorInformationEx(RelationNumaNode, ...);
    // Bind thread to socket via SetThreadAffinityMask
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << socket);
    // Allocate on specific node
    VirtualAllocExNuma(hProcess, NULL, size,
                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node);
#elif defined(__linux__)
    // Linux: sched_setaffinity + mbind
    CPU_ZERO(&mask); CPU_SET(socket, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    mbind(ptr, size, MPOL_BIND, &node_mask, 64, MPOL_MF_STRICT);
#endif
```

**Why:** Without NUMA binding, a thread on socket 0 may allocate on socket 1's
memory (2× latency). Binding improves throughput by 15-30% on multi-socket systems.

### 14.4 Guard Pages

Protect against stack overflow by allocating a guard page at the end of the stack:

```cpp
// Windows: PAGE_GUARD on the last page of a reserved region
SYSTEM_INFO si; GetSystemInfo(&si);
void* stack = VirtualAlloc(NULL, stack_size + si.dwPageSize,
                           MEM_RESERVE, PAGE_NOACCESS);
VirtualAlloc(stack, stack_size, MEM_COMMIT, PAGE_READWRITE);
VirtualProtect((char*)stack + stack_size, si.dwPageSize,
               PAGE_GUARD | PAGE_NOACCESS, &old);
```

Accessing the guard page triggers `STATUS_GUARD_PAGE_VIOLATION`, caught by
the signal handler for graceful recovery.

---

## 15. Model Merging & Pruning

### 15.1 Linear Model Merging

$$
\theta_{\text{merged}} = \lambda \cdot \theta_1 + (1 - \lambda) \cdot \theta_2
$$

Used for model soup (Wortsman et al., 2022) — averaging fine-tuned checkpoints.

### 15.2 Task Arithmetic

$$
\theta_{\text{merged}} = \theta_{\text{base}} + \beta \cdot (\theta_{\text{ft}} - \theta_{\text{base}})
$$

Add the "task vector" (fine-tuning delta) to a base model. Allows composing
multiple skills: $\theta_{\text{combo}} = \theta_{\text{base}} + \sum_i \beta_i \cdot \tau_i$.

### 15.3 TIES Merging (Trim, Elect, Sign)

**Trim:** Keep only top-$k$% of task vectors (by magnitude).
**Elect:** Resolve sign conflicts between tasks (use majority vote).
**Sign:** Average only parameters with consistent sign across tasks.

**Why:** Task vectors often conflict (one task wants to increase weight, another
wants to decrease). TIES resolves these conflicts.

### 15.4 Magnitude Pruning

$$
\text{mask}_i = \begin{cases}
0 & |w_i| < \text{percentile}(|\mathbf{w}|, p) \\
1 & \text{otherwise}
\end{cases}
$$

### 15.5 Structured (Channel) Pruning

$$
\text{norm}_r = \sqrt{\sum_c W_{rc}^2}, \quad
\text{mask}_r = \begin{cases}
0 & \text{norm}_r < \text{threshold} \\
1 & \text{otherwise}
\end{cases}
$$

Removes entire output channels, which enables actual speedup (unlike unstructured
pruning which requires sparse hardware support).

---

## 16. Reinforcement Learning from Human Feedback

### 16.1 PPO for Language Models

The policy $\pi_\theta$ generates tokens. The reward model $R_\phi$ scores
generations. PPO optimizes:

$$
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}} \Bigg[
\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} A(x,y)
- \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
\Bigg]
$$

The KL penalty keeps the policy close to the reference model (preventing
reward hacking).

### 16.2 GRPO (Group Relative PPO)

GRPO removes the value network by computing advantages within a group of
$G$ sampled outputs:

$$
A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}, \quad
\mu_G = \frac{1}{G}\sum_{g=1}^G r_g, \quad
\sigma_G^2 = \frac{1}{G}\sum_{g=1}^G (r_g - \mu_G)^2
$$

---

## 17. Knowledge Distillation

### 17.1 Logit-Based Distillation

$$
\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}(
\sigma(\mathbf{z}_t / T) \;\|\; \sigma(\mathbf{z}_s / T)
) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{z}_s, y)
$$

where $\sigma$ is softmax and $y$ is the hard label.

**Temperature $T$:** Controls softness. Higher $T$ reveals more structure in
the teacher's output distribution (e.g., "cat" is closer to "dog" than to "car").

### 17.2 Feature-Based Distillation

Also match intermediate representations:

$$
\mathcal{L}_{\text{feat}} = \sum_{\ell \in \text{hints}}
\|\mathbf{h}_{\ell}^{\text{student}} - \mathbf{W}_{\text{proj}} \mathbf{h}_{\ell}^{\text{teacher}}\|^2_2
$$

**Why:** Matching hidden states often produces better student models than
matching only logits.

---

## 18. CPU Feature Detection

Runtime dispatch for SIMD-optimized paths:

```cpp
struct CPUFeatures {
    bool avx  : 1;   // Advanced Vector Extensions (128-bit)
    bool avx2 : 1;   // AVX2 (256-bit integer)
    bool avx512f : 1;     // AVX-512 Foundation
    bool avx512dq : 1;    // AVX-512 Double/Quad
    bool avx512bw : 1;    // AVX-512 Byte/Word
    bool avx512vl : 1;    // AVX-512 Vector Length
    bool neon : 1;        // ARM NEON (128-bit SIMD)
    bool sve  : 1;        // ARM SVE (scalable vectors)

    static CPUFeatures detect() {
        CPUFeatures f = {};
#ifdef _WIN32
        int cpuInfo[4] = {};
        __cpuidex(cpuInfo, 0, 0);
        if (cpuInfo[0] >= 1) {
            __cpuidex(cpuInfo, 1, 0);
            f.avx = (cpuInfo[2] & (1 << 28)) != 0;
        }
        if (cpuInfo[0] >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            f.avx2     = (cpuInfo[1] & (1 << 5)) != 0;
            f.avx512f  = (cpuInfo[1] & (1 << 16)) != 0;
            f.avx512dq = (cpuInfo[1] & (1 << 17)) != 0;
            f.avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
            f.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
        }
#endif
        return f;
    }
};
```

**Why:** Auto-vectorization by compilers is unreliable. Detecting features
at runtime allows dispatching to hand-tuned SIMD kernels on compatible CPUs.

---

## References

1. Zhang & Sennrich (2019). "Root Mean Square Layer Normalization." *NeurIPS*.
2. Hendrycks & Gimpel (2016). "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*.
3. Shazeer (2020). "GLU Variants Improve Transformer." *arXiv:2002.05202*.
4. Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.
5. Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." *NeurIPS*.
6. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
7. Press et al. (2022). "Train Short, Test Long: Attention with Linear Biases." *NeurIPS*.
8. Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
9. Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization." *ICLR*.
10. He et al. (2015). "Delving Deep into Rectifiers." *ICCV*.
11. Szegedy et al. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*.
12. Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI*.
13. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
14. Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization." *ICML*.
15. Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." *ICLR*.
16. Lin et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys*.
17. Wortsman et al. (2022). "Model Soups: Averaging Weights of Multiple Fine-Tuned Models." *NeurIPS*.
18. Hinton et al. (2015). "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531*.
