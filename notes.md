# Attention Is All You Need.

> *[Vaswani et al](https://arxiv.org/pdf/1706.03762)*

### Abstract

- Proposed Transformer Architecture
- Achieved 28.4 BLEU (understandable translations with significant grammar errors, see [link](https://cloud.google.com/translate/docs/advanced/automl-evaluate))

### Intro
- RNNs suck. 
  - Too long of a computation time, as is non-parallelizable
  - Batching is very limited given high memory constraints (assuming sequence length 100, we'd need approx 100xForwardPassMemory / 100xBackwardPassMemory to compute forward and backward pass )
  - Factoring Tricks and the non-dependence of MatMul on the $i$th component of a vector alongside the Attention Mechanism allows for parellization, enabling faster training and inference.
  - Reaches SOTA in 12 hours of 8xP100 training.
  
### Background

Unliked previous works, the number of operations the Transformer uses to relate features at an arbitrary token $t$ to another $\hat{t}$ is reduced to a constant number.

While this reduces expressiveness, given attention-pooling, Multi-Head Attention enables (akin to how a ConvNet has multiple kernels) the extraction of distinct features for a single sequence.

### Architecture

The encoder is composed of $6$ of the following blocks:

1. Multi-Head Attention
2. Add (residual connection)
3. LayerNorm
4. Positional Feed Forward (per token)
5. Add (residual connection)
6. LayerNorm

The residual connection is always computed prior to the LayerNorm.

Note that for a tensor $\in \mathbb{R}^{b \times l \times h_e}$ where $b$ is batch size, $l$ is sequence length, $h_e$ is embedding dim, the LayerNorm is computed per embedding vector per batch, individually.

This means for every hidden representation of a single token $t$,  $\vec{h}_t \in \mathbb{R}^{h_e}$, we compute statistics (mean and var) and normalize, both across this given vector of features. Not across multiple tokens in a given sequence not across the batch dimension (not batchnorm).

The decoder is structured of $6$ layer as well.

1. Masked Multi Head Attention
2. Add & Norm
3. Multi Head Attention (Q comes from previous layer and K, V come from the encoder)
4. Positional FNN.
5. Add & Norm

Repeat both 6 times -> Linear Layer -> Softmax.

Residual Connections in between each sublayer.

Attention mechanism is the scaled-dot product, by the dimension of $d_k$, as $\sqrt{d_k}$. 

The reasoning for this is that for a higher valued dot product, or equivalently a higher valued attention-weighting or higher valued softmax output, the attention mechanism assigns lower values to all other attention weights, minimizing their gradients for the componetns with lower valued attention weights.

Attention is computed as:

```math
\hspace{.25in} h_i = \sum_{t=0}^{l}\underbrace{\text{softmax} \left( \frac{(W_i^q q) \cdot (W_i^k k_t)^\top}{\sum_{j=1}^{l} \exp(W_i^q q \cdot (W_i^k k_j)^\top)} \right)}_{\alpha_{i,t}}(W_i^{v}v_t) \in \mathbb{R}^{h_v}
```

where Multi-Head attention then follows as:

```math
H_{\text{cat}} = \text{Concat}(h_1, h_2, \dots, h_n) \in \mathbb{R^{n \cdot h_v}}
```

They used $h = 8$ attention heads in parallel, where dimensions were as $d_k = d_v = \frac{d_{\text{model}}}{h} = 64$.

$d_{\text{model}}$ is the size of the input embeddings.

The Positional FNN contain two layers, where the hidden one has size 2048 while output is $512 = d_\text{model}$.

Positional Encoding,

```math
p_{i, 2j} = \sin\left(\frac{i}{1000^{\frac{2j}{d}}}\right)
\\[3mm]
p_{i, {2j + 1}} = \cos\left(\frac{i}{1000^{\frac{2j}{d}}}\right)
```

to encode diff positions

> see [here](https://vxnuaj.com/infinity/attention#:~:text=.-,Positional%20Encoding,-Assuming%20we%27re%20using) for a better explanation I wrote.

### Why Self-Attention

The amount of computations per layer, given the matmul can be parallelized on a significant number of GPUs.

> *Note that OpenAI trained GPT3 on 10k A100's*

The computational complexity is quadratic with respect to the length of input sequence for a given layer, $\mathcal{O}(n^2 \times d)$. For an RNN, per layer we have $\mathcal{O}(n \times d^2)$.

This can be interpreted as the transformer being more computationally efficient for tasks where sequence length is relatively small with respect to the dimensionality of a given layer.

The inverse is said for the RNN where a given layer is more computationally efficient if the dimensionality is significantly small with respect to $n$.

### Training

They trained on WMT 2014 English-German dataset, 4.5 million sentence pairs.

Encoded using byte pair encoding, to a total source vocabulary of about 37000 tokens. For English-French, they use a larter WMT 2014 English-French dataset of 36M sentences and split into 32000 vocabulary using word piece.

Each batch contained sentence pairs containing approximately 25k source tokens and 25k target tokens.

- 8 P100 GPUs. 
- For base models Trained for 100k steps or 12 hours.
- For larger models, trained for 300k steps or 3.5 days.

They run adam optimizer:
- $\beta_1 = .9$
- $\beta_2 = .98$
- $\epsilon = 10^{-9}$

Learning rate schedule is warmed up linearly for the first count of warm-up steps, then decreased proportional to the inverse square root of step number. Warm up steps is equal to $4000$

```math
\text{lr}(t) = d_{\text{model}}^{-.5} \times \min \left( t^{-0.5}, t \times \text{warmup\_steps}^{-1.5} \right)
```

Dropout is applied to the identity residual connections, before addition. Dropout is also applied to the positional encodings after the addition to the original embeddings.

```math
X_{\mathcal{P}} = \text{Dropout}(X + \mathcal{P}(X))
```

Label smoothing is applied with value of $\epsilon = .1$